"""Training utilities and loss computation."""
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

from config import TrainingConfig
from tokenizer import CharTokenizer

class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_lr: float,
        total_steps: int
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.step_num = 0
    
    def step(self):
        """Update learning rate."""
        self.current_step += 1
        self.step_num += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing phase
            progress = (self.current_step - self.warmup_steps) / \
                      (self.total_steps - self.warmup_steps)
            lr = 0.5 * self.max_lr * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            "current_step": self.current_step,
            "step_num": self.step_num,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load scheduler state."""
        self.current_step = state_dict["current_step"]
        self.step_num = state_dict.get("step_num", self.current_step)
        self.warmup_steps = state_dict["warmup_steps"]
        self.max_lr = state_dict["max_lr"]
        self.total_steps = state_dict["total_steps"]


def compute_ctc_weight(
    epoch: int,
    total_epochs: int,
    config: TrainingConfig
) -> float:
    """Reference-style CTC weight schedule with multiple kinds."""
    start = config.ctc_weight
    end = config.ctc_weight_end
    hold = config.ctc_hold_epochs
    kind = (config.ctc_weight_kind or "cosine").lower()

    if epoch <= hold:
        return float(start)

    span = max(1, total_epochs - hold)
    t = min(max(epoch - hold, 0), span) / span  # 0..1

    if kind == "linear":
        w = start + (end - start) * t
    elif kind == "cosine":
        w = end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))
    elif kind == "exp":
        k = 5.0
        w = end + (start - end) * math.exp(-k * t)
    elif kind == "poly":
        p = 2.0
        w = end + (start - end) * (1 - t) ** p
    elif kind == "step":
        steps = [0.33, 0.66]
        vals  = [0.55, 0.40]
        w = start
        for s, v in zip(steps, vals):
            if t >= s:
                w = v
        w = max(min(w, start), end)
    elif kind == "sigmoid":
        k = 10.0
        w = end + (start - end) / (1 + math.exp(k * (t - 0.5)))
    elif kind == "cyclic":
        cycles = 2
        tt = (t * cycles) % 1.0
        w = start + (end - start) * abs(2 * tt - 1)
    else:
        w = start + (end - start) * t

    return float(w)

class LossComputer:
    """Compute joint CTC-Attention loss."""
    
    def __init__(self, tokenizer: CharTokenizer):
        self.tokenizer = tokenizer
        self.ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    def compute(
        self,
        model_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ctc_weight: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute joint loss.
        
        Args:
            model_outputs: (ctc_logits, enc_lengths, attn_logits)
            targets: (Y_attn, y_attn_len, Y_ctc, y_ctc_len)
            ctc_weight: Weight for CTC loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        ctc_logits, enc_lengths, attn_logits = model_outputs
        Y_attn, y_attn_len, Y_ctc, y_ctc_len = targets
        
        # CTC loss
        log_probs = nn.functional.log_softmax(ctc_logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)  # [T, B, V]
        
        loss_ctc = self.ctc_loss(
            log_probs,
            Y_ctc,
            enc_lengths.to(torch.long),
            y_ctc_len.to(torch.long)
        )
        
        # Attention loss
        # Shift targets for teacher forcing
        ys_out = Y_attn[:, 1:]  # Remove <sos>
        assert attn_logits.size(1) == ys_out.size(1), \
            f"Decoder length mismatch: {attn_logits.size(1)} vs {ys_out.size(1)}"
        loss_attn = self.ce_loss(
            attn_logits.reshape(-1, attn_logits.size(-1)),
            ys_out.reshape(-1)
        )
        
        # Combined loss
        total_loss = ctc_weight * loss_ctc + (1 - ctc_weight) * loss_attn
        
        return total_loss, {
            "ctc": loss_ctc.item(),
            "attn": loss_attn.item(),
            "total": total_loss.item()
        }


class Trainer:
    """Training manager."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        loss_computer: LossComputer,
        config: TrainingConfig,
        writer: Optional[SummaryWriter] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_computer = loss_computer
        self.config = config
        self.writer = writer
        self.global_step = 0
    
    def train_step(
        self,
        batch: tuple,
        ctc_weight: float
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Unpack batch
        X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
        X = X.to(self.config.device)
        x_len = x_len.to(self.config.device)
        Y_attn = Y_attn.to(self.config.device)
        Y_ctc = Y_ctc.to(self.config.device)
        
        # Forward pass
        ys_in = Y_attn[:, :-1]  # Remove </s> for input
        outputs = self.model(X, x_len, ys_in)
        
        # Compute loss
        targets = (Y_attn, y_attn_len, Y_ctc, y_ctc_len)
        loss, loss_dict = self.loss_computer.compute(outputs, targets, ctc_weight)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        
        # Logging
        if self.writer:
            self.writer.add_scalar("train/loss", loss_dict["total"], self.global_step)
            self.writer.add_scalar("train/ctc_loss", loss_dict["ctc"], self.global_step)
            self.writer.add_scalar("train/attn_loss", loss_dict["attn"], self.global_step)
            self.writer.add_scalar("train/lr", self.scheduler.get_lr(), self.global_step)
        
        self.global_step += 1
        
        return loss_dict