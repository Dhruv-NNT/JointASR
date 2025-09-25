"""Evaluation utilities and metrics."""

import torch
from typing import List, Dict, Tuple
from evaluate import load as load_metric

from tokenizer import CharTokenizer

class Evaluator:
    """Model evaluation utilities."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: CharTokenizer,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.wer_metric = load_metric("wer")
    
    @torch.no_grad()
    def decode_ctc_greedy(
        self,
        features: torch.Tensor,
        feat_lengths: torch.Tensor
    ) -> List[str]:
        """
        Greedy CTC decoding.
        
        Args:
            features: Input features [B, T, D]
            feat_lengths: Valid lengths [B]
            
        Returns:
            List of decoded texts
        """
        self.model.eval()
        
        # Get encoder output
        enc_out, enc_lengths, _ = self.model.forward_encoder(features, feat_lengths)
        
        # Get CTC predictions
        logits = self.model.ctc_head(enc_out)
        predictions = logits.argmax(dim=-1)  # [B, T]
        
        # Decode each sequence
        texts = []
        for b in range(predictions.size(0)):
            length = int(enc_lengths[b].item())
            seq = predictions[b, :length]
            
            # Remove blanks and collapse repeats
            decoded = []
            prev = None
            for token_id in seq.tolist():
                if token_id == self.tokenizer.blank_id:
                    prev = token_id
                    continue
                if token_id == prev:
                    continue
                decoded.append(token_id)
                prev = token_id
            
            texts.append(self.tokenizer.ids2text(decoded))
        
        return texts
    
    @torch.no_grad()
    def decode_attention_greedy(
        self,
        features: torch.Tensor,
        feat_lengths: torch.Tensor,
        max_length: int = 256
    ) -> List[str]:
        """
        Greedy attention decoding.
        
        Args:
            features: Input features [B, T, D]
            feat_lengths: Valid lengths [B]
            max_length: Maximum decode length
            
        Returns:
            List of decoded texts
        """
        self.model.eval()
        
        # Get encoder output
        enc_out, _, enc_mask = self.model.forward_encoder(features, feat_lengths)
        
        batch_size = features.size(0)
        device = features.device
        
        # Initialize with SOS token
        ys = torch.full(
            (batch_size, 1), 
            self.tokenizer.sos_id, 
            dtype=torch.long, 
            device=device
        )
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Decode step by step
        for _ in range(max_length):
            logits = self.model.forward_decoder(enc_out, enc_mask, ys)
            next_tokens = logits[:, -1, :].argmax(dim=-1)  # [B]
            
            ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)
            
            # Check for EOS
            finished |= (next_tokens == self.tokenizer.eos_id)
            if finished.all():
                break
        
        # Convert to text
        texts = []
        for b in range(batch_size):
            seq = ys[b, 1:]  # Remove SOS
            
            # Truncate at EOS
            eos_positions = (seq == self.tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                seq = seq[:eos_positions[0]]
            
            texts.append(self.tokenizer.ids2text(seq.tolist()))
        
        return texts
    
    def compute_wer(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Word Error Rate.
        
        Args:
            predictions: Predicted texts
            references: Reference texts
            
        Returns:
            WER score (0-1)
        """
        # Normalize texts
        predictions = [self._normalize_text(p) for p in predictions]
        references = [self._normalize_text(r) for r in references]
        
        return float(self.wer_metric.compute(
            predictions=predictions,
            references=references
        ))
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for WER computation."""
        return " ".join(text.lower().strip().split())
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: tuple,
        decode_method: str = "ctc"
    ) -> Dict[str, float]:
        """
        Evaluate a single batch.
        
        Args:
            batch: Data batch
            decode_method: "ctc" or "attention"
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        # Unpack batch
        X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
        X = X.to(self.device)
        x_len = x_len.to(self.device)
        
        # Decode
        if decode_method == "ctc":
            predictions = self.decode_ctc_greedy(X, x_len)
        else:
            predictions = self.decode_attention_greedy(X, x_len)
        
        # Get references
        references = []
        for b in range(Y_ctc.size(0)):
            length = int(y_ctc_len[b].item())
            ids = Y_ctc[b, :length].tolist()
            references.append(self.tokenizer.ids2text(ids))
        
        # Compute WER
        wer = self.compute_wer(predictions, references)
        
        return {
            "wer": wer,
            "predictions": predictions,
            "references": references
        }
    
@torch.no_grad()
def compute_wer_over_loader(model, loader, tokenizer: CharTokenizer, decode: str = "ctc", device: str = "cuda") -> float:
    """Compute average WER over an entire loader (decode='ctc' or 'attention')."""
    metric = load_metric("wer")
    model.eval()
    for X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len in loader:
        X, x_len = X.to(device), x_len.to(device)
        if decode == "ctc":
            preds = Evaluator(model, tokenizer, device).decode_ctc_greedy(X, x_len)
        else:
            preds = Evaluator(model, tokenizer, device).decode_attention_greedy(X, x_len)
        refs = []
        for b in range(Y_ctc.size(0)):
            L = int(y_ctc_len[b].item())
            refs.append(tokenizer.ids2text(Y_ctc[b, :L].tolist()))
        preds = [" ".join(p.lower().strip().split()) for p in preds]
        refs  = [" ".join(r.lower().strip().split()) for r in refs]
        metric.add_batch(predictions=preds, references=refs)
    return float(metric.compute())

@torch.no_grad()
def batch_wer_and_one_sample(model, batch, tokenizer: CharTokenizer, device: str = "cuda", max_len: int = 128):
    """Compute CTC+Attention WER for THIS batch, and return one sample strings {ref, ctc, attn}."""
    model.eval()
    X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len = batch
    X, x_len = X.to(device), x_len.to(device)
    ev = Evaluator(model, tokenizer, device)
    ctc_preds  = ev.decode_ctc_greedy(X, x_len)
    attn_preds = ev.decode_attention_greedy(X, x_len, max_length=max_len)
    refs = []
    for b in range(Y_ctc.size(0)):
        L = int(y_ctc_len[b].item())
        refs.append(tokenizer.ids2text(Y_ctc[b, :L].tolist()))
    norm = lambda xs: [" ".join(s.lower().strip().split()) for s in xs]
    wer = load_metric("wer")
    wer_ctc  = float(wer.compute(references=norm(refs), predictions=norm(ctc_preds)))
    wer_attn = float(wer.compute(references=norm(refs), predictions=norm(attn_preds)))

    import random
    i = random.randrange(len(refs))
    sample = {"ref": refs[i], "ctc": ctc_preds[i], "attn": attn_preds[i]}
    return wer_ctc, wer_attn, sample
