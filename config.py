"""Configuration classes for ASR training."""

import os
from dataclasses import dataclass
from typing import Optional, Literal
import torch

@dataclass
class MelConfig:
    """Mel-spectrogram configuration."""
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 400
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 7600.0
    power: float = 2.0
    top_db: Optional[float] = 80.0
    center: bool = True
    pad_mode: str = "reflect"
    mel_norm: Optional[str] = "slaney"
    mel_scale: str = "slaney"

@dataclass
class HFRobustConfig:
    """HuggingFace Wav2Vec2 feature extraction configuration."""
    model_name: str = "aware-ai/wav2vec2-xls-r-300m-english"
    layer: int = 24  # which hidden_states[i] to return
    sample_rate: int = 16000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_dim: int = 80
    stride1: int = 2
    stride2: int = 2
    sub1_channels: int = 256
    sub2_channels: int = 512
    enc1_layers: int = 3
    enc2_layers: int = 3
    dec_layers: int = 6
    n_heads: int = 4
    d_ff: int = 2048
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 70
    peak_lr: float = 1e-3
    warmup_steps: int = 13450
    grad_clip: float = 5.0
    ctc_weight: float = 0.8
    ctc_weight_end: float = 0.3
    ctc_hold_epochs: int = 15
    ctc_weight_kind: str = "linear"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataConfig:
    """Data configuration."""
    csv_path: str = "train.csv"
    val_csv_path: str = "val.csv"
    vocab_json: str = "vocab.json"
    save_dir: str = "checkpoints"
    resume: bool = True
    resume_path: str = ""

@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mel: Optional[MelConfig] = None
    hf_robust: Optional[HFRobustConfig] = None
    
    # Feature mode settings
    feature_mode: Literal["mel", "robust"] = "mel"
    robust_source: Literal["disk", "hf"] = "disk"
    feature_root: Literal["robust-ft", "xls-r-300m", "xlsr300m-ft"] = "xls-r-300m"
    robust_layer: str = "24"
    
    def update_for_mode(self):
        """Update configuration based on feature mode."""
        if self.feature_mode == "robust":
            self.model.input_dim = 1024
            self.model.stride1 = 2
            self.model.sub1_channels = 512
            self.model.stride2 = 1
            self.model.sub2_channels = 512
        else:
            self.model.input_dim = 80
            self.model.stride1 = 2
            self.model.sub1_channels = 256
            self.model.sub2_channels = 512
            self.model.stride2 = 2
