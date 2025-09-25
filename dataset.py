"""Dataset and data loading utilities."""

import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from typing import Optional, Tuple, List

from config import Config, MelConfig, HFRobustConfig
from tokenizer import CharTokenizer
from features import LogMelSpec

class ASRDataset(Dataset):
    """ASR dataset for training and validation."""
    
    def __init__(
        self,
        csv_path: str,
        tokenizer: CharTokenizer,
        config: Config,
    ):
        """
        Initialize ASR dataset.
        
        Args:
            csv_path: Path to CSV with audio_filename and transcription columns
            tokenizer: Character tokenizer
            config: Main configuration
        """
        self.df = pd.read_csv(csv_path)
        self._validate_csv()
        
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup feature extractor for mel mode
        self.mel_extractor = None
        if config.feature_mode == "mel":
            assert config.mel is not None, "Mel config required for mel mode"
            self.mel_extractor = LogMelSpec(config.mel).eval()
    
    def _validate_csv(self):
        """Validate CSV has required columns."""
        required_cols = {"audio_filename", "transcription"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must have columns: {required_cols}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.
        
        Returns:
            features: Audio features or raw audio
            y_attn: Target IDs for attention decoder
            y_ctc: Target IDs for CTC
        """
        row = self.df.iloc[idx]
        audio_path = str(row["audio_filename"])
        text = row["transcription"]
        
        # Extract features based on mode
        if self.config.feature_mode == "robust":
            features = self._get_robust_features(audio_path)
        else:
            features = self._get_mel_features(audio_path)
        
        # Prepare targets
        y_attn = self.tokenizer.text2ids(text, add_sos_eos=True)
        y_ctc = self.tokenizer.text2ids(text, add_sos_eos=False)
        
        return features, y_attn, y_ctc
    
    def _get_robust_features(self, audio_path: str) -> torch.Tensor:
        """Get robust features (either from disk or return audio for HF)."""
        if self.config.robust_source == "hf":
            # Return raw audio for on-the-fly extraction
            wav, sr = torchaudio.load(audio_path)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0)
            return (wav.to(torch.float32).contiguous(), int(sr))
        else:
            # Load precomputed features from disk
            feat_path = self._audio_to_feature_path(audio_path)
            arr = np.load(feat_path)
            
            # Validate and transpose if needed
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D features, got {arr.shape}")
            
            if arr.shape[1] != self.config.model.input_dim:
                if arr.shape[0] == self.config.model.input_dim:
                    arr = arr.T
                else:
                    raise ValueError(f"Feature dim mismatch: {arr.shape}")
            
            return torch.from_numpy(arr.astype(np.float32)).contiguous()
    
    def _get_mel_features(self, audio_path: str) -> torch.Tensor:
        """Extract mel-spectrogram features."""
        wav, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != self.config.mel.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.config.mel.sample_rate
            )
        
        # Extract mel features
        mel_spec = self.mel_extractor(wav).squeeze(0)  # [n_mels, T]
        return mel_spec.transpose(0, 1).contiguous().to(torch.float32)  # [T, n_mels]

    def _audio_to_feature_path(self, audio_path: str) -> str:
        root = self.config.feature_root
        # Ensure we include 'features/' exactly once to mirror original code
        if not root.startswith("features/"):
            root = f"features/{root}"

        layer = str(self.config.robust_layer)
        feat_path = (
            audio_path
            .replace("audios", f"{root}/{layer}")
            .replace("wav/arctic_", "")
            .replace(".wav", ".npy")
        )
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(
                f"Feature file not found: {feat_path} "
                f"(from audio='{audio_path}', feature_root='{self.config.feature_root}', layer='{layer}')"
            )
        return feat_path


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Handles both precomputed features and raw audio for HF extraction.
    """
    feats, y_attn, y_ctc = zip(*batch)
    
    # Check if we have raw audio (for HF extraction) or features
    if isinstance(feats[0], tuple):
        # Raw audio for HF extraction
        wavs, srs = [], []
        for wav, sr in feats:
            if wav.dim() == 2 and wav.size(0) == 1:
                wav = wav.squeeze(0)
            wavs.append(wav)
            srs.append(int(sr))
        
        Y_attn, y_attn_len = pad_sequences_1d(y_attn, 0)
        Y_ctc, y_ctc_len = pad_sequences_1d(y_ctc, 0)
        
        return wavs, srs, Y_attn, y_attn_len, Y_ctc, y_ctc_len
    
    # Precomputed features
    X, x_len = pad_sequences_2d(feats, 0.0)
    Y_attn, y_attn_len = pad_sequences_1d(y_attn, 0)
    Y_ctc, y_ctc_len = pad_sequences_1d(y_ctc, 0)
    
    return X, x_len, Y_attn, y_attn_len, Y_ctc, y_ctc_len


def pad_sequences_2d(seqs: List[torch.Tensor], pad_value: float = 0.0):
    """Pad 2D sequences to same length."""
    lens = [s.size(0) for s in seqs]
    max_len = max(lens)
    feat_dim = seqs[0].size(1)
    
    out = seqs[0].new_full((len(seqs), max_len, feat_dim), pad_value)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
    
    return out, torch.tensor(lens, dtype=torch.long)


def pad_sequences_1d(seqs: List[torch.Tensor], pad_value: int = 0):
    """Pad 1D sequences to same length."""
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    
    out = seqs[0].new_full((len(seqs), max_len), pad_value)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    
    return out, torch.tensor(lens, dtype=torch.long)