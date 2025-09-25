"""Feature extraction modules for ASR."""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import List, Optional, Tuple
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from config import MelConfig, HFRobustConfig

class LogMelSpec(nn.Module):
    """Log Mel-spectrogram feature extractor."""
    
    def __init__(self, cfg: MelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            n_mels=cfg.n_mels,
            center=cfg.center,
            pad_mode=cfg.pad_mode,
            power=cfg.power,
            norm=cfg.mel_norm,
            mel_scale=cfg.mel_scale,
        )
        
        self.to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power" if cfg.power == 2.0 else "magnitude",
            top_db=cfg.top_db
        )
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract log mel-spectrogram features.
        Returns: [B, n_mels, frames]
        """
        x = self._prepare_input(x)
        mel_spec = self.mel(x)
        return self.to_db(mel_spec)
    
    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input tensor to [B, T] format."""
        if x.ndim == 1:
            return x.unsqueeze(0)
        elif x.ndim == 2:
            if x.size(0) > 1 and x.size(1) != 1:
                return x.mean(dim=0, keepdim=False).unsqueeze(0)
            return x
        elif x.ndim == 3:
            if x.size(1) > 1:
                x = x.mean(dim=1)
            else:
                x = x.squeeze(1)
            return x
        else:
            raise ValueError(f"Unexpected audio tensor shape: {tuple(x.shape)}")


class HFRobustExtractor(nn.Module):
    """HuggingFace Wav2Vec2 feature extractor."""
    
    def __init__(self, cfg: HFRobustConfig):
        super().__init__()
        self.cfg = cfg
        
        # Load model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(cfg.model_name)
        self.model = Wav2Vec2Model.from_pretrained(cfg.model_name).to(cfg.device)
        self.hidden_size = self.model.config.hidden_size
        
        # Tracking for debug prints
        self._printed_info = False
        
        print(f"[HF] Loaded {cfg.model_name}")
        print(f"     Hidden size: {self.hidden_size}")
        print(f"     Layers: {self.model.config.num_hidden_layers}")
    
    @torch.no_grad()
    def extract_batch(
        self, 
        wav_list: List[torch.Tensor], 
        sr_list: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched feature extraction that mirrors the reference single-file flow,
        but uses padding + attention_mask so each item is treated independently.
        Returns:
            features: [B, T_max, D]
            lengths:  [B] valid frame lengths (no. of frames before padding)
        """
        # Prepare audio to 1-D, 16kHz, float32
        processed_audio = self._prepare_batch(wav_list, sr_list)
        
        # EXPLICIT padding + attention mask (key difference)
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding="longest",             # explicit (same as True but clearer)
            return_attention_mask=True,    # lets the model ignore padded samples
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        
        # Forward with hidden states (same as reference, just batched)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = self._get_layer_output(outputs)          # [B, T_max, D]
        self._print_debug_info(outputs, hidden_states)
        
        # True output lengths per item (so downstream CTC sees correct T)
        in_lengths  = inputs["attention_mask"].sum(dim=1)        # audio samples (after pad)
        out_lengths = self.model._get_feat_extract_output_lengths(in_lengths)  # frames
        
        return (
            hidden_states.detach().cpu().to(torch.float32).contiguous(),
            out_lengths.cpu().to(torch.long)
        )

    @torch.no_grad()
    def extract_batch_like_ref(
        self,
        wav_list: List[torch.Tensor],
        sr_list: Optional[List[int]] = None,
        layer_idx: Optional[int] = None,
        as_numpy: bool = True,
    ) -> Tuple[List[np.ndarray] | List[torch.Tensor], torch.Tensor]:
        """
        Reference-style outputs: list of per-utterance features [T_i, D] (no padding).
        Still runs a single batched forward, then splits by true lengths.
        """
        processed_audio = self._prepare_batch(wav_list, sr_list)
        inputs = self.feature_extractor(
            processed_audio,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)

        # Select layer like the reference: ft[f]
        hs_list = outputs.hidden_states if outputs.hidden_states is not None else (outputs.last_hidden_state,)
        L = len(hs_list)
        f = self.cfg.layer if layer_idx is None else int(layer_idx)
        if f < 0:
            f = L + f
        f = max(0, min(f, L - 1))
        hs = hs_list[f]  # [B, T_max, D]

        # True lengths and split into per-item arrays
        in_lengths  = inputs["attention_mask"].sum(dim=1)
        out_lengths = self.model._get_feat_extract_output_lengths(in_lengths).tolist()

        chunks = []
        for b, T in enumerate(out_lengths):
            one = hs[b, :T].detach().cpu().contiguous()  # [T, D]
            chunks.append(one.numpy() if as_numpy else one)
        return chunks, torch.tensor(out_lengths, dtype=torch.long)

    @torch.no_grad()
    def extract_from_tensor(
        self, 
        wav: torch.Tensor, 
        sr: int
    ) -> torch.Tensor:
        """
        Extract features from a single audio tensor.
        Returns: [T, D]
        """
        inputs = self._prepare_single(wav, sr)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = self._get_layer_output(outputs)
        
        # Remove batch dimension
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            
        self._print_debug_info(outputs, hidden_states)
        
        return hidden_states.detach().cpu().to(torch.float32).contiguous()
    
    def _prepare_batch(
        self, 
        wav_list: List[torch.Tensor], 
        sr_list: Optional[List[int]]
    ) -> List[np.ndarray]:
        """Prepare batch of audio: mono, resampled to cfg.sample_rate, float32 arrays."""
        processed = []
        for i, wav in enumerate(wav_list):
            if not isinstance(wav, torch.Tensor):
                wav = torch.as_tensor(wav)

            # To mono 1-D
            if wav.dim() == 2:
                wav = wav.mean(dim=0)
            elif wav.dim() > 2:
                wav = wav.flatten()
            
            # Resample if needed
            sr = self.cfg.sample_rate if sr_list is None else int(sr_list[i])
            if sr != self.cfg.sample_rate:
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), sr, self.cfg.sample_rate
                ).squeeze(0)
            
            processed.append(wav.detach().cpu().numpy().astype(np.float32))
        return processed
    
    def _prepare_single(self, wav: torch.Tensor, sr: int) -> dict:
        """Prepare single audio for processing (reference-style)."""
        # Convert to mono if needed
        if wav.ndim == 2:
            wav = wav.mean(0) if wav.size(0) > 1 else wav.squeeze(0)
        elif wav.ndim > 2:
            wav = wav.flatten()
        
        # Resample if needed
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(
                wav.unsqueeze(0), sr, self.cfg.sample_rate
            ).squeeze(0)
        
        inputs = self.feature_extractor(
            wav, sampling_rate=self.cfg.sample_rate, return_tensors="pt"
        )
        return {k: v.to(self.cfg.device) for k, v in inputs.items()}
    
    def _get_layer_output(self, outputs) -> torch.Tensor:
        """
        Get hidden states from specified layer.
        Mirrors the reference: `hidden_states = outputs.hidden_states; ft[f]`.
        Supports negative indices (e.g., -1 for last).
        """
        hidden_states = (
            outputs.hidden_states 
            if outputs.hidden_states is not None 
            else (outputs.last_hidden_state,)
        )
        num_layers = len(hidden_states)
        layer_idx = self.cfg.layer
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        layer_idx = max(0, min(layer_idx, num_layers - 1))
        return hidden_states[layer_idx]  # [B, T_max, D] or [T, D] if single
    
    def _print_debug_info(self, outputs, hidden_states):
        """Print debug information once."""
        if not self._printed_info:
            self._printed_info = True
            hs_list = outputs.hidden_states or (outputs.last_hidden_state,)
            print(f"[HF] Using layer {self.cfg.layer} -> hidden_states[{self.cfg.layer}]")
            print(f"     Total hidden_states entries: {len(hs_list)}")
            print(f"     Output shape: {tuple(hidden_states.shape)}")
