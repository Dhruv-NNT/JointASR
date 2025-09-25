"""ASR model architecture."""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from config import ModelConfig

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)
    
    def _build_pe(self, max_len: int) -> torch.Tensor:
        """Build positional encoding matrix."""
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * 
            (-math.log(10000.0) / self.d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        
        # Grow PE table if needed
        if seq_len > self.pe.size(0):
            new_len = int(2 ** math.ceil(math.log2(seq_len)))
            self.pe = self._build_pe(new_len).to(x.device)
        
        return self.dropout(x + self.pe[:seq_len].unsqueeze(0))


class SubsampleBlock(nn.Module):
    """Subsampling block with convolution and transformer encoder."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        num_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()
        
        self.stride = stride
        
        # Convolutional subsampling
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=True
            ),
            nn.ReLU(inplace=True)
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(out_channels, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input comes as [B, T, D]; conv1d expects [B, D, T]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B,T,D], got {tuple(x.shape)}")
        x = x.transpose(1, 2)  # [B, D, T]

        # Convolutional subsampling
        x = self.conv(x)  # [B, D', T']

        # Update lengths (ceiling division)
        new_lengths = (lengths + (self.stride - 1)) // self.stride

        # Back to [B, T', D'] for transformer
        x = x.transpose(1, 2)

        # Positional encoding
        x = self.pos_enc(x)

        # Padding mask and Transformer encoder
        pad_mask = create_padding_mask(new_lengths, x.size(1))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x, new_lengths


class JointASR(nn.Module):
    """Joint CTC-Attention ASR model."""
    
    def __init__(
        self,
        config: ModelConfig,
        vocab_size: int,
        pad_id: int,
        sos_id: int,
        eos_id: int
    ):
        super().__init__()
        
        self.config = config
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        
        # Two-stage encoder
        self.encoder_stage1 = SubsampleBlock(
            in_channels=config.input_dim,
            out_channels=config.sub1_channels,
            stride=config.stride1,
            num_layers=config.enc1_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        
        self.encoder_stage2 = SubsampleBlock(
            in_channels=config.sub1_channels,
            out_channels=config.sub2_channels,
            stride=config.stride2,
            num_layers=config.enc2_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        
        # Decoder
        self.embed = nn.Embedding(vocab_size, config.sub2_channels, padding_idx=pad_id)
        self.pos_dec = PositionalEncoding(config.sub2_channels, config.dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.sub2_channels,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.dec_layers
        )
        
        # Output heads
        self.ctc_head = nn.Linear(config.sub2_channels, vocab_size)
        self.attn_head = nn.Linear(config.sub2_channels, vocab_size)
    
    def forward_encoder(
        self, 
        features: torch.Tensor, 
        feat_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Returns:
            enc_out: Encoder output [B, T', D]
            enc_lengths: Valid lengths [B]
            enc_mask: Padding mask [B, T']
        """
        # Stage 1 encoding
        x, lengths = self.encoder_stage1(features, feat_lengths)
        
        # Stage 2 encoding
        x, lengths = self.encoder_stage2(x, lengths)
        
        # Create final padding mask
        mask = create_padding_mask(lengths, x.size(1))
        
        return x, lengths, mask
    
    def forward_decoder(
        self,
        enc_out: torch.Tensor,
        enc_mask: torch.Tensor,
        ys_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            enc_out: Encoder output [B, T', D]
            enc_mask: Encoder padding mask [B, T']
            ys_in: Input target sequence [B, U]
            
        Returns:
            Decoder output logits [B, U, V]
        """
        # Embed targets
        tgt = self.embed(ys_in)
        tgt = self.pos_dec(tgt)
        
        # Create attention masks
        seq_len = ys_in.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=ys_in.device), 
            diagonal=1
        ).bool()
        tgt_pad_mask = (ys_in == self.pad_id)
        
        # Decode
        dec_out = self.decoder(
            tgt, enc_out,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=enc_mask
        )
        
        return self.attn_head(dec_out)
    
    def forward(
        self,
        features: torch.Tensor,
        feat_lengths: torch.Tensor,
        ys_in: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Returns:
            ctc_logits: CTC output [B, T', V]
            enc_lengths: Encoder lengths [B]
            attn_logits: Attention output [B, U, V]
        """
        # Encode
        enc_out, enc_lengths, enc_mask = self.forward_encoder(features, feat_lengths)
        
        # CTC head
        ctc_logits = self.ctc_head(enc_out)
        
        # Attention decoder
        attn_logits = self.forward_decoder(enc_out, enc_mask, ys_in)
        
        return ctc_logits, enc_lengths, attn_logits


def create_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create padding mask from lengths."""
    if max_len is None:
        max_len = int(lengths.max())
    
    range_tensor = torch.arange(max_len, device=lengths.device)
    return range_tensor.unsqueeze(0) >= lengths.unsqueeze(1)