"""Text recognition architectures.

Currently provided:
- CRNNLite: CNN + BiLSTM + CTC baseline (fast, trainable on modest hardware).
- SVTRLite: lightweight SVTR-style mixing (permuted attention over height/width tokens).
- SVTRv2: CTC with FRM + Local Mixing + SGM (ICCV 2025, arXiv 2411.15858).

Both output logits shape [B, T, V] suitable for CTC loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CRNNLite(nn.Module):
    def __init__(self, vocab_size: int, input_height: int = 48) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # h/2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),  # h/4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),  # h/8
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)),  # h/16
            nn.Conv2d(512, 512, 2, padding=0), nn.ReLU(inplace=True),
        )
        feat_h = max(input_height // 16 - 1, 1)
        self.rnn = nn.LSTM(
            input_size=512 * feat_h,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)  # [B, C, H, W]
        b, c, h, w = feat.shape
        feat = feat.permute(0, 3, 1, 2).reshape(b, w, c * h)  # [B, W, C*H]
        out, _ = self.rnn(feat)
        logits = self.fc(out)  # [B, W, V]
        return logits


class _MixingBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.mlp(self.norm2(x))
        return x


class SVTRLite(nn.Module):
    """Compact SVTR — single-stage patch + mixing.

    Input: [B, 3, H, W]. H divisible by 8 recommended.
    Output: [B, W/4, vocab_size] logits.
    """

    def __init__(self, vocab_size: int, dim: int = 192, depth: int = 4, input_height: int = 48) -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, stride=2, padding=1),  # /2
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),  # /4
            nn.GELU(),
        )
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse height
        self.blocks = nn.ModuleList([_MixingBlock(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)  # [B, D, H/4, W/4]
        x = self.h_pool(x).squeeze(2)  # [B, D, W/4]
        x = x.transpose(1, 2)  # [B, W/4, D]
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class SVTRMedical(SVTRLite):
    """SVTR with reinvested parameter budget for the medical B2B pivot
    (_53). Same ~3.6 M params as SVTRLite but redistributed:

        SVTR-lite (ko_en_v1, vocab 11309)         SVTR-medical (ko_en_medical_v1, vocab 2053)
        dim=192, depth=4                          dim=320, depth=6
        head Linear(192 → 11309) = 2.17 M         head Linear(320 → 2053) = 0.66 M (−70%)
        MixingBlocks 4 × 192² ≈ 1.18 M            MixingBlocks 6 × 320² ≈ 2.46 M (+108%)
        TOTAL ≈ 3.54 M                            TOTAL ≈ 3.6 M (same budget)

    The user's key insight (_53): 61% of SVTR-lite's parameters went to
    chars that never occur in the target domain (rare hangul artefacts,
    Chinese from wiki). Pruning the vocab to the top 2000 observed chars
    (99.63% coverage) lets the freed ~1.8 M params strengthen the actual
    feature extractor, which is where OOD generalisation lives.

    Same input/output contract as SVTRLite — callers don't need to change.
    """

    def __init__(self, vocab_size: int, dim: int = 320, depth: int = 6,
                 input_height: int = 48) -> None:
        super().__init__(vocab_size=vocab_size, dim=dim, depth=depth,
                          input_height=input_height)


class SVTRDeep(SVTRLite):
    """Depth-scaled SVTR for iter 41 arch-capacity experiment.

    Same dim (320) as SVTRMedical so the patch embed, head, and first six
    MixingBlocks load from a medical/consumer checkpoint via strict=False.
    Extra 4 blocks are randomly initialised and trained with a short
    warm-start schedule — tests whether extra depth closes the OOD gap
    vs Paddle (subtitle CER plateau), against the risk that 10 blocks
    overfit the ~60K-pair training mix.
    """

    def __init__(self, vocab_size: int, dim: int = 320, depth: int = 10,
                 input_height: int = 48) -> None:
        super().__init__(vocab_size=vocab_size, dim=dim, depth=depth,
                          input_height=input_height)


class SVTRWide(SVTRLite):
    """Width-expanded SVTR: dim=384 (vs SVTRDeep's 320), same depth=10.

    Tests whether the OOD gap is bottlenecked by per-layer representation
    capacity rather than depth (iter 154 showed depth=14 NEG). Patch embed
    + first 10 blocks load from svtr_deep checkpoint via strict=False with
    shape-mismatch layers randomly re-initialised.
    """

    def __init__(self, vocab_size: int, dim: int = 384, depth: int = 10,
                 input_height: int = 48) -> None:
        super().__init__(vocab_size=vocab_size, dim=dim, depth=depth,
                          input_height=input_height)


class SVTRXDeep(SVTRLite):
    """Depth-extended SVTR for iter 154 capacity ceiling experiment.

    Same dim (320) as SVTRDeep. depth=14 (vs SVTRDeep's 10, SVTRMedical's
    6). Patch embed + first 10 MixingBlocks load from svtr_deep checkpoint
    via strict=False; the extra 4 blocks are randomly initialised and
    trained with warm-start (LR 1e-4, 5000 iters). Same warm-start
    pattern that iter 41 used to extend svtr_medical → svtr_deep.

    Tests whether the iter 150 v033 raw CER (58.62%) is bottlenecked by
    arch capacity rather than data quality.
    """

    def __init__(self, vocab_size: int, dim: int = 320, depth: int = 14,
                 input_height: int = 48) -> None:
        super().__init__(vocab_size=vocab_size, dim=dim, depth=depth,
                          input_height=input_height)


class ParseQ(nn.Module):
    """Autoregressive PARSeq with teacher forcing and causal masking.

    Encoder: SVTRLite-style patch embedding + mixing blocks.
    Decoder: transformer decoder with causal self-attention + cross-attention
    to encoder features. Uses token embedding + position embedding for
    autoregressive generation.

    Training:  forward(images, tgt_tokens) with teacher forcing → [B, T, V]
    Inference: generate(images, sos_id, eos_id) → [B, T] token indices
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 320,
        enc_depth: int = 8,
        dec_depth: int = 3,
        num_heads: int = 8,
        max_label_length: int = 50,
        input_height: int = 48,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_label_length = max_label_length
        self.dim = dim

        self.patch = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.enc_blocks = nn.ModuleList(
            [_MixingBlock(dim, num_heads) for _ in range(enc_depth)]
        )
        self.enc_norm = nn.LayerNorm(dim)

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_label_length, dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_depth)
        self.dec_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        mask = torch.triu(torch.ones(max_label_length, max_label_length), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x)
        x = self.h_pool(x).squeeze(2)
        x = x.transpose(1, 2)
        for blk in self.enc_blocks:
            x = blk(x)
        return self.enc_norm(x)

    def decode(self, memory: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        B, T = tgt_tokens.shape
        tok = self.token_embed(tgt_tokens)
        pos = self.pos_embed(torch.arange(T, device=tgt_tokens.device))
        tgt = tok + pos.unsqueeze(0)
        out = self.decoder(tgt, memory, tgt_mask=self.causal_mask[:T, :T])
        return self.head(self.dec_norm(out))

    def forward(self, x: torch.Tensor, tgt_tokens: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.encode(x)
        if tgt_tokens is not None:
            return self.decode(memory, tgt_tokens)
        return self._generate_parallel(memory)

    def _generate_parallel(self, memory: torch.Tensor) -> torch.Tensor:
        B = memory.shape[0]
        pos = self.pos_embed(torch.arange(self.max_label_length, device=memory.device))
        tgt = pos.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(tgt, memory)
        return self.head(self.dec_norm(out))

    @torch.no_grad()
    def generate(self, x: torch.Tensor, sos_id: int, eos_id: int) -> torch.Tensor:
        memory = self.encode(x)
        B = memory.shape[0]
        device = memory.device
        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        for _ in range(self.max_label_length - 1):
            logits = self.decode(memory, generated)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return generated[:, 1:]


class ViTSTR(nn.Module):
    """ViT-based Scene Text Recognition with DeiT pretrained encoder + CTC.

    Uses timm's DeiT-small as the visual encoder (pretrained on ImageNet),
    replacing our custom patch embed + mixing blocks. Height is collapsed
    via adaptive pooling so the output is [B, W_tokens, vocab_size] for CTC.

    Benefits over SVTR: pretrained visual features from DeiT capture
    texture/edge patterns that our scratch-trained patch embed misses.
    """

    def __init__(
        self,
        vocab_size: int,
        model_name: str = "deit_small_patch16_224",
        input_height: int = 48,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        import timm
        self.vit = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0,
            dynamic_img_size=True,
        )
        self.dim = self.vit.embed_dim
        self.h_pool = nn.AdaptiveAvgPool1d(80)
        self.head = nn.Linear(self.dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        tokens = self.vit.forward_features(x)
        if tokens.dim() == 3 and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        tokens = tokens.transpose(1, 2)
        tokens = self.h_pool(tokens)
        tokens = tokens.transpose(1, 2)
        return self.head(tokens)


class _FRM(nn.Module):
    """Feature Rearrangement Module (SVTRv2).

    Replaces simple height pooling with a learnable depthwise rearrangement.
    Improves CTC alignment for wide-aspect-ratio text (e.g. dialogue subtitles).

    Input:  [B, D, H, W]
    Output: [B, T, D]   T = W/4 (after patch embed 4× downsampling)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))
        self.dw = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.h_pool(x).squeeze(2)   # [B, D, W]
        x = self.pw(torch.relu(self.dw(x))) + x  # residual depthwise rearrange
        x = x.transpose(1, 2)           # [B, W, D]
        return self.norm(x)


class _LocalMixingBlock(nn.Module):
    """Local Mixing block: depthwise conv instead of global attention.

    O(T·k) vs O(T²) — better for long-sequence (wide subtitle) crops.
    Used in the first half of SVTRv2 blocks.
    """

    def __init__(self, dim: int, kernel_size: int = 7, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                            padding=kernel_size // 2, groups=dim)
        self.pw = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x).transpose(1, 2)   # [B, D, T]
        h = self.pw(torch.relu(self.dw(h))).transpose(1, 2)  # [B, T, D]
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SVTRv2(nn.Module):
    """SVTRv2 — CTC recogniser with FRM + Local/Global hybrid mixing + SGM.

    Architecture (dim=320, depth=10):
      patch embed (4×DS) → FRM → 5× LocalMixing → 5× GlobalMixing → CTC head
      optional SGM aux head shares last encoder state (training only)

    Warm-start from svtr_deep:
      patch weights + last `depth - local_depth` global _MixingBlock weights
      transfer via strict=False; FRM + LocalMixing blocks init from scratch.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 320,
        depth: int = 10,
        local_depth: int = 5,
        input_height: int = 48,
        sgm_aux: bool = True,
    ) -> None:
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(3, dim // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.frm = _FRM(dim)
        blocks: list[nn.Module] = []
        for i in range(depth):
            blocks.append(
                _LocalMixingBlock(dim) if i < local_depth else _MixingBlock(dim)
            )
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(dim, vocab_size)
        self.sgm_head = nn.Linear(dim, vocab_size) if sgm_aux else None

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.patch(x)       # [B, D, H/4, W/4]
        x = self.frm(x)         # [B, T, D]
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        if self.sgm_head is not None and self.training:
            return logits, self.sgm_head(x)
        return logits


def build_recognition(arch: str, vocab_size: int, input_height: int = 48) -> nn.Module:
    if arch == "crnn":
        return CRNNLite(vocab_size=vocab_size, input_height=input_height)
    if arch in {"svtr", "svtr_lite"}:
        return SVTRLite(vocab_size=vocab_size, input_height=input_height)
    if arch in {"svtr_medical", "svtr_base"}:
        return SVTRMedical(vocab_size=vocab_size, input_height=input_height)
    if arch == "svtr_deep":
        return SVTRDeep(vocab_size=vocab_size, input_height=input_height)
    if arch == "svtr_wide":
        return SVTRWide(vocab_size=vocab_size, input_height=input_height)
    if arch == "svtr_xdeep":
        return SVTRXDeep(vocab_size=vocab_size, input_height=input_height)
    if arch == "parseq":
        return ParseQ(vocab_size=vocab_size, input_height=input_height)
    if arch == "vitstr":
        return ViTSTR(vocab_size=vocab_size, input_height=input_height)
    if arch == "svtrv2":
        return SVTRv2(vocab_size=vocab_size, input_height=input_height)
    if arch == "svtrv2_wide":
        # 2026-06-08 capacity test: width-expand prod SVTRv2 dim 320->448
        # (~2x mixing/attn params). hard-negative mining (NEG) proved the
        # recognizer is capacity-bound, not data-bound; depth was ceilinged
        # (iter154 depth14 NEG) so expand WIDTH. Scratch-train (dim mismatch
        # blocks warm-start). NOTE: check CPU inference latency before deploy
        # (bigger model may exceed Railway p50 budget ~378ms).
        return SVTRv2(vocab_size=vocab_size, dim=448, input_height=input_height)
    raise ValueError(f"Unknown recognition arch: {arch}")
