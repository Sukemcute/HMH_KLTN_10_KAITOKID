"""
types.py – Dataclasses và Enums cho qYOLOv10n INT8 Golden Python
Phase 1 – Golden Python Oracle
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np


# ─── Enums ───────────────────────────────────────────────────────────────────

class Primitive(Enum):
    RS_DENSE_3x3     = 0
    OS_1x1           = 1
    DW_3x3           = 2
    MAXPOOL_5x5      = 3
    MOVE             = 4
    CONCAT           = 5
    UPSAMPLE_NEAREST = 6
    EWISE_ADD        = 7
    DW_7x7_MULTIPASS = 8
    GEMM_ATTN_BASIC  = 9


class PartitionMode(Enum):
    TILE_HW   = 0
    TILE_COUT = 1
    TILE_CIN  = 2


class ActivationMode(Enum):
    NONE  = 0
    SILU  = 1
    RELU  = 2
    RELU6 = 3


# ─── Core Data Classes ───────────────────────────────────────────────────────

@dataclass
class QuantParams:
    """Quantization parameters cho một tensor."""
    scale: float
    zp: int
    dtype: str = "int8"   # "int8" hoặc "int32"

    def __post_init__(self):
        assert self.scale > 0, f"scale phải > 0, nhận được {self.scale}"

    def __repr__(self):
        return f"QuantParams(scale={self.scale:.6f}, zp={self.zp})"


@dataclass
class TensorMeta:
    """Tensor INT8 kèm quant metadata – đơn vị cơ bản trong model forward."""
    tensor: np.ndarray   # INT8 numpy array
    scale: float
    zp: int

    def __post_init__(self):
        assert self.scale > 0, f"scale phải > 0"
        assert self.tensor.dtype in (np.int8, np.uint8), \
            f"tensor phải INT8/UINT8, nhận được {self.tensor.dtype}"

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def quant_params(self) -> QuantParams:
        return QuantParams(scale=self.scale, zp=self.zp)

    def dequantize(self) -> np.ndarray:
        """Chuyển về float32 để debug/verify. KHÔNG dùng trong execution path."""
        return (self.tensor.astype(np.float32) - self.zp) * self.scale


@dataclass
class LayerSpec:
    """Spec cho một layer trong model forward."""
    idx:          int
    block_type:   str              # "Conv", "QC2f", "SCDown", ...
    primitive_seq: List[int]       # list of Primitive IDs
    in_shape:     tuple
    out_shape:    tuple
    stride:       int  = 1
    kernel:       int  = 3
    sources:      List[int] = field(default_factory=lambda: [-1])
    hold_output:  bool = False
    hold_until:   int  = -1        # layer idx khi hold được giải phóng
    output_name:  Optional[str] = None   # "P3", "P4", "P5" nếu là output


@dataclass
class TileFlags:
    """Flags cho một TILE_DESC."""
    first_tile:   bool = False
    edge_tile_h:  bool = False
    edge_tile_w:  bool = False
    hold_skip:    bool = False
    need_swizzle: bool = False
    psum_carry_in: bool = False


@dataclass
class LastFlags:
    """Last-pass flags."""
    last_cin:    bool = False
    last_kernel: bool = False
    last_reduce: bool = False

    @property
    def last_pass(self) -> bool:
        """True khi toàn bộ reduction hoàn tất → kích hoạt PPU."""
        return self.last_cin and self.last_kernel and self.last_reduce


@dataclass
class ConvWeights:
    """Trọng số INT8 cho một conv primitive."""
    W_int8:   np.ndarray    # [Cout, Cin, kH, kW] hoặc [Cout, kH, kW] cho DW
    B_int32:  np.ndarray    # [Cout]
    scale_w:  np.ndarray    # [Cout] per-output-channel
    zp_w:     np.ndarray    # [Cout] – phải = 0 (symmetric)
    scale_out: float        # scale của output tensor
    zp_out:   int           # zp của output tensor
    activation: int = 0     # ACT_NONE=0, ACT_SILU=1


@dataclass
class RequantParams:
    """Fixed-point requant parameters – pre-computed offline."""
    M_int:  np.ndarray    # [Cout] INT32 multipliers
    shift:  int           # common shift (hoặc per-channel nếu cần)
    zp_out: int
