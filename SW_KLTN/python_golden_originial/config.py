"""
config.py – Hằng số kiến trúc cho qYOLOv10n INT8 Accelerator Golden Python
Phase 1 – Golden Python Oracle
"""

# ─── Hardware Architecture Constants ───────────────────────────────────────
INPUT_BANKS   = 3      # 3 GLB input banks (h mod 3)
OUTPUT_BANKS  = 4      # 4 GLB output banks (out_row mod 4)
LANES         = 16     # PE lane width
PSUM_BITS     = 32     # PSUM accumulator: INT32
ACT_BITS      = 8      # Activation: INT8
WEIGHT_BITS   = 8      # Weight: INT8

# DW_7x7 multi-pass split: (rows_pass1, rows_pass2, rows_pass3)
DW7x7_SPLIT   = (3, 3, 1)   # rows 0-2, 3-5, 6

# ─── INT8 Range ─────────────────────────────────────────────────────────────
INT8_MIN  = -128
INT8_MAX  =  127
INT32_MIN = -(2**31)
INT32_MAX =  (2**31) - 1

# ─── Primitive IDs ──────────────────────────────────────────────────────────
P0_RS_DENSE_3x3     = 0
P1_OS_1x1           = 1
P2_DW_3x3           = 2
P3_MAXPOOL_5x5      = 3
P4_MOVE             = 4
P5_CONCAT           = 5
P6_UPSAMPLE_NEAREST = 6
P7_EWISE_ADD        = 7
P8_DW_7x7_MULTIPASS = 8
P9_GEMM_ATTN_BASIC  = 9

PRIMITIVE_NAMES = {
    P0_RS_DENSE_3x3:     "RS_DENSE_3x3",
    P1_OS_1x1:           "OS_1x1",
    P2_DW_3x3:           "DW_3x3",
    P3_MAXPOOL_5x5:      "MAXPOOL_5x5",
    P4_MOVE:             "MOVE",
    P5_CONCAT:           "CONCAT",
    P6_UPSAMPLE_NEAREST: "UPSAMPLE_NEAREST",
    P7_EWISE_ADD:        "EWISE_ADD",
    P8_DW_7x7_MULTIPASS: "DW_7x7_MULTIPASS",
    P9_GEMM_ATTN_BASIC:  "GEMM_ATTN_BASIC",
}

# ─── Activation Modes ────────────────────────────────────────────────────────
ACT_NONE  = 0
ACT_SILU  = 1
ACT_RELU  = 2
ACT_RELU6 = 3

# ─── Quantization Policy ────────────────────────────────────────────────────
QUANT_WEIGHT_ZP = 0     # Weight zero-point luôn = 0 (symmetric)
ROUNDING_MODE   = "half_up"  # "half_up" hoặc "half_even" – phải nhất quán
