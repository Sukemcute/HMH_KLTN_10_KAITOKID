"""
Golden model for YOLOv10n INT8 Accelerator V4.
Exact INT8/INT32/INT64 arithmetic matching RTL Golden Rules.

RULE 1:  Signed INT8 [-128, 127]
RULE 2:  Half-up rounding (NOT floor)
RULE 3:  INT64 for PPU multiply
RULE 4:  ReLU: y = max(0, x)
RULE 5:  Padding fill = zp_x
RULE 6:  Per-output-channel bias/m_int/shift
RULE 10: zp_out added AFTER activation, BEFORE clamp
"""
import numpy as np

LANES   = 20
PE_ROWS = 3
PE_COLS = 4


def clamp_i8(x):
    return int(max(-128, min(127, x)))


def ppu_golden(psum_i32, bias_i32, m_int_u32, shift_u8, zp_out_i8, relu=False):
    """Post-Processing Unit — matches ppu.sv exactly."""
    biased  = int(psum_i32) + int(bias_i32)
    product = biased * int(m_int_u32)
    sh = int(shift_u8)
    if sh > 0:
        rounded = product + (1 << (sh - 1))
    else:
        rounded = product
    shifted = rounded >> sh
    activated = max(0, shifted) if relu else shifted
    return clamp_i8(activated + int(zp_out_i8))


def rs3_golden(inp, wgt, bias, m_int, shift, zp_out, relu, stride, padding, zp_x):
    """
    RS3 (Conv 3x3 row-stationary) golden model.
    HW behaviour: kh is parallelised across PE rows; kw iterates with
    the SAME input address (no horizontal sliding). Each kw value only
    selects a different weight.
    """
    Hin, Win, Cin = inp.shape
    Cout, _, Kh, Kw = wgt.shape
    Hout = (Hin + 2 * padding - Kh) // stride + 1
    Wout = Win
    out = np.zeros((Hout, Wout, Cout), dtype=np.int8)

    for ho in range(Hout):
        for wo in range(Wout):
            for co in range(Cout):
                psum = 0
                for ci in range(Cin):
                    for kh in range(Kh):
                        hi = ho * stride + kh - padding
                        a = int(zp_x) if (hi < 0 or hi >= Hin) else int(inp[hi, wo, ci])
                        for kw in range(Kw):
                            psum += a * int(wgt[co, ci, kh, kw])
                out[ho, wo, co] = ppu_golden(
                    psum, bias[co], m_int[co], shift[co], zp_out[co], relu)
    return out


def os1_golden(inp, wgt, bias, m_int, shift, zp_out, relu):
    """OS1 (Conv 1x1 output-stationary) — no padding, kh=kw=1."""
    Hin, Win, Cin = inp.shape
    Cout, _ = wgt.shape
    out = np.zeros((Hin, Win, Cout), dtype=np.int8)

    for ho in range(Hin):
        for wo in range(Win):
            for co in range(Cout):
                psum = 0
                for ci in range(Cin):
                    psum += int(inp[ho, wo, ci]) * int(wgt[co, ci])
                out[ho, wo, co] = ppu_golden(
                    psum, bias[co], m_int[co], shift[co], zp_out[co], relu)
    return out


def dw3_golden(inp, wgt, bias, m_int, shift, zp_out, relu, stride, padding, zp_x):
    """
    DW3 (Depthwise Conv 3x3) — per-channel convolution.
    Same kw non-sliding behaviour as RS3.
    wgt shape: [Cin, Kh, Kw]
    """
    Hin, Win, Cin = inp.shape
    _, Kh, Kw = wgt.shape
    Hout = (Hin + 2 * padding - Kh) // stride + 1
    Wout = Win
    out = np.zeros((Hout, Wout, Cin), dtype=np.int8)

    for ho in range(Hout):
        for wo in range(Wout):
            for ch in range(Cin):
                psum = 0
                for kh in range(Kh):
                    hi = ho * stride + kh - padding
                    a = int(zp_x) if (hi < 0 or hi >= Hin) else int(inp[hi, wo, ch])
                    for kw in range(Kw):
                        psum += a * int(wgt[ch, kh, kw])
                out[ho, wo, ch] = ppu_golden(
                    psum, bias[ch], m_int[ch], shift[ch], zp_out[ch], relu)
    return out


def mp5_golden(inp, padding=2):
    """
    MP5 (MaxPool 5x5 stride=1) — signed max, padding = INT8_MIN.
    Horizontal sliding IS present (pool_packed logic in subcluster_datapath).
    """
    Hin, Win, Cin = inp.shape
    K = 5
    Hout = Hin + 2 * padding - K + 1
    Wout = Win
    out = np.zeros((Hout, Wout, Cin), dtype=np.int8)

    for ho in range(Hout):
        for wo in range(Wout):
            for ch in range(Cin):
                mx = -128
                for kh in range(K):
                    for kw in range(K):
                        hi = ho + kh - padding
                        wi = wo + kw - padding
                        if 0 <= hi < Hin and 0 <= wi < Win:
                            mx = max(mx, int(inp[hi, wi, ch]))
                out[ho, wo, ch] = np.int8(mx)
    return out
