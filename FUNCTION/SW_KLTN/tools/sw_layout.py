"""
sw_layout.py -- Python mirror of HW SRAM banking / addressing.

Mirrors addr_gen_input.sv, addr_gen_weight.sv, addr_gen_output.sv so that
Python golden vectors can be packed into the exact same bank layout the RTL
uses and memh files can be loaded via $readmemh → ext_wr in testbenches.

LANES is taken from accel_pkg (RTL default 20).  PE_COLS = 4.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# ---------- RTL constants (must match accel_pkg.sv) ----------
LANES    = 20
PE_COLS  = 4
PE_ROWS  = 3
INPUT_BANKS  = 3
OUTPUT_BANKS = PE_COLS   # 4


def _wblk_total(win: int) -> int:
    return (win + LANES - 1) // LANES


# ═══════════════════════════════════════════════════════════════
#  INPUT PACKING  (mirrors addr_gen_input.sv)
# ═══════════════════════════════════════════════════════════════

def pack_input_to_banks(
    X_int8: np.ndarray,
    zp_x: int = 0,
) -> list[dict[int, np.ndarray]]:
    """
    Pack X_int8 [1, Cin, Hin, Win] into 3 input banks.

    Banking rule : bank = h % 3
    Address rule : addr = (h // 3) * wblk_total * cin + c * wblk_total + wblk

    Each bank entry is an int8 array of length LANES (one width-block).

    Returns
    -------
    banks : list of 3 dicts  {addr: int8[LANES]}
    """
    assert X_int8.ndim == 4 and X_int8.shape[0] == 1
    _, cin, hin, win = X_int8.shape
    wbt = _wblk_total(win)

    banks: list[dict[int, np.ndarray]] = [{} for _ in range(INPUT_BANKS)]

    for h in range(hin):
        bk = h % 3
        for c in range(cin):
            for wb in range(wbt):
                addr = (h // 3) * wbt * cin + c * wbt + wb
                lane_data = np.full(LANES, zp_x, dtype=np.int8)
                w_start = wb * LANES
                w_end   = min(w_start + LANES, win)
                for l in range(w_end - w_start):
                    lane_data[l] = X_int8[0, c, h, w_start + l]
                banks[bk][addr] = lane_data
    return banks


# ═══════════════════════════════════════════════════════════════
#  WEIGHT PACKING  (mirrors addr_gen_weight.sv)
# ═══════════════════════════════════════════════════════════════

def pack_weight_rs3(
    W_int8: np.ndarray,
) -> list[dict[int, np.ndarray]]:
    """
    Pack RS3 weight [Cout, Cin, 3, kW=3] into 3 weight banks.

    Banking rule : bank = kh_row (0,1,2)
    Address rule (per PE column col):
        addr = (cout_base + col) * cin * kw + cin_iter * kw + kw_iter

    We store all columns interleaved at the same address; TB writes one
    lane vector per address where lane[col_offset] holds the weight value.
    """
    cout, cin, kh, kw = W_int8.shape
    assert kh == 3 and kw == 3

    banks: list[dict[int, np.ndarray]] = [{} for _ in range(3)]

    for kh_row in range(3):
        bk = kh_row
        for co in range(cout):
            for ci in range(cin):
                for kwi in range(kw):
                    addr = co * cin * kw + ci * kw + kwi
                    if addr not in banks[bk]:
                        banks[bk][addr] = np.zeros(LANES, dtype=np.int8)
                    banks[bk][addr][0] = W_int8[co, ci, kh_row, kwi]
    return banks


def pack_weight_os1(
    W_int8: np.ndarray,
) -> list[dict[int, np.ndarray]]:
    """
    Pack OS1 weight [Cout, Cin, 1, 1] into weight banks.

    Banking : bank = 0 (only 1 kernel row → iter_kh_row = 0)
    Address : addr = cout * cin + cin_iter
    """
    cout, cin = W_int8.shape[0], W_int8.shape[1]
    banks: list[dict[int, np.ndarray]] = [{} for _ in range(3)]

    for co in range(cout):
        for ci in range(cin):
            addr = co * cin + ci
            if addr not in banks[0]:
                banks[0][addr] = np.zeros(LANES, dtype=np.int8)
            banks[0][addr][0] = W_int8[co, ci, 0, 0]
    return banks


def pack_weight_dw3(
    W_int8: np.ndarray,
) -> list[dict[int, np.ndarray]]:
    """
    Pack DW3 weight [C, 3, 3] (depthwise) into 3 weight banks.

    Banking : bank = kh_row
    Address : addr = channel * kw + kw_iter
    """
    ch, kh, kw = W_int8.shape
    assert kh == 3 and kw == 3

    banks: list[dict[int, np.ndarray]] = [{} for _ in range(3)]

    for kh_row in range(3):
        bk = kh_row
        for c in range(ch):
            for kwi in range(kw):
                addr = c * kw + kwi
                if addr not in banks[bk]:
                    banks[bk][addr] = np.zeros(LANES, dtype=np.int8)
                banks[bk][addr][0] = W_int8[c, kh_row, kwi]
    return banks


def pack_weight_dw7(
    W_int8: np.ndarray,
) -> list[dict[int, np.ndarray]]:
    """
    Pack DW7 weight [C, 7, 7] (depthwise 7x7 multipass) into 3 weight banks.

    Banking : bank = kh_row % 3
    Address : addr = channel * kw + kw_iter
    """
    ch, kh, kw = W_int8.shape
    assert kh == 7 and kw == 7

    banks: list[dict[int, np.ndarray]] = [{} for _ in range(3)]

    for kh_row in range(kh):
        bk = kh_row % 3
        for c in range(ch):
            for kwi in range(kw):
                addr = c * kw + kwi
                if addr not in banks[bk]:
                    banks[bk][addr] = np.zeros(LANES, dtype=np.int8)
                banks[bk][addr][0] = W_int8[c, kh_row, kwi]
    return banks


# ═══════════════════════════════════════════════════════════════
#  OUTPUT UNPACKING  (mirrors addr_gen_output.sv)
# ═══════════════════════════════════════════════════════════════

def output_addr(
    h_out: int, cout_group: int, wblk: int,
    wout: int, cout: int,
) -> tuple[int, int]:
    """
    Compute (bank, addr) for a single output position.

    Banking : bank = col (0..3), where col = cout % PE_COLS
    Address : h_out * cout_groups_total * wblk_total + cout_group * wblk_total + wblk
    """
    wbt = _wblk_total(wout)
    cout_groups_total = (cout + PE_COLS - 1) // PE_COLS
    bk = cout_group % PE_COLS   # same as col in RTL (cout_group is already group idx)
    # In RTL, col = local column within the group, bank = col.
    # For the first group (cout_group=0), cols 0..3 → banks 0..3.
    # Output addr within each bank:
    addr = h_out * cout_groups_total * wbt + cout_group * wbt + wblk
    return bk, addr


def unpack_output_banks(
    banks: list[dict[int, np.ndarray]],
    cout: int, hout: int, wout: int,
) -> np.ndarray:
    """
    Unpack output banks (4 banks, one per PE column) back to Y[1, Cout, Hout, Wout].
    """
    Y = np.zeros((1, cout, hout, wout), dtype=np.int8)
    wbt = _wblk_total(wout)
    cout_groups_total = (cout + PE_COLS - 1) // PE_COLS

    for co in range(cout):
        col = co % PE_COLS
        cg  = co // PE_COLS
        bk  = col
        for ho in range(hout):
            for wb in range(wbt):
                addr = ho * cout_groups_total * wbt + cg * wbt + wb
                if addr in banks[bk]:
                    w_start = wb * LANES
                    w_end   = min(w_start + LANES, wout)
                    for l in range(w_end - w_start):
                        Y[0, co, ho, w_start + l] = banks[bk][addr][l]
    return Y


# ═══════════════════════════════════════════════════════════════
#  MEMH WRITER / READER
# ═══════════════════════════════════════════════════════════════

def write_bank_memh(
    path: str,
    bank: dict[int, np.ndarray],
) -> int:
    """
    Write one bank dict {addr: int8[LANES]} as a .memh file.

    Format: @ADDR HH HH HH ... (LANES hex bytes per address)

    Returns the number of addresses written.
    """
    if not bank:
        with open(path, "w") as f:
            f.write("// empty bank\n")
        return 0

    with open(path, "w", encoding="utf-8") as f:
        for addr in sorted(bank.keys()):
            vals = bank[addr]
            hex_str = " ".join(f"{int(v) & 0xFF:02x}" for v in vals)
            f.write(f"@{addr:04x} {hex_str}\n")
    return len(bank)


def write_quant_params(
    path: str,
    bias: np.ndarray,
    m_int: np.ndarray,
    shift: np.ndarray,
    zp_out: int,
    channels: int,
) -> None:
    """
    Write quantization parameters as a simple text file the TB can parse.

    Format (one line per channel):
        CHAN BIAS_HEX M_INT_HEX SHIFT ZP_OUT
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"// channels={channels} zp_out={zp_out}\n")
        for c in range(channels):
            b  = int(bias[c]) if c < len(bias) else 0
            mi = int(m_int[c]) if c < len(m_int) else 1
            sh = int(shift[c]) if c < len(shift) else 0
            f.write(f"{c} {b & 0xFFFFFFFF:08x} {mi & 0xFFFFFFFF:08x} {sh} {zp_out}\n")


def write_expected_memh(
    path: str,
    Y_int8: np.ndarray,
    cout: int, hout: int, wout: int,
    bank_id: int,
) -> int:
    """
    Write expected output for one bank (bank_id) as memh file.

    Uses addr_gen_output addressing to map Y[1,Cout,Hout,Wout] to bank entries.

    Returns number of addresses written.
    """
    wbt = _wblk_total(wout)
    cout_groups_total = (cout + PE_COLS - 1) // PE_COLS

    bank: dict[int, np.ndarray] = {}
    for co in range(cout):
        col = co % PE_COLS
        if col != bank_id:
            continue
        cg = co // PE_COLS
        for ho in range(hout):
            for wb in range(wbt):
                addr = ho * cout_groups_total * wbt + cg * wbt + wb
                lane_data = np.zeros(LANES, dtype=np.int8)
                w_start = wb * LANES
                w_end   = min(w_start + LANES, wout)
                for l in range(w_end - w_start):
                    lane_data[l] = Y_int8[0, co, ho, w_start + l]
                bank[addr] = lane_data

    return write_bank_memh(path, bank)
