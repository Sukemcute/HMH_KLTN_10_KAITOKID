"""
Vector generator for Stage 8 cosimulation.
Generates .memh files (one hex byte per line) matching the HW bank layout.

Usage:  python gen_vectors.py --out ../vectors
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from golden_model import (LANES, PE_ROWS, PE_COLS,
                          rs3_golden, os1_golden, dw3_golden, mp5_golden)

# ─────────────────────────────────────────────────────────────
#  .memh helpers
# ─────────────────────────────────────────────────────────────
def _hex8(v):
    return format(int(v) & 0xFF, '02X')

def write_memh(path, flat_bytes):
    """Write a flat list of uint8 values, one hex byte per line."""
    with open(path, 'w') as f:
        for b in flat_bytes:
            f.write(_hex8(b) + '\n')

def write_quant_params(path, bias, m_int, shift, zp_out):
    with open(path, 'w') as f:
        for i in range(len(bias)):
            f.write(f'{i} {int(bias[i])} {int(m_int[i])} {int(shift[i])} {int(zp_out[i])}\n')

# ─────────────────────────────────────────────────────────────
#  Input bank layout  (addr_gen_input.sv)
#    bank = h_in % 3
#    addr = (h_in // 3) * wblk_total * Cin + cin * wblk_total + wblk
#    each address = LANES bytes
# ─────────────────────────────────────────────────────────────
def input_to_banks(inp, Win, Cin):
    """inp: [Hin, Win, Cin] int8 → dict {bank: flat_bytes}"""
    Hin = inp.shape[0]
    wblk_total = (Win + LANES - 1) // LANES
    banks = {0: {}, 1: {}, 2: {}}
    for h in range(Hin):
        bank = h % 3
        for ci in range(Cin):
            for wb in range(wblk_total):
                addr = (h // 3) * wblk_total * Cin + ci * wblk_total + wb
                row = []
                for ln in range(LANES):
                    w_idx = wb * LANES + ln
                    row.append(int(inp[h, w_idx, ci]) if w_idx < Win else 0)
                banks[bank][addr] = row
    return banks

def bank_dict_to_flat(bdict):
    if not bdict:
        return []
    max_addr = max(bdict.keys()) + 1
    flat = []
    for a in range(max_addr):
        row = bdict.get(a, [0] * LANES)
        flat.extend(row)
    return flat

def bank_max_addr(bdict):
    return (max(bdict.keys()) + 1) if bdict else 0

# ─────────────────────────────────────────────────────────────
#  Weight bank layout  (addr_gen_weight.sv)
#    bank = kh % 3
#    RS3: addr[col] = (cout_group*4+col)*Cin*Kw + cin*Kw + kw
#    OS1: addr[col] = (cout_group*4+col)*Cin + cin
#    DW3: addr[col] = (ch_group*4+col)*Kw + kw
#    weight is broadcast: same value for all LANES
# ─────────────────────────────────────────────────────────────
def weight_rs3_to_banks(wgt, Cin, Kw):
    """wgt: [Cout, Cin, Kh, Kw] → {bank: flat_bytes}"""
    Cout, _, Kh, _ = wgt.shape
    banks = {0: {}, 1: {}, 2: {}}
    for kh in range(Kh):
        bank = kh % 3
        for co in range(Cout):
            for ci in range(Cin):
                for kw in range(Kw):
                    addr = co * Cin * Kw + ci * Kw + kw
                    val  = int(wgt[co, ci, kh, kw])
                    banks[bank][addr] = [val] * LANES
    return banks

def weight_os1_to_banks(wgt, Cin):
    """wgt: [Cout, Cin] → bank 0 only (kh=0)"""
    Cout, _ = wgt.shape
    banks = {0: {}, 1: {}, 2: {}}
    for co in range(Cout):
        for ci in range(Cin):
            addr = co * Cin + ci
            val  = int(wgt[co, ci])
            banks[0][addr] = [val] * LANES
    return banks

def weight_dw3_to_banks(wgt, Kw):
    """wgt: [Cin, Kh, Kw] → banks by kh"""
    Cin, Kh, _ = wgt.shape
    banks = {0: {}, 1: {}, 2: {}}
    for kh in range(Kh):
        bank = kh % 3
        for ch in range(Cin):
            for kw in range(Kw):
                addr = ch * Kw + kw
                val  = int(wgt[ch, kh, kw])
                banks[bank][addr] = [val] * LANES
    return banks

# ─────────────────────────────────────────────────────────────
#  Output bank layout  (addr_gen_output.sv)
#    bank[col] = col
#    addr = h_out * cout_groups_total * wblk_total + cout_group * wblk_total + wblk
# ─────────────────────────────────────────────────────────────
def output_to_banks(out_arr, Wout, Cout):
    """out_arr: [Hout, Wout, Cout] int8 → {bank: flat_bytes}"""
    Hout = out_arr.shape[0]
    wblk_total = (Wout + LANES - 1) // LANES
    cout_grp_total = (Cout + PE_COLS - 1) // PE_COLS
    banks = {0: {}, 1: {}, 2: {}, 3: {}}
    for ho in range(Hout):
        for co in range(Cout):
            col       = co % PE_COLS
            cout_grp  = co // PE_COLS
            for wb in range(wblk_total):
                addr = ho * cout_grp_total * wblk_total + cout_grp * wblk_total + wb
                row = []
                for ln in range(LANES):
                    w_idx = wb * LANES + ln
                    row.append(int(out_arr[ho, w_idx, co]) if w_idx < Wout else 0)
                banks[col][addr] = row
    return banks

# ─────────────────────────────────────────────────────────────
#  MP5 output bank layout (different: bank = iter_mp5_ch % 4)
#    addr = h_out * cout_groups_total * wblk_total
#           + (iter_mp5_ch // 4) * wblk_total + wblk
# ─────────────────────────────────────────────────────────────
def mp5_output_to_banks(out_arr, Wout, Cout):
    return output_to_banks(out_arr, Wout, Cout)

# ─────────────────────────────────────────────────────────────
#  Primitive generators
# ─────────────────────────────────────────────────────────────

def gen_rs3(rng, outdir):
    Cin, Cout, Hin, Win = 1, 4, 3, 20
    Kh, Kw, stride, pad = 3, 3, 1, 1
    zp_x = np.int8(0)
    relu = True

    inp = rng.randint(-128, 128, (Hin, Win, Cin)).astype(np.int8)
    wgt = rng.randint(-128, 128, (Cout, Cin, Kh, Kw)).astype(np.int8)

    bias   = np.zeros(Cout, dtype=np.int32)
    m_int  = np.ones(Cout,  dtype=np.uint32)
    shift  = np.full(Cout, 10, dtype=np.uint8)
    zp_out = np.zeros(Cout, dtype=np.int8)

    gold = rs3_golden(inp, wgt, bias, m_int, shift, zp_out, relu, stride, pad, zp_x)

    os.makedirs(outdir, exist_ok=True)
    ib = input_to_banks(inp, Win, Cin)
    wb = weight_rs3_to_banks(wgt, Cin, Kw)
    ob = output_to_banks(gold, Win, Cout)

    for b in range(3):
        write_memh(os.path.join(outdir, f'input_bank{b}.memh'), bank_dict_to_flat(ib[b]))
        write_memh(os.path.join(outdir, f'weight_bank{b}.memh'), bank_dict_to_flat(wb[b]))
    for b in range(4):
        write_memh(os.path.join(outdir, f'expected_out_bank{b}.memh'), bank_dict_to_flat(ob[b]))
    write_quant_params(os.path.join(outdir, 'quant_params.txt'), bias, m_int, shift, zp_out)

    meta = dict(pe_mode='PE_RS3', cin=Cin, cout=Cout, hin=Hin, win=Win,
                hout=gold.shape[0], wout=gold.shape[1],
                kh=Kh, kw=Kw, stride=stride, padding=pad,
                activation='relu', zp_x=int(zp_x), seed=42,
                in_bank_addrs=[bank_max_addr(ib[i]) for i in range(3)],
                wt_bank_addrs=[bank_max_addr(wb[i]) for i in range(3)],
                out_bank_addrs=[bank_max_addr(ob[i]) for i in range(4)])
    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  RS3: {gold.shape} generated')


def gen_os1(rng, outdir):
    Cin, Cout, Hin, Win = 4, 4, 3, 20
    zp_x = np.int8(0)
    relu = True

    inp = rng.randint(-128, 128, (Hin, Win, Cin)).astype(np.int8)
    wgt = rng.randint(-128, 128, (Cout, Cin)).astype(np.int8)

    bias   = np.zeros(Cout, dtype=np.int32)
    m_int  = np.ones(Cout,  dtype=np.uint32)
    shift  = np.full(Cout, 12, dtype=np.uint8)
    zp_out = np.zeros(Cout, dtype=np.int8)

    gold = os1_golden(inp, wgt, bias, m_int, shift, zp_out, relu)

    os.makedirs(outdir, exist_ok=True)
    ib = input_to_banks(inp, Win, Cin)
    wb = weight_os1_to_banks(wgt, Cin)
    ob = output_to_banks(gold, Win, Cout)

    for b in range(3):
        write_memh(os.path.join(outdir, f'input_bank{b}.memh'), bank_dict_to_flat(ib[b]))
        write_memh(os.path.join(outdir, f'weight_bank{b}.memh'), bank_dict_to_flat(wb[b]))
    for b in range(4):
        write_memh(os.path.join(outdir, f'expected_out_bank{b}.memh'), bank_dict_to_flat(ob[b]))
    write_quant_params(os.path.join(outdir, 'quant_params.txt'), bias, m_int, shift, zp_out)

    meta = dict(pe_mode='PE_OS1', cin=Cin, cout=Cout, hin=Hin, win=Win,
                hout=gold.shape[0], wout=gold.shape[1],
                kh=1, kw=1, stride=1, padding=0,
                activation='relu', zp_x=int(zp_x), seed=42,
                in_bank_addrs=[bank_max_addr(ib[i]) for i in range(3)],
                wt_bank_addrs=[bank_max_addr(wb[i]) for i in range(3)],
                out_bank_addrs=[bank_max_addr(ob[i]) for i in range(4)])
    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  OS1: {gold.shape} generated')


def gen_dw3(rng, outdir):
    Cin, Hin, Win = 4, 3, 20
    Cout = Cin
    Kh, Kw, stride, pad = 3, 3, 1, 1
    zp_x = np.int8(0)
    relu = True

    inp = rng.randint(-128, 128, (Hin, Win, Cin)).astype(np.int8)
    wgt = rng.randint(-128, 128, (Cin, Kh, Kw)).astype(np.int8)

    bias   = np.zeros(Cin, dtype=np.int32)
    m_int  = np.ones(Cin,  dtype=np.uint32)
    shift  = np.full(Cin, 10, dtype=np.uint8)
    zp_out = np.zeros(Cin, dtype=np.int8)

    gold = dw3_golden(inp, wgt, bias, m_int, shift, zp_out, relu, stride, pad, zp_x)

    os.makedirs(outdir, exist_ok=True)
    ib = input_to_banks(inp, Win, Cin)
    wb = weight_dw3_to_banks(wgt, Kw)
    ob = output_to_banks(gold, Win, Cout)

    for b in range(3):
        write_memh(os.path.join(outdir, f'input_bank{b}.memh'), bank_dict_to_flat(ib[b]))
        write_memh(os.path.join(outdir, f'weight_bank{b}.memh'), bank_dict_to_flat(wb[b]))
    for b in range(4):
        write_memh(os.path.join(outdir, f'expected_out_bank{b}.memh'), bank_dict_to_flat(ob[b]))
    write_quant_params(os.path.join(outdir, 'quant_params.txt'), bias, m_int, shift, zp_out)

    meta = dict(pe_mode='PE_DW3', cin=Cin, cout=Cout, hin=Hin, win=Win,
                hout=gold.shape[0], wout=gold.shape[1],
                kh=Kh, kw=Kw, stride=stride, padding=pad,
                activation='relu', zp_x=int(zp_x), seed=42,
                in_bank_addrs=[bank_max_addr(ib[i]) for i in range(3)],
                wt_bank_addrs=[bank_max_addr(wb[i]) for i in range(3)],
                out_bank_addrs=[bank_max_addr(ob[i]) for i in range(4)])
    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  DW3: {gold.shape} generated')


def gen_mp5(rng, outdir):
    Cin, Hin, Win, pad = 4, 3, 20, 2
    Cout = Cin
    Hout = Hin + 2 * pad - 5 + 1

    inp = rng.randint(-128, 128, (Hin, Win, Cin)).astype(np.int8)
    gold = mp5_golden(inp, padding=pad)

    os.makedirs(outdir, exist_ok=True)
    ib = input_to_banks(inp, Win, Cin)
    ob = mp5_output_to_banks(gold, Win, Cout)

    for b in range(3):
        write_memh(os.path.join(outdir, f'input_bank{b}.memh'), bank_dict_to_flat(ib[b]))
    for b in range(4):
        write_memh(os.path.join(outdir, f'expected_out_bank{b}.memh'), bank_dict_to_flat(ob[b]))

    bias   = np.zeros(Cin, dtype=np.int32)
    m_int  = np.ones(Cin,  dtype=np.uint32)
    shift  = np.zeros(Cin,  dtype=np.uint8)
    zp_out = np.zeros(Cin, dtype=np.int8)
    write_quant_params(os.path.join(outdir, 'quant_params.txt'), bias, m_int, shift, zp_out)

    meta = dict(pe_mode='PE_MP5', cin=Cin, cout=Cout, hin=Hin, win=Win,
                hout=Hout, wout=Win,
                kh=5, kw=5, stride=1, padding=pad,
                activation='none', zp_x=0, seed=42,
                in_bank_addrs=[bank_max_addr(ib[i]) for i in range(3)],
                wt_bank_addrs=[0, 0, 0],
                out_bank_addrs=[bank_max_addr(ob[i]) for i in range(4)])
    with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  MP5: {gold.shape} generated  (Hout={Hout})')


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'vectors'))
    args = ap.parse_args()

    rng = np.random.RandomState(42)
    print('Generating Stage 8 test vectors …')
    gen_rs3(rng, os.path.join(args.out, 'rs3'))
    gen_os1(rng, os.path.join(args.out, 'os1'))
    gen_dw3(rng, os.path.join(args.out, 'dw3'))
    gen_mp5(rng, os.path.join(args.out, 'mp5'))
    print('Done.')


if __name__ == '__main__':
    main()
