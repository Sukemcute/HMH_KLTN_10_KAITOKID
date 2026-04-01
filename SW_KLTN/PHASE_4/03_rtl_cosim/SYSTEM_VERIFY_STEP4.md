# PHASE 4 — Bước 4: Kiểm thử **đúng hệ thống** (có test được không?)

**Có.** Bước 4 nên tách làm hai phần: **4A chỉ phần mềm** và **4B RTL + phần mềm so kết quả**.

---

## 4A — Phần mềm (không cần Vivado): “Hệ thống dữ liệu” có sẵn sàng không?

Trả lời câu hỏi: *input, weight, descriptor, golden P3/P4/P5 có đủ và khớp kích thước không?*

```bat
cd E:\KLTN_HMH_FINAL\SW_KLTN\PHASE_4
python 03_rtl_cosim\verify_system_readiness.py
REM Trước đó chạy: python 01_export\generate_descriptors.py --output 02_golden_data\
python 03_rtl_cosim\verify_system_readiness.py --strict-descriptors
```

Ngoài ra (đã có sẵn):

| Mục đích | Lệnh |
|----------|------|
| Model PyTorch ≡ golden hex P3/P4/P5 | `verify_full_model_outputs.py` |
| Block Upsample/QConcat/… | `verify_complex_blocks.py` |
| Conv L0/L1/L3/L17 | `verify_conv_layer.py` |
| Gộp nhiều mục | `run_phase4_verify.bat` |

→ **Test được hoàn toàn bằng phần mềm**: đây là “system test” của **chuỗi export + metadata + consistency**.  
Nó **không** chứng minh RTL đúng, nhưng chứng minh **mặt bằng dữ liệu** để RTL có cái để so.

---

## 4B — Phần mềm + mô phỏng RTL: “Các module `.sv` ghép lại có ra đúng tensor không?”

Cần **Vivado xsim** (hoặc simulator khác):

1. Compile toàn bộ PHASE_3 + TB hệ thống (`run_cosim.tcl`).
2. Chạy `tb_accel_top.sv` / `tb_accel_e2e.sv` — kiểm tra **CSR, DMA, FSM, nối dây**.
3. Chạy `tb_golden_check.sv` (khi IP thật sự chạy inference end-to-end):
   - Nạp `input_act.hex`, weight, descriptor vào model DDR.
   - Kick CSR → chờ done.
   - So vùng nhớ output với `golden_P3/P4/P5.hex` **hoặc** `$writememh` dump ra file rồi:

```bat
python 01_export\compare_golden_vs_rtl.py --golden ..\02_golden_data\golden_P3.hex --rtl <dump_P3.hex> --name P3 --shape 1,128,80,80
```

→ **Test được**: đây mới là “bước 4 đúng nghĩa hệ thống” cho **RTL**.

---

## Nên xây dựng PHASE 4 bước 4 như thế nào?

```
PHASE_4/
├── 03_rtl_cosim/
│   ├── verify_system_readiness.py   ← 4A (mới)
│   ├── SYSTEM_VERIFY_STEP4.md       ← tài liệu này
│   ├── run_ppu_golden.tcl           ← đã có: PPU trong hệ thống con
│   ├── run_cosim.tcl                ← compile full RTL
│   └── tb_golden_check.sv           ← 4B: so P3/P4/P5 (khi DUT sẵn sàng)
```

**Thứ tự khuyến nghị:**

1. **4A pass** (`verify_system_readiness` + `verify_full_model_outputs`).
2. **PPU pass** (`tb_ppu_golden`) — đã làm.
3. **`tb_accel_top` / `tb_accel_e2e` pass** — không treo, bus đúng.
4. **4B** `tb_golden_check` + `compare_golden_vs_rtl` — bit-exact P3/P4/P5.

---

## Tóm tắt

| Câu hỏi | Trả lời |
|---------|---------|
| Bước 4 “đúng hệ thống” test **bằng phần mềm** được không? | **Có — phần 4A** (dữ liệu + nhất quán + model ≡ golden). |
| Làm sao biết **RTL ghép đúng**? | **Phần 4B**: sim + so hex với golden (không thay bằng chỉ nhìn file `.sv`). |
