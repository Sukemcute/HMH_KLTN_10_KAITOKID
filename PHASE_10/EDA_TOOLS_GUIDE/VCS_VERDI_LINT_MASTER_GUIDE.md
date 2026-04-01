# HƯỚNG DẪN CHUYÊN SÂU: VCS + VERDI + LINT
## Quy Trình Từ SystemVerilog → Compile → Debug → Signoff

> **Mục tiêu**: Master 3 tools cốt lõi trong verification flow
> **Áp dụng**: YOLOv10n INT8 Accelerator RTL (PHASE_10)
> **Tools**: Synopsys VCS (compile+simulate), Synopsys Verdi (waveform debug), SpyGlass/HAL LINT

---

# ════════════════════════════════════════════════════════════════
# PHẦN 0: TỔNG QUAN QUY TRÌNH — BIG PICTURE
# ════════════════════════════════════════════════════════════════

```
DESIGN FLOW:

  ┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ SystemVerilog│───►│  LINT    │───►│   VCS    │───►│  VERDI   │
  │ Source Code  │    │ (syntax  │    │(compile  │    │(waveform │
  │ (.sv files)  │    │  check)  │    │ +simulate│    │  debug)  │
  └─────────────┘    └──────────┘    │ +coverage)│    └──────────┘
                                     └──────────┘
  Bước 1: Viết RTL          Bước 2: LINT check     Bước 3: Compile
  (.sv modules + TB)        (syntax, style,         (VCS tạo binary)
                             connectivity)
                                                    Bước 4: Simulate
                                                    (chạy testbench)

                                                    Bước 5: Debug
                                                    (Verdi xem sóng)

  INPUT → PROCESS → OUTPUT mỗi tool:
  ┌───────────┬─────────────────────────┬──────────────────────────┐
  │ Tool      │ INPUT                   │ OUTPUT                   │
  ├───────────┼─────────────────────────┼──────────────────────────┤
  │ LINT      │ .sv files, waiver file  │ lint report (.rpt)       │
  │           │                         │ violations list          │
  │           │                         │ cleaned code             │
  ├───────────┼─────────────────────────┼──────────────────────────┤
  │ VCS       │ .sv files, TB, +define  │ simv (executable binary) │
  │           │                         │ *.fsdb (waveform dump)   │
  │           │                         │ simulation log           │
  │           │                         │ coverage database        │
  ├───────────┼─────────────────────────┼──────────────────────────┤
  │ VERDI     │ .sv files (source)      │ Waveform viewer (GUI)    │
  │           │ *.fsdb (waveform data)  │ Signal trace             │
  │           │                         │ Schematic view           │
  │           │                         │ Source annotation         │
  └───────────┴─────────────────────────┴──────────────────────────┘
```

---

# ════════════════════════════════════════════════════════════════
# PHẦN 1: VCS — SYNOPSYS VCS COMPILER & SIMULATOR
# ════════════════════════════════════════════════════════════════

## 1.1. VCS Là Gì?

```
VCS = Verilog Compiler Simulator (Synopsys)
  - Compile SystemVerilog → native executable (simv)
  - Simulate testbench → chạy DUT, output results
  - Dump waveform → .fsdb file cho Verdi
  - Coverage → functional + code coverage database

2 bước chính:
  Bước 1: vcs (compile) → tạo ./simv
  Bước 2: ./simv (run) → tạo kết quả + waveform
```

## 1.2. VCS Compile — Lệnh Cơ Bản

### Compile đơn giản nhất:
```bash
vcs -sverilog -full64 -debug_access+all \
    file1.sv file2.sv tb_top.sv \
    -o simv
```

### Compile với đầy đủ options (recommended):
```bash
vcs -sverilog \
    -full64 \
    -debug_access+all \
    -timescale=1ns/1ps \
    +incdir+./packages \
    +define+SIMULATION \
    -f filelist.f \
    -top tb_top \
    -o simv \
    -l compile.log \
    +lint=all \
    -assert svaext \
    -cm line+cond+fsm+tgl+branch \
    -cm_dir ./coverage.vdb \
    +v2k \
    -LDFLAGS -Wl,--no-as-needed
```

### Giải thích từng option:

```
┌──────────────────────────┬──────────────────────────────────────────────┐
│ Option                   │ Ý nghĩa                                      │
├──────────────────────────┼──────────────────────────────────────────────┤
│ -sverilog                │ Enable SystemVerilog parsing                  │
│ -full64                  │ 64-bit compilation (bắt buộc trên Linux 64)  │
│ -debug_access+all        │ Enable FULL debug (waveform, breakpoint, ...) │
│ -timescale=1ns/1ps       │ Default timescale                            │
│ +incdir+./packages       │ Include directory (cho `include)              │
│ +define+SIMULATION       │ Preprocessor define                          │
│ -f filelist.f            │ File list (1 file per line)                   │
│ -top tb_top              │ Top-level module name                         │
│ -o simv                  │ Output executable name                        │
│ -l compile.log           │ Log file                                      │
│ +lint=all                │ Enable lint warnings during compile           │
│ -assert svaext           │ Enable SVA assertions                        │
│ -cm line+cond+fsm+tgl   │ Coverage metrics                             │
│ -cm_dir ./coverage.vdb   │ Coverage database directory                   │
│ +v2k                     │ Verilog-2001 compatibility                    │
│ -CFLAGS                  │ C compiler flags (cho DPI)                    │
│ -LDFLAGS                 │ Linker flags                                  │
│ -R                       │ Compile AND run immediately                   │
│ -gui                     │ Launch Verdi GUI after compile                │
└──────────────────────────┴──────────────────────────────────────────────┘
```

### File list format (filelist.f):
```
# filelist.f — VCS compile order
# Packages first
./packages/accel_pkg.sv
./packages/desc_pkg.sv
./packages/csr_pkg.sv

# RTL modules (dependency order)
./01_dsp_pair/rtl/dsp_pair_int8.sv
./02_pe_unit/rtl/pe_unit.sv
./03_column_reduce/rtl/column_reduce.sv
./04_comparator_tree/rtl/comparator_tree.sv
./05_silu_lut/rtl/silu_lut.sv
./06_ppu/rtl/ppu.sv

# Testbench
./06_ppu/tb/tb_ppu.sv
```

## 1.3. VCS Run — Chạy Simulation

### Chạy cơ bản:
```bash
./simv                         # Chạy simulation
./simv -l sim.log              # Với log file
./simv +vcs+finish+100us       # Timeout sau 100us
```

### Chạy với waveform dump (cho Verdi):
```bash
# Cách 1: Dump tất cả signals (FSDB format cho Verdi)
./simv -l sim.log +fsdbfile+waveform.fsdb +fsdb+all

# Cách 2: Dùng $fsdbDump trong testbench
# (thêm vào initial block của TB):
#   initial begin
#     $fsdbDumpfile("waveform.fsdb");
#     $fsdbDumpvars(0, tb_top);   // 0 = dump tất cả hierarchy
#   end

./simv -l sim.log
```

### Các option runtime quan trọng:
```
┌──────────────────────────────┬──────────────────────────────────────┐
│ Option                       │ Ý nghĩa                              │
├──────────────────────────────┼──────────────────────────────────────┤
│ -l sim.log                   │ Log output to file                   │
│ +fsdbfile+wave.fsdb          │ FSDB waveform file name              │
│ +fsdb+all                    │ Dump ALL signals                     │
│ +fsdb+delta                  │ Dump delta-cycle changes             │
│ +vcs+finish+<time>           │ Force finish after time              │
│ +ntb_random_seed=12345       │ Set random seed                      │
│ +vcs+lic+wait                │ Wait for license                     │
│ -cm line+cond+fsm+tgl       │ Enable runtime coverage              │
│ -cm_dir ./coverage.vdb       │ Coverage output dir                  │
│ +UVM_TESTNAME=test1          │ UVM test name (nếu dùng UVM)        │
│ +vcs+stop+<time>             │ Stop (pause) at time                 │
│ -gui                         │ Launch Verdi GUI                     │
│ -ucli                        │ Interactive UCLI debugger             │
└──────────────────────────────┴──────────────────────────────────────┘
```

## 1.4. VCS + FSDB Dump (Trong Testbench)

### Code thêm vào TB để dump waveform:
```systemverilog
module tb_top;
  // ... DUT instantiation ...

  // ═══════════ FSDB Waveform Dump ═══════════
  initial begin
    // Tạo file FSDB cho Verdi
    $fsdbDumpfile("waveform.fsdb");

    // Dump tất cả signals từ tb_top trở xuống (depth = 0 = unlimited)
    $fsdbDumpvars(0, tb_top);

    // Hoặc dump selective:
    // $fsdbDumpvars(1, tb_top);           // Chỉ level 1 (top signals)
    // $fsdbDumpvars(0, tb_top.u_dut);     // Từ DUT trở xuống
    // $fsdbDumpvars("+mda");              // Dump multi-dimensional arrays
    // $fsdbDumpvars("+struct");           // Dump struct fields

    // Force finish nếu quá lâu
    #10_000_000;  // 10ms
    $display("TIMEOUT!");
    $finish;
  end

  // Dump memory arrays (cần explicit cho arrays):
  initial begin
    $fsdbDumpMDA(0, tb_top);  // Dump Multi-Dimensional Arrays
  end
endmodule
```

### FSDB vs VCD comparison:
```
┌──────────┬──────────────────────┬──────────────────────────┐
│          │ FSDB (Synopsys)      │ VCD (Standard)           │
├──────────┼──────────────────────┼──────────────────────────┤
│ Viewer   │ Verdi                │ GTKWave, ModelSim        │
│ Size     │ 5-10× nhỏ hơn       │ Lớn                      │
│ Speed    │ Nhanh hơn           │ Chậm                     │
│ Feature  │ Struct, MDA, enum    │ Cơ bản                   │
│ Command  │ $fsdbDumpfile       │ $dumpfile                │
│          │ $fsdbDumpvars       │ $dumpvars                │
└──────────┴──────────────────────┴──────────────────────────┘
→ LUÔN DÙNG FSDB nếu có Verdi. VCD chỉ khi không có license.
```

## 1.5. VCS Compile + Run Script Mẫu (Cho PHASE_10)

```bash
#!/bin/bash
# run_vcs.sh — Compile & simulate PHASE_10 stage_2 modules

PROJ="$HOME/HMH_KLTN/PHASE_10/stage_2"
TB=$1  # e.g., tb_dsp_pair_int8

echo "=== VCS Compile: $TB ==="

# Step 1: Compile
vcs -sverilog -full64 \
    -debug_access+all \
    -timescale=1ns/1ps \
    +incdir+${PROJ}/packages \
    -f ${PROJ}/sim/filelist.f \
    ${PROJ}/*/tb/${TB}.sv \
    -top ${TB} \
    -o ./simv_${TB} \
    -l compile_${TB}.log \
    +lint=all \
    2>&1 | tee vcs_compile.log

if [ $? -ne 0 ]; then
    echo "COMPILE FAILED! Check compile_${TB}.log"
    exit 1
fi

echo "=== VCS Run: $TB ==="

# Step 2: Simulate + dump FSDB
./simv_${TB} \
    -l sim_${TB}.log \
    +fsdbfile+${TB}.fsdb \
    +fsdb+all \
    2>&1 | tee vcs_sim.log

echo "=== Done. Waveform: ${TB}.fsdb ==="
echo "=== Open in Verdi: verdi -ssf ${TB}.fsdb ==="
```

## 1.6. VCS Output Files

```
Sau compile + run, VCS tạo:
┌────────────────────────┬──────────────────────────────────────┐
│ File/Dir               │ Ý nghĩa                              │
├────────────────────────┼──────────────────────────────────────┤
│ simv                   │ Executable (TÁI SỬ DỤNG nếu RTL     │
│                        │ không đổi, chỉ đổi runtime args)    │
│ simv.daidir/           │ Compile database (cho Verdi source)  │
│ csrc/                  │ C source cache (tái sử dụng)         │
│ *.fsdb                 │ Waveform data (cho Verdi)            │
│ compile.log            │ Compile messages                     │
│ sim.log                │ Simulation output ($display, etc.)   │
│ coverage.vdb/          │ Coverage database                    │
│ ucli.key               │ UCLI command history                 │
│ inter.vpd              │ VPD waveform (nếu dùng VPD)         │
│ DVEfiles/              │ DVE debug files                      │
└────────────────────────┴──────────────────────────────────────┘

TÁI SỬ DỤNG:
  - simv: chạy lại ./simv với args khác (không cần recompile)
  - simv.daidir: Verdi dùng để link source code
  - coverage.vdb: merge nhiều runs bằng urg tool
```

---

# ════════════════════════════════════════════════════════════════
# PHẦN 2: VERDI — SYNOPSYS VERDI WAVEFORM DEBUGGER
# ════════════════════════════════════════════════════════════════

## 2.1. Verdi Là Gì?

```
Verdi = Automated Debug System (Synopsys)
  - Xem waveform từ FSDB file
  - Trace signal qua hierarchy (click signal → xem source)
  - Schematic view (xem kết nối modules)
  - Source code annotation (highlight active signals)
  - Protocol analyzer, assertion debug, coverage analysis
```

## 2.2. Khởi Động Verdi

### Cách 1: Mở standalone với FSDB file
```bash
verdi -ssf waveform.fsdb &
```

### Cách 2: Mở với source code + FSDB
```bash
verdi -sverilog \
      -f filelist.f \
      -ssf waveform.fsdb \
      -top tb_top &
```

### Cách 3: Mở từ VCS (compile + GUI)
```bash
vcs -sverilog -full64 -debug_access+all -f filelist.f -top tb_top -o simv -gui
# → Verdi opens automatically after compile
```

### Cách 4: Mở với VCS database
```bash
verdi -simflow \
      -simBin ./simv \
      -ssf waveform.fsdb &
```

## 2.3. Verdi GUI — Các Cửa Sổ Chính

```
┌─────────────────────────────────────────────────────────────┐
│                    VERDI GUI LAYOUT                          │
│                                                             │
│ ┌──────────────────┐  ┌────────────────────────────────────┐│
│ │ Source Browser    │  │         nWave (Waveform)           ││
│ │ (code + hier.)   │  │  ┌─────────────────────────────┐   ││
│ │                  │  │  │ Signal names │ Waveform view │   ││
│ │ ■ tb_top         │  │  │  clk         │ ┌┐┌┐┌┐┌┐┌┐  │   ││
│ │  ├── u_dut       │  │  │  rst_n       │ ────┐         │   ││
│ │  │  ├── u_pe0    │  │  │  pe_en       │    ┌──┐      │   ││
│ │  │  ├── u_pe1    │  │  │  psum[31:0]  │ ====≡====    │   ││
│ │  │  └── u_ppu    │  │  │  act_out[7:0]│   ╔══╗       │   ││
│ │  └── ...         │  │  └─────────────────────────────┘   ││
│ └──────────────────┘  └────────────────────────────────────┘│
│                                                             │
│ ┌──────────────────────────────────────────────────────────┐│
│ │              nSchema (Schematic)                          ││
│ │  ┌────┐    ┌────┐    ┌────┐                              ││
│ │  │ PE │───►│ COL│───►│PPU │                              ││
│ │  │UNIT│    │ RED│    │    │                              ││
│ │  └────┘    └────┘    └────┘                              ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 2.4. Verdi nWave — Xem & Thao Tác Waveform

### Thêm signal vào waveform:

```
CÁCH 1: Kéo thả (Drag & Drop)
  - Từ Source Browser: chọn signal → kéo vào nWave window

CÁCH 2: Phím tắt
  - Chọn signal trong source code → nhấn Ctrl+W (Add to Wave)
  - Hoặc: Ctrl+4 (Add to Wave, nhóm mới)

CÁCH 3: Get Signals dialog
  - Menu: Signal → Get Signals (hoặc phím tắt G)
  - Gõ tên signal: tb_top.u_dut.pe_en → Add
  - Wildcard: tb_top.u_dut.pe_psum* → Add all matching

CÁCH 4: Add hierarchy
  - Click phải lên module trong hierarchy → "Add to Wave"
  - Sẽ add TẤT CẢ signals của module đó

CÁCH 5: Command line trong Verdi console
  - wvAddSignal "/tb_top/u_dut/pe_en"
  - wvAddSignal "/tb_top/u_dut/psum_out[0]"
```

### Điều hướng thời gian:

```
┌─────────────────────┬────────────────────────────────────────────┐
│ Phím tắt            │ Chức năng                                  │
├─────────────────────┼────────────────────────────────────────────┤
│ Z                   │ Zoom IN (phóng to)                         │
│ Shift+Z             │ Zoom OUT (thu nhỏ)                        │
│ F                   │ Zoom FIT (hiển thị toàn bộ waveform)      │
│ W                   │ Zoom to WINDOW (vùng đã select)            │
│ ← →                │ Scroll trái/phải                           │
│ Home                │ Về đầu simulation                          │
│ End                 │ Về cuối simulation                         │
│ Ctrl+G              │ Go to TIME (nhập thời gian cụ thể)        │
│ C                   │ Set CURSOR tại vị trí click                │
│ Shift+C             │ Set MARKER (marker thứ 2 để đo khoảng cách)│
│ N                   │ Next EDGE (rising/falling) của signal chọn │
│ Shift+N             │ Previous EDGE                              │
│ B                   │ Next rising EDGE (posedge)                 │
│ Shift+B             │ Previous rising edge                       │
└─────────────────────┴────────────────────────────────────────────┘
```

### Thao tác signal:

```
┌─────────────────────┬────────────────────────────────────────────┐
│ Phím tắt            │ Chức năng                                  │
├─────────────────────┼────────────────────────────────────────────┤
│ Ctrl+W              │ ADD signal đang chọn vào waveform          │
│ Delete              │ REMOVE signal khỏi waveform                │
│ Ctrl+A              │ Select ALL signals                         │
│ R                   │ Đổi RADIX (hex/dec/bin/signed_dec/ascii)   │
│ Ctrl+R              │ Cycle qua các radix                        │
│ X                   │ Expand BUS (hiện từng bit riêng)           │
│ Ctrl+X              │ Collapse BUS (gộp lại)                     │
│ Ctrl+Shift+A        │ Add ANALOG display cho signal              │
│ I                   │ Invert signal (đảo 0/1)                   │
│ Ctrl+D              │ Duplicate signal (copy thêm 1 bản)        │
│ Ctrl+Up/Down        │ Di chuyển signal lên/xuống trong list      │
│ G                   │ Group signals (tạo nhóm)                   │
│ Alt+G               │ Ungroup                                    │
│ S                   │ Split grouped signals                      │
└─────────────────────┴────────────────────────────────────────────┘
```

### Trace & Debug:

```
┌─────────────────────┬────────────────────────────────────────────┐
│ Phím tắt            │ Chức năng                                  │
├─────────────────────┼────────────────────────────────────────────┤
│ Ctrl+T              │ TRACE DRIVER (ai drive signal này?)        │
│ Ctrl+Shift+T        │ TRACE LOAD (signal này đi đến đâu?)       │
│ Ctrl+E              │ Find in SOURCE code                        │
│ Ctrl+F              │ FIND signal by name                        │
│ Ctrl+Shift+F        │ FIND value (tìm thời điểm signal = value) │
│ Ctrl+B              │ Set BOOKMARK                               │
│ F5                  │ FORCE value (interactive sim)               │
│ Ctrl+P              │ Print waveform                             │
│ Ctrl+Shift+S        │ Save waveform SESSION (tái sử dụng!)      │
│ Ctrl+Shift+O        │ Open saved session                         │
└─────────────────────┴────────────────────────────────────────────┘
```

### Hiển thị nâng cao:

```
┌─────────────────────┬────────────────────────────────────────────┐
│ Phím tắt            │ Chức năng                                  │
├─────────────────────┼────────────────────────────────────────────┤
│ Ctrl+Shift+A        │ ANALOG display (cho bus signals → waveform)│
│ Alt+C               │ COLOR signal (đổi màu để phân biệt)       │
│ Ctrl+H              │ HIGHLIGHT transitions                      │
│ T                   │ Toggle TRANSACTION display                 │
│ Ctrl+M              │ Add MARKER (đánh dấu thời điểm)           │
│ Ctrl+Shift+M        │ Remove marker                              │
│ Ctrl+L              │ Add LABEL cho group                        │
└─────────────────────┴────────────────────────────────────────────┘
```

## 2.5. Verdi Source Browser — Phím Tắt

```
┌─────────────────────┬────────────────────────────────────────────┐
│ Phím tắt            │ Chức năng                                  │
├─────────────────────┼────────────────────────────────────────────┤
│ Double-click signal │ Highlight driver/load trong source         │
│ Ctrl+Click signal   │ Jump to DEFINITION                        │
│ Ctrl+Shift+Click    │ Jump to INSTANTIATION                     │
│ Ctrl+1              │ Toggle Source view                        │
│ Ctrl+2              │ Toggle nWave view                         │
│ Ctrl+3              │ Toggle nSchema view (schematic)           │
│ F2                  │ Navigate FORWARD in history                │
│ F3                  │ Navigate BACKWARD in history               │
│ Ctrl+F              │ Find in source                            │
│ Ctrl+G              │ Go to line number                         │
└─────────────────────┴────────────────────────────────────────────┘
```

## 2.6. Verdi nSchema — Xem Schematic

```
Mở nSchema:
  Menu: Tools → New nSchema Window
  Hoặc: Ctrl+3

Trong nSchema:
  Double-click module → dive INTO hierarchy
  Right-click → Show Connectivity
  Ctrl+Click pin → trace signal
  + / - : Zoom in/out
  F : Fit to window
  H : Highlight selected net
```

## 2.7. Verdi Console Commands (Tcl)

```tcl
# Mở FSDB file
fsdbOpenFile "waveform.fsdb"

# Add signals
wvAddSignal "/tb_top/clk" "/tb_top/rst_n" "/tb_top/u_dut/pe_en"

# Add tất cả signals của 1 module
wvAddSignal "/tb_top/u_dut/*"

# Go to time
wvGoToTime 1000ns

# Set radix
wvSetRadix -hexadecimal "/tb_top/u_dut/psum_out"
wvSetRadix -signedDecimal "/tb_top/u_dut/psum_out"

# Zoom
wvZoomFit
wvZoomIn
wvZoomOut

# Save session (TÁI SỬ DỤNG!)
wvSaveSignal "my_debug.ses"

# Load session
wvRestoreSignal "my_debug.ses"

# Search value
wvSearchValue -value "8'hFF" "/tb_top/u_dut/act_out"

# Bookmark
wvCreateBookmark -name "error_point" -time 5000ns
```

## 2.8. Session Save/Restore (RẤT QUAN TRỌNG)

```
Verdi cho phép SAVE toàn bộ setup → reload sau:

Save: File → Save Signal / Ctrl+Shift+S
  → Lưu: signal list + radix + color + group + zoom level

Restore: File → Restore Signal / Ctrl+Shift+O
  → Khôi phục y hệt session cũ

Dùng khi:
  - Compile lại VCS (FSDB mới) → mở Verdi → load session → cùng signals
  - Chia sẻ debug setup với teammate
  - Quay lại debug sau khi fix code
```

---

# ════════════════════════════════════════════════════════════════
# PHẦN 3: LINT — STATIC ANALYSIS
# ════════════════════════════════════════════════════════════════

## 3.1. LINT Là Gì?

```
LINT = Static code analysis (KHÔNG chạy simulation)
  - Kiểm tra TRƯỚC khi simulate
  - Phát hiện: syntax errors, width mismatch, unconnected ports,
    clock domain crossing, combinational loops, latch inference, ...
  - Tools: Synopsys SpyGlass, Cadence HAL, Mentor Questa Lint,
           Verilator (open-source)

Tại sao cần:
  VCS có thể compile code có bug logic → chạy được nhưng SAI
  LINT bắt bug TRƯỚC khi mất thời gian simulate
```

## 3.2. Verilator Lint (Open-Source, Khuyến nghị cho dự án)

### Cài đặt:
```bash
# Ubuntu/WSL:
sudo apt-get install verilator

# Hoặc từ source:
git clone https://github.com/verilator/verilator
cd verilator && autoconf && ./configure && make && sudo make install
```

### Chạy lint cơ bản:
```bash
verilator --lint-only -Wall \
    --top-module tb_dsp_pair_int8 \
    -I./packages \
    ./packages/accel_pkg.sv \
    ./01_dsp_pair/rtl/dsp_pair_int8.sv \
    ./01_dsp_pair/tb/tb_dsp_pair_int8.sv
```

### Chạy lint cho toàn bộ project:
```bash
verilator --lint-only -Wall -Wno-fatal \
    --top-module subcluster_datapath \
    -I./stage_2/packages \
    -f filelist_all.f \
    2>&1 | tee lint_report.log
```

### Verilator warnings giải thích:

```
┌──────────────────────┬──────────────────────────────────────────────┐
│ Warning              │ Ý nghĩa & Cách fix                          │
├──────────────────────┼──────────────────────────────────────────────┤
│ UNUSED               │ Signal declared nhưng không dùng             │
│                      │ Fix: xóa hoặc /* verilator lint_off UNUSED */│
│ UNDRIVEN             │ Signal đọc nhưng không ai drive              │
│                      │ Fix: kết nối input hoặc gán default          │
│ WIDTH                │ Width mismatch trong assign/connect          │
│                      │ Fix: explicit cast hoặc resize               │
│ PINMISSING           │ Module port không kết nối                    │
│                      │ Fix: kết nối hoặc .port() (leave open)       │
│ CASEINCOMPLETE       │ case statement thiếu default                 │
│                      │ Fix: thêm default clause                     │
│ LATCH                │ Combinational block tạo latch (thiếu else)   │
│                      │ Fix: thêm else hoặc default assignment       │
│ MULTIDRIVEN          │ Signal bị drive từ nhiều always block        │
│                      │ Fix: gộp logic vào 1 block                   │
│ BLKSEQ               │ Blocking assign (=) trong sequential block   │
│                      │ Fix: dùng non-blocking (<=) trong always_ff  │
│ COMBDLY              │ Non-blocking (<=) trong combinational block   │
│                      │ Fix: dùng blocking (=) trong always_comb     │
│ INITIALDLY           │ Delay trong initial block (không synth)      │
│                      │ Fix: OK cho testbench, remove cho RTL        │
└──────────────────────┴──────────────────────────────────────────────┘
```

### Suppress warnings (khi intentional):
```systemverilog
// Suppress cho 1 dòng:
/* verilator lint_off UNUSED */ logic unused_signal;

// Suppress cho block:
/* verilator lint_off WIDTH */
assign short_bus = long_bus;   // intentional truncation
/* verilator lint_on WIDTH */

// Suppress trong command line:
verilator --lint-only -Wno-WIDTH -Wno-UNUSED ...
```

## 3.3. SpyGlass Lint (Synopsys, nếu có license)

### Chạy SpyGlass:
```bash
# Tạo project
spyglass -project my_proj.prj

# Hoặc batch mode:
spyglass -batch -goal lint \
    -designread "filelist.f" \
    -top subcluster_datapath \
    -reportdir ./spyglass_report
```

### SpyGlass rules phổ biến:
```
W_446: Unconnected port
W_116: Inferred latch
W_164: Truncation in assignment
W_224: Multi-driven net
W_240: Clock domain crossing
W_391: Missing case default
W_528: Width mismatch
W_110: Missing sensitivity list (always block)
```

## 3.4. Lint Script Mẫu (Cho PHASE_10)

```bash
#!/bin/bash
# lint_check.sh — Run Verilator lint on all PHASE_10 modules

PROJ="$HOME/HMH_KLTN/PHASE_10"
S2="${PROJ}/stage_2"
S4="${PROJ}/stage_4"
S5="${PROJ}/stage_5"

echo "=== LINT CHECK: PHASE_10 ==="

# Stage 2: Compute atoms
for module in dsp_pair_int8 pe_unit column_reduce comparator_tree silu_lut ppu; do
    echo "--- Checking: $module ---"
    verilator --lint-only -Wall -Wno-fatal \
        -I${S2}/packages \
        ${S2}/packages/*.sv \
        ${S2}/*/rtl/*.sv \
        --top-module $module \
        2>&1 | grep -E "Error|Warning" | head -20
done

# Stage 5: Subcluster
echo "--- Checking: subcluster_datapath ---"
verilator --lint-only -Wall -Wno-fatal \
    -I${S2}/packages \
    ${S2}/packages/*.sv \
    ${S2}/*/rtl/*.sv \
    ${S4}/*/rtl/*.sv \
    ${S5}/rtl/*.sv \
    --top-module subcluster_datapath \
    2>&1 | tee lint_subcluster.log

echo "=== DONE. Check lint_subcluster.log ==="
```

---

# ════════════════════════════════════════════════════════════════
# PHẦN 4: QUY TRÌNH HOÀN CHỈNH — STEP BY STEP
# ════════════════════════════════════════════════════════════════

## 4.1. Quy Trình 7 Bước

```
BƯỚC 1: Viết RTL (.sv)
    │
    ▼
BƯỚC 2: LINT check (Verilator)
    │   verilator --lint-only -Wall ...
    │   Fix tất cả warnings (trừ intentional)
    ▼
BƯỚC 3: Viết Testbench (.sv)
    │   - DUT instantiation
    │   - Clock generation
    │   - Stimulus + golden comparison
    │   - $fsdbDumpfile + $fsdbDumpvars
    ▼
BƯỚC 4: VCS Compile
    │   vcs -sverilog -full64 -debug_access+all ...
    │   Check compile.log cho errors
    ▼
BƯỚC 5: VCS Run
    │   ./simv +fsdbfile+wave.fsdb +fsdb+all
    │   Check sim.log cho PASS/FAIL
    ▼
BƯỚC 6: Verdi Debug (nếu FAIL)
    │   verdi -ssf wave.fsdb -f filelist.f
    │   - Add failing signals
    │   - Trace driver/load
    │   - Find error time
    │   - Compare expected vs actual
    ▼
BƯỚC 7: Fix RTL → quay lại BƯỚC 2
    │   (simv có thể reuse nếu chỉ đổi testbench)
    ▼
DONE: ALL PASS → tiến sang module tiếp theo
```

## 4.2. Ví Dụ Cụ Thể: Debug PPU Rounding

```
Scenario: tb_ppu FAIL ở test half-up rounding

BƯỚC 1: Xem sim.log
  → "FAIL: Test 1 lane 5: expected 150, got 149"
  → Sai 1 LSB → nghi rounding

BƯỚC 2: Mở Verdi
  verdi -ssf tb_ppu.fsdb -f filelist.f &

BƯỚC 3: Add signals
  - Ctrl+W: tb_top.u_dut.biased_s1[5]     (sau bias add)
  - Ctrl+W: tb_top.u_dut.y_raw_s2[5]      (sau requant)
  - Ctrl+W: tb_top.u_dut.y_act_s3[5]      (sau activation)
  - Ctrl+W: tb_top.u_dut.act_out[5]       (final output)

BƯỚC 4: Navigate to error time
  - Ctrl+G → nhập thời gian từ log (ví dụ 500ns)
  - Xem giá trị tại cursor:
    biased_s1[5] = 300    ✓
    mult (internal) = 300 × M = ???
    → Click Ctrl+T để trace mult signal

BƯỚC 5: Phát hiện
  - mult = 300, shift = 1
  - Expected: (300 + 1) >> 1 = 150 (half-up)
  - Actual:   300 >> 1 = 150       (floor = cũng 150!)
  - Thử case khác: mult = 301, shift = 1
  - Expected: (301 + 1) >> 1 = 151
  - Actual:   301 >> 1 = 150       ← SAI! Floor thay vì half-up!

BƯỚC 6: Fix RTL
  - Mở ppu.sv, tìm dòng shift
  - Thêm rounding offset: + (1 <<< (sh-1))
  - Save → recompile → rerun → PASS!
```

## 4.3. Tips & Best Practices

```
1. LUÔN chạy LINT trước VCS
   → Bắt lỗi syntax nhanh hơn compile rất nhiều
   → LINT = 2 giây, VCS compile = 30-60 giây

2. LUÔN dump FSDB trong testbench
   → Thêm $fsdbDumpfile + $fsdbDumpvars vào MỌI TB
   → Khi FAIL → mở Verdi ngay, không cần rerun

3. Dùng Verdi SESSION save
   → Save signal list sau khi setup debug
   → Next time: chỉ cần load session → instantly ready

4. VCS incremental compile
   → vcs -Mupdate  (chỉ recompile files thay đổi)
   → Tiết kiệm 10-50% compile time

5. Dùng +define+DEBUG cho conditional dump
   → Trong RTL: `ifdef DEBUG $display(...) `endif
   → Compile: vcs +define+DEBUG ... (bật) hoặc bỏ (tắt)

6. Coverage
   → Compile: -cm line+cond+fsm+tgl+branch
   → Run: -cm_dir coverage.vdb
   → Report: urg -dir coverage.vdb -report coverage_html
   → Mở coverage_html/dashboard.html trong browser

7. Parallel compilation
   → vcs -j8 (8 parallel compile jobs)
   → Tiết kiệm 60-70% compile time cho project lớn
```

---

# ════════════════════════════════════════════════════════════════
# PHẦN 5: TỔNG HỢP PHÍM TẮT — QUICK REFERENCE
# ════════════════════════════════════════════════════════════════

## 5.1. VCS Command Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────┐
│ VCS COMPILE                                                        │
├────────────────────────────────────────────────────────────────────┤
│ vcs -sverilog -full64 ...              Compile SV                  │
│ vcs -sverilog -R ...                   Compile + Run               │
│ vcs -Mupdate ...                       Incremental compile         │
│ vcs -j8 ...                            Parallel (8 jobs)           │
│ vcs -gui ...                           Launch Verdi after compile  │
├────────────────────────────────────────────────────────────────────┤
│ VCS RUN                                                            │
├────────────────────────────────────────────────────────────────────┤
│ ./simv                                 Run simulation              │
│ ./simv -l sim.log                      With log                    │
│ ./simv +fsdbfile+w.fsdb +fsdb+all      Dump all to FSDB            │
│ ./simv +vcs+finish+100us               Timeout at 100us            │
│ ./simv -gui                            Run with Verdi GUI          │
│ ./simv -ucli                           Interactive debug            │
│ ./simv +ntb_random_seed=42             Set random seed             │
└────────────────────────────────────────────────────────────────────┘
```

## 5.2. Verdi Keyboard Shortcut Cheat Sheet

```
┌──────────────────────────────────────────────────────────────────┐
│ NAVIGATION                                                        │
├──────────┬───────────────────────────────────────────────────────┤
│ Z        │ Zoom in                                               │
│ Shift+Z  │ Zoom out                                              │
│ F        │ Zoom fit (all)                                        │
│ W        │ Zoom to selection                                     │
│ ← →     │ Scroll left/right                                     │
│ Home     │ Go to start                                           │
│ End      │ Go to end                                             │
│ Ctrl+G   │ Go to time                                            │
│ C        │ Place cursor                                          │
│ N        │ Next edge                                             │
│ B        │ Next posedge                                          │
├──────────┼───────────────────────────────────────────────────────┤
│ SIGNAL MANAGEMENT                                                 │
├──────────┼───────────────────────────────────────────────────────┤
│ Ctrl+W   │ Add signal to wave                                    │
│ G        │ Get signals (search dialog)                           │
│ Delete   │ Remove signal                                         │
│ R        │ Change radix (hex/dec/bin)                            │
│ X        │ Expand bus                                            │
│ Ctrl+D   │ Duplicate signal                                      │
│ I        │ Invert signal                                         │
│ Alt+C    │ Change color                                          │
├──────────┼───────────────────────────────────────────────────────┤
│ DEBUG / TRACE                                                     │
├──────────┼───────────────────────────────────────────────────────┤
│ Ctrl+T   │ Trace driver (who drives this?)                       │
│ Ctrl+Sh+T│ Trace load (where does this go?)                      │
│ Ctrl+E   │ Find in source code                                   │
│ Ctrl+F   │ Find signal by name                                   │
│ Ctrl+Sh+F│ Find value at time                                    │
├──────────┼───────────────────────────────────────────────────────┤
│ SESSION / FILE                                                    │
├──────────┼───────────────────────────────────────────────────────┤
│ Ctrl+Sh+S│ Save signal session                                   │
│ Ctrl+Sh+O│ Open/restore session                                  │
│ Ctrl+P   │ Print waveform                                        │
├──────────┼───────────────────────────────────────────────────────┤
│ VIEWS                                                             │
├──────────┼───────────────────────────────────────────────────────┤
│ Ctrl+1   │ Source view                                           │
│ Ctrl+2   │ nWave view                                            │
│ Ctrl+3   │ nSchema view                                          │
└──────────┴───────────────────────────────────────────────────────┘
```

## 5.3. Verilator Lint Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────┐
│ BASIC                                                              │
├────────────────────────────────────────────────────────────────────┤
│ verilator --lint-only ...              Lint-only mode              │
│ verilator --lint-only -Wall ...        All warnings                │
│ verilator --lint-only -Wno-fatal ...   Don't abort on warning      │
│ verilator --lint-only -Wno-WIDTH ...   Suppress WIDTH warning      │
├────────────────────────────────────────────────────────────────────┤
│ COMMON WARNINGS                                                    │
├────────────────────────────────────────────────────────────────────┤
│ UNUSED          Unused variable/signal                             │
│ UNDRIVEN        Undriven signal                                    │
│ WIDTH           Bit-width mismatch                                 │
│ PINMISSING      Unconnected module port                            │
│ CASEINCOMPLETE  Missing default in case                            │
│ LATCH           Latch inferred (missing else)                      │
│ MULTIDRIVEN     Multiple drivers                                   │
│ BLKSEQ          Blocking assign in sequential block                │
├────────────────────────────────────────────────────────────────────┤
│ SUPPRESS IN CODE                                                   │
├────────────────────────────────────────────────────────────────────┤
│ /* verilator lint_off UNUSED */        Suppress for next line      │
│ /* verilator lint_on UNUSED */         Re-enable                   │
│ /* verilator public */                 Make signal accessible      │
└────────────────────────────────────────────────────────────────────┘
```

---

# ════════════════════════════════════════════════════════════════
# PHẦN 6: ÁP DỤNG CHO PHASE_10
# ════════════════════════════════════════════════════════════════

## 6.1. Quy trình cho mỗi module trong PHASE_10

```
Ví dụ: Verify dsp_pair_int8

# 1. LINT check
verilator --lint-only -Wall \
    -I./stage_2/packages \
    ./stage_2/packages/accel_pkg.sv \
    ./stage_2/01_dsp_pair/rtl/dsp_pair_int8.sv

# 2. VCS compile
vcs -sverilog -full64 -debug_access+all \
    -timescale=1ns/1ps \
    ./stage_2/packages/accel_pkg.sv \
    ./stage_2/01_dsp_pair/rtl/dsp_pair_int8.sv \
    ./stage_2/01_dsp_pair/tb/tb_dsp_pair_int8.sv \
    -top tb_dsp_pair_int8 \
    -o simv_dsp -l compile_dsp.log

# 3. VCS run + dump FSDB
./simv_dsp -l sim_dsp.log +fsdbfile+dsp.fsdb +fsdb+all

# 4. Check result
grep -E "PASS|FAIL|ERROR" sim_dsp.log

# 5. Debug nếu FAIL
verdi -ssf dsp.fsdb \
      -sverilog \
      ./stage_2/packages/accel_pkg.sv \
      ./stage_2/01_dsp_pair/rtl/dsp_pair_int8.sv \
      ./stage_2/01_dsp_pair/tb/tb_dsp_pair_int8.sv &

# Trong Verdi:
#   Ctrl+W → add signals: x_a, x_b, w, psum_a, psum_b
#   R → signed decimal
#   Navigate to FAIL time → compare expected vs actual
```

## 6.2. Master Script cho toàn bộ PHASE_10

```bash
#!/bin/bash
# master_verify.sh — LINT + VCS + Verdi cho tất cả modules
# Usage: ./master_verify.sh [module_name]
# Example: ./master_verify.sh dsp_pair_int8

PROJ="$HOME/HMH_KLTN/PHASE_10"
MODULE=$1

echo "╔══════════════════════════════════════════╗"
echo "║  PHASE_10 Verification: $MODULE          ║"
echo "╚══════════════════════════════════════════╝"

# Step 1: LINT
echo "[1/3] LINT check..."
verilator --lint-only -Wall -Wno-fatal \
    -I${PROJ}/stage_2/packages \
    -f ${PROJ}/filelist_${MODULE}.f \
    2>&1 | tee lint_${MODULE}.log

LINT_ERRORS=$(grep -c "Error" lint_${MODULE}.log)
if [ "$LINT_ERRORS" -gt 0 ]; then
    echo "LINT FAILED: $LINT_ERRORS errors. Fix before proceeding."
    exit 1
fi
echo "LINT: PASS"

# Step 2: VCS Compile + Run
echo "[2/3] VCS compile + simulate..."
vcs -sverilog -full64 -debug_access+all \
    -timescale=1ns/1ps \
    -f ${PROJ}/filelist_${MODULE}.f \
    -top tb_${MODULE} \
    -o simv_${MODULE} \
    -l compile_${MODULE}.log \
    2>&1 | tail -5

./simv_${MODULE} -l sim_${MODULE}.log \
    +fsdbfile+${MODULE}.fsdb +fsdb+all

# Step 3: Check results
echo "[3/3] Results:"
grep -E "PASS|FAIL|ERROR|★" sim_${MODULE}.log | tail -20

FAIL_COUNT=$(grep -c "FAIL" sim_${MODULE}.log)
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "═══ FAILURES DETECTED ═══"
    echo "Debug: verdi -ssf ${MODULE}.fsdb -f filelist_${MODULE}.f &"
else
    echo ""
    echo "★ ALL TESTS PASSED ★"
fi
```
