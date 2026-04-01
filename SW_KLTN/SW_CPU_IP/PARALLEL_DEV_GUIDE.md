# Hướng Dẫn Phát Triển Song Song: PC (GĐ1+GĐ3) & FPGA IP (GĐ2)
## Virtex-7 + PCIe — Từ Stub IP Đến Real Accelerator

> **Bối cảnh**: Board Virtex-7 đã gắn vào PC qua PCIe.
> IP GĐ2 chưa xong → tạo Stub IP để verify đường truyền trước.
> Phát triển GĐ1+GĐ3 trên PC song song với GĐ2 trên FPGA.

---

# MỤC LỤC

```
PHẦN A: TỔNG QUAN CHIẾN LƯỢC PHÁT TRIỂN SONG SONG
PHẦN B: TOOLS CẦN THIẾT & SETUP
PHẦN C: FPGA SIDE — VIVADO BLOCK DESIGN + STUB IP
PHẦN D: PC SIDE — XDMA DRIVER + APPLICATION
PHẦN E: STEP-BY-STEP TEST FLOW
PHẦN F: TÍCH HỢP REAL IP (thay thế Stub)
PHẦN G: GĐ1 + GĐ3 CODE HOÀN CHỈNH TRÊN PC
```

---

# PHẦN A: TỔNG QUAN CHIẾN LƯỢC

## A.1. Ý tưởng chính

```
╔═══════════════════════════════════════════════════════════════════════╗
║  PHÁT TRIỂN SONG SONG — 2 Team / 2 Track                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  TRACK 1: PC Software (GĐ1 + GĐ3)     TRACK 2: FPGA IP (GĐ2)      ║
║  ─────────────────────────────────      ────────────────────────      ║
║  Tools: Visual Studio / CMake           Tools: Vivado                 ║
║  Lang:  C++ (OpenCV, XDMA driver)      Lang:  SystemVerilog          ║
║                                                                       ║
║  Sprint 1: Setup PCIe + DDR3            Sprint 1: Design RTL         ║
║  Sprint 2: Write/Read DDR3 test         Sprint 2: Simulate modules   ║
║  Sprint 3: GĐ1 preprocessing           Sprint 3: Integrate IP       ║
║  Sprint 4: GĐ3 postprocessing          Sprint 4: On-chip test       ║
║  Sprint 5: Full pipeline test           Sprint 5: Replace Stub IP    ║
║                                                                       ║
║  ★ STUB IP: IP giả đơn giản trên FPGA, đóng vai accel_top          ║
║    → Nhận data từ DDR3, xử lý đơn giản, ghi lại DDR3               ║
║    → Verify: PC ghi → FPGA đọc → FPGA xử lý → FPGA ghi → PC đọc   ║
║    → Khi real IP xong: thay Stub bằng accel_top → done!             ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

## A.2. Các milestone cần đạt

```
Milestone 0: [FPGA] Bitstream chạy được — PCIe link up, DDR3 calibrated
             [PC]   XDMA driver installed, device nhận diện

Milestone 1: [PC→DDR3] PC ghi 1 MB data vào DDR3, đọc lại đúng 100%
             → Chứng minh: PCIe + XDMA + DDR3 path OK

Milestone 2: [PC→DDR3→StubIP→DDR3→PC] PC ghi input, StubIP xử lý,
             PC đọc output → data khớp expected
             → Chứng minh: Full round-trip data path OK

Milestone 3: [GĐ1] PC preprocess ảnh → ghi X_int8 vào DDR3

Milestone 4: [GĐ3] PC đọc P3/P4/P5 từ DDR3 → postprocess → bbox

Milestone 5: [GĐ1+StubIP+GĐ3] Full pipeline: ảnh → preprocess →
             DDR3 → Stub → DDR3 → postprocess → bbox (dummy)

Milestone 6: [REPLACE] Thay Stub bằng real accel_top → real inference
```

## A.3. Kiến trúc hệ thống tổng thể

```
    ┌─────────────────────────────────────────────────┐
    │                    HOST PC                       │
    │                                                  │
    │  ┌──────────────────────────────────────────┐   │
    │  │  PC Application (C++, Visual Studio)      │   │
    │  │                                           │   │
    │  │  ┌──────────┐  ┌──────────┐  ┌────────┐  │   │
    │  │  │  GĐ1     │  │  Control │  │  GĐ3   │  │   │
    │  │  │ Preproc  │  │  (start/ │  │ Postpr │  │   │
    │  │  │ OpenCV   │  │  poll/   │  │ detect │  │   │
    │  │  │ Quantize │  │  IRQ)    │  │ bbox   │  │   │
    │  │  └────┬─────┘  └────┬─────┘  └────┬───┘  │   │
    │  │       │             │             │       │   │
    │  │  ┌────▼─────────────▼─────────────▼───┐   │   │
    │  │  │         XDMA Driver API            │   │   │
    │  │  │  xdma_write() / xdma_read()        │   │   │
    │  │  │  xdma_user_write() (AXI-Lite CSR)  │   │   │
    │  │  └────────────────┬───────────────────┘   │   │
    │  └───────────────────┼───────────────────────┘   │
    │                      │                           │
    └──────────────────────┼───────────────────────────┘
                           │  PCIe x4/x8
    ┌──────────────────────┼───────────────────────────┐
    │                      │     Virtex-7 FPGA Board   │
    │  ┌───────────────────▼───────────────────────┐   │
    │  │           XDMA IP (PCIe Endpoint)          │   │
    │  │     Xilinx DMA/Bridge Subsystem for PCIe   │   │
    │  │                                             │   │
    │  │  ┌─────────┐        ┌──────────────────┐   │   │
    │  │  │ H2C DMA │        │ AXI-Lite Bypass  │   │   │
    │  │  │(Host→   │        │ (CSR access)     │   │   │
    │  │  │ Card)   │        └────────┬─────────┘   │   │
    │  │  │         │                 │              │   │
    │  │  │ C2H DMA │                 │              │   │
    │  │  │(Card→   │                 │              │   │
    │  │  │ Host)   │                 │              │   │
    │  │  └────┬────┘                 │              │   │
    │  └───────┼──────────────────────┼──────────────┘   │
    │          │ AXI4 (256b)          │ AXI-Lite (32b)   │
    │  ┌───────▼──────────────────────▼──────────────┐   │
    │  │           AXI Interconnect                   │   │
    │  └───┬─────────────────────┬───────────────────┘   │
    │      │                     │                       │
    │  ┌───▼──────────┐   ┌─────▼─────────────────┐    │
    │  │  MIG DDR3    │   │  Stub IP / accel_top   │    │
    │  │  Controller  │   │  (AXI-Lite Slave: CSR) │    │
    │  │              │   │  (AXI4 Master: DMA)    │    │
    │  └──────────────┘   └────────────────────────┘    │
    │                                                    │
    └────────────────────────────────────────────────────┘
```

---

# PHẦN B: TOOLS CẦN THIẾT & SETUP

## B.1. Danh sách tools

```
╔══════════════════════════════════════════════════════════════════════╗
║  FPGA SIDE (Vivado)                                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  ① Xilinx Vivado Design Suite 2019.2+ (hoặc mới hơn)              ║
║     → Tạo block design, synthesize, implement, generate bitstream  ║
║     → IP Integrator cho XDMA + MIG + AXI Interconnect             ║
║                                                                      ║
║  ② Xilinx Vivado HLS (optional, cho Stub IP nếu viết C++)          ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PC SIDE (Software)                                                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  ③ Visual Studio 2019/2022 (Community Edition — FREE)              ║
║     → C/C++ project cho PC application                              ║
║     → Compile driver test program + GĐ1 + GĐ3                     ║
║                                                                      ║
║  ④ Xilinx XDMA Driver (Windows hoặc Linux)                         ║
║     Windows: Xilinx cung cấp signed driver (.sys + .inf)           ║
║     Linux:   dma_ip_drivers (open source trên GitHub)               ║
║     → Cho phép PC đọc/ghi DDR3 trên FPGA board qua PCIe           ║
║                                                                      ║
║  ⑤ OpenCV 4.x (for preprocessing / postprocessing)                 ║
║     → Resize, normalize, draw bounding boxes                       ║
║     → vcpkg install opencv4:x64-windows                            ║
║                                                                      ║
║  ⑥ CMake (optional, thay cho VS project)                           ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  COMMON                                                              ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⑦ Python 3.x + NumPy + PyTorch (Golden reference — đã có từ Ph1) ║
║     → Tạo test vectors (X_int8, expected P3/P4/P5)                ║
║                                                                      ║
║  ⑧ Git (version control)                                           ║
╚══════════════════════════════════════════════════════════════════════╝
```

## B.2. Cài đặt XDMA Driver (Windows)

```
STEP 1: Download Xilinx XDMA driver
  → Từ Xilinx GitHub: https://github.com/Xilinx/dma_ip_drivers
  → Hoặc từ Vivado installation: 
    C:\Xilinx\Vivado\2023.1\data\xicom\drivers\pcie\xdma\

STEP 2: Build driver (Windows)
  → Mở "x64 Native Tools Command Prompt for VS 2022"
  → cd xdma_driver\windows\
  → Chạy build script hoặc mở .sln trong Visual Studio

STEP 3: Install driver
  → Click phải vào xdma.inf → Install
  → Hoặc: Device Manager → Update Driver → Browse → chọn folder driver
  → Cần tắt Secure Boot trong BIOS nếu driver unsigned

STEP 4: Verify
  → Device Manager → hiện "Xilinx DMA" device
  → Thấy các device files:
     \\.\xdma0_h2c_0   (Host to Card DMA channel 0)
     \\.\xdma0_c2h_0   (Card to Host DMA channel 0)
     \\.\xdma0_user     (AXI-Lite bypass — for CSR access)
     \\.\xdma0_control  (XDMA internal registers)

STEP 5: Test với xdma_test.exe (Xilinx cung cấp)
  → xdma_rw.exe --write --device xdma0_h2c_0 --addr 0x0 --data 0xDEADBEEF
  → xdma_rw.exe --read  --device xdma0_c2h_0 --addr 0x0
  → Nếu đọc ra 0xDEADBEEF → PCIe + DDR3 path OK!
```

## B.3. Cài đặt XDMA Driver (Linux — recommended)

```
STEP 1: Clone driver source
  git clone https://github.com/Xilinx/dma_ip_drivers.git
  cd dma_ip_drivers/XDMA/linux-kernel/xdma/

STEP 2: Build kernel module
  make clean && make
  sudo make install    (hoặc sudo insmod xdma.ko)

STEP 3: Load driver
  sudo modprobe xdma
  # Hoặc: sudo insmod xdma.ko

STEP 4: Verify device files
  ls /dev/xdma0_*
  # Phải thấy:
  #   /dev/xdma0_h2c_0    (DMA write: Host → FPGA DDR3)
  #   /dev/xdma0_c2h_0    (DMA read:  FPGA DDR3 → Host)
  #   /dev/xdma0_user      (AXI-Lite bypass for CSR)
  #   /dev/xdma0_control   (XDMA config registers)

STEP 5: Quick test
  # Write 4 bytes to DDR3 offset 0:
  echo -n -e '\xef\xbe\xad\xde' | sudo dd of=/dev/xdma0_h2c_0 bs=4 count=1
  
  # Read 4 bytes from DDR3 offset 0:
  sudo dd if=/dev/xdma0_c2h_0 bs=4 count=1 | xxd
  # Expect: 0000000: efbe adde

STEP 6: Test program
  cd dma_ip_drivers/XDMA/linux-kernel/tests/
  ./run_tests.sh
```

## B.4. Setup Visual Studio Project (Windows)

```
STEP 1: Tạo project mới
  Visual Studio → File → New Project → "Console App (C++)"
  Name: "yolov10n_host"
  Location: SW_CPU_IP\host_app\

STEP 2: Cấu hình project
  → Project Properties → C/C++ → General → Additional Include Directories:
    - path\to\opencv\include
    - path\to\xdma_driver\include
  → Linker → Additional Library Directories:
    - path\to\opencv\lib
  → Linker → Input → Additional Dependencies:
    - opencv_world4xx.lib

STEP 3: Cấu trúc source files
  host_app/
  ├── xdma_api.h          ← XDMA driver wrapper
  ├── xdma_api.cpp        ← XDMA read/write functions
  ├── memory_map.h        ← DDR3 address definitions
  ├── preprocess.h/.cpp   ← GĐ1: resize + quantize
  ├── postprocess.h/.cpp  ← GĐ3: dequant + detect + bbox
  ├── stub_test.cpp       ← Milestone 1-2: DDR3 read/write test
  └── main.cpp            ← Full inference pipeline

STEP 4: Build Configuration
  → x64 Release mode (for performance)
  → Debug mode ban đầu (for testing)
```

---

# PHẦN C: FPGA SIDE — VIVADO BLOCK DESIGN + STUB IP

## C.1. Tạo Vivado Block Design

```
STEP 1: Tạo project Vivado
  → Vivado → Create Project → RTL Project
  → Part: xc7vx690tffg1761-2 (hoặc board part nếu có)

STEP 2: Create Block Design
  → IP Integrator → Create Block Design → name: "system_top"

STEP 3: Add XDMA IP
  → Add IP → search "DMA/Bridge Subsystem for PCI Express"
  → Double-click để configure:
  
  ┌─────────────────────────────────────────────────────┐
  │  XDMA Configuration                                 │
  │                                                     │
  │  Basic:                                             │
  │    Mode:           Advanced                         │
  │    Device/Port Type: PCI Express Endpoint device    │
  │    PCIe Block:     X0Y0 (check board schematic)     │
  │    Lane Width:     X4 hoặc X8 (theo board)          │
  │    Link Speed:     5.0 GT/s (Gen2) hoặc 8.0 (Gen3) │
  │                                                     │
  │  PCIE:BARs:                                         │
  │    BAR0: AXI-Lite (CSR access) — 64KB              │
  │    BAR1: AXI4     (DDR3 DMA)   — 1GB              │
  │                                                     │
  │  DMA:                                               │
  │    Number of DMA Read  (H2C): 1                    │
  │    Number of DMA Write (C2H): 1                    │
  │    AXI Data Width: 256 bits                        │
  │    AXI Address Width: 40 bits                      │
  │    Enable AXI-Lite Master (User): Yes ★            │
  │                                                     │
  │  Interrupts:                                        │
  │    Number of User Interrupts: 1 (for IP done)      │
  └─────────────────────────────────────────────────────┘

STEP 4: Add MIG DDR3
  → Add IP → search "Memory Interface Generator (MIG 7 Series)"
  → Configure theo board schematic:
    - DDR3 SDRAM
    - Data Width: 64 bits
    - Clock: 200 MHz input, 800 MHz DDR
    - AXI interface: 256-bit, 200 MHz

STEP 5: Add AXI Interconnect
  → Add IP → "AXI Interconnect"
  → 2 Masters (XDMA H2C + Stub IP AXI Master)
  → 2 Slaves  (DDR3 MIG + Stub IP AXI-Lite)

STEP 6: Add Stub IP (xem mục C.2)

STEP 7: Connect everything (xem diagram C.3)
```

## C.2. Stub IP — IP Giả Đơn Giản

Stub IP đóng vai trò accel_top nhưng chỉ làm việc đơn giản:
- Nhận lệnh start qua CSR
- Đọc N bytes từ DDR3 address A
- Copy/Transform data
- Ghi kết quả vào DDR3 address B
- Báo done qua CSR + IRQ

```
Stub IP có 3 mức phức tạp tăng dần:

Level 1 — LOOPBACK: Copy nguyên input → output (memcpy trên FPGA)
  → Test: PC ghi X → FPGA copy X → PC đọc X → so sánh → phải giống

Level 2 — TRANSFORM: Đọc input, thêm offset +1 cho mỗi byte, ghi output
  → Test: PC ghi X → FPGA ghi (X+1) → PC đọc (X+1) → verify

Level 3 — SIZE MATCH: Đọc 1.23MB input, ghi 716KB output (đúng kích thước P3+P4+P5)
  → Test: Verify đúng kích thước data round-trip
  → Đây là "form factor" test — khi thay bằng real IP, sizes đã đúng
```

### C.2.1. Stub IP Level 1 — Loopback (SystemVerilog)

```systemverilog
// stub_accel_top.sv — Loopback IP for testing PCIe + DDR3 path
// Mimics accel_top interface: AXI-Lite slave (CSR) + AXI4 master (DMA)
module stub_accel_top (
  input  logic        clk,
  input  logic        rst_n,

  // AXI-Lite Slave (CSR from PC via XDMA bypass)
  input  logic [11:0] s_axil_awaddr,
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [11:0] s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,

  // AXI4 Master (DMA to DDR3)
  output logic [39:0]  m_axi_araddr,
  output logic [7:0]   m_axi_arlen,
  output logic [2:0]   m_axi_arsize,
  output logic [1:0]   m_axi_arburst,
  output logic         m_axi_arvalid,
  input  logic         m_axi_arready,
  input  logic [255:0] m_axi_rdata,
  input  logic [1:0]   m_axi_rresp,
  input  logic         m_axi_rlast,
  input  logic         m_axi_rvalid,
  output logic         m_axi_rready,

  output logic [39:0]  m_axi_awaddr,
  output logic [7:0]   m_axi_awlen,
  output logic [2:0]   m_axi_awsize,
  output logic [1:0]   m_axi_awburst,
  output logic         m_axi_awvalid,
  input  logic         m_axi_awready,
  output logic [255:0] m_axi_wdata,
  output logic [31:0]  m_axi_wstrb,
  output logic         m_axi_wlast,
  output logic         m_axi_wvalid,
  input  logic         m_axi_wready,
  input  logic [1:0]   m_axi_bresp,
  input  logic         m_axi_bvalid,
  output logic         m_axi_bready,

  // Interrupt output
  output logic         irq
);

  // ═══════════ CSR Registers ═══════════
  logic [31:0] csr_ctrl;        // 0x000: [0]start [1]reset [2]irq_clr
  logic [31:0] csr_status;      // 0x004: [0]busy [1]done [2]irq
  logic [31:0] csr_src_addr_lo; // 0x010: source DDR3 address (low)
  logic [31:0] csr_src_addr_hi; // 0x014: source DDR3 address (high)
  logic [31:0] csr_dst_addr_lo; // 0x018: destination DDR3 address (low)
  logic [31:0] csr_dst_addr_hi; // 0x01C: destination DDR3 address (high)
  logic [31:0] csr_xfer_bytes;  // 0x020: number of bytes to transfer
  logic [31:0] csr_version;     // 0x008: version (read-only)

  assign csr_version = 32'h594F_FFFF; // "YO" + stub marker

  // ═══════════ AXI-Lite Slave (CSR) ═══════════
  // Write channel
  logic aw_handshake, w_handshake;
  logic [11:0] wr_addr_reg;

  assign s_axil_awready = !aw_handshake || (aw_handshake && w_handshake);
  assign s_axil_wready  = !w_handshake  || (aw_handshake && w_handshake);
  assign s_axil_bresp   = 2'b00;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_handshake <= 0;
      w_handshake  <= 0;
      wr_addr_reg  <= 0;
      s_axil_bvalid <= 0;
      csr_ctrl <= 0;
      csr_src_addr_lo <= 32'h0030_0000; // default input addr
      csr_src_addr_hi <= 0;
      csr_dst_addr_lo <= 32'h0080_0000; // default output addr
      csr_dst_addr_hi <= 0;
      csr_xfer_bytes  <= 32'd1228800;   // default: 1.23MB
    end else begin
      // Capture AW
      if (s_axil_awvalid && s_axil_awready) begin
        aw_handshake <= 1;
        wr_addr_reg  <= s_axil_awaddr;
      end
      // Capture W
      if (s_axil_wvalid && s_axil_wready)
        w_handshake <= 1;

      // Both ready → perform write
      if (aw_handshake && w_handshake) begin
        aw_handshake  <= 0;
        w_handshake   <= 0;
        s_axil_bvalid <= 1;

        case (wr_addr_reg)
          12'h000: csr_ctrl        <= s_axil_wdata;
          12'h010: csr_src_addr_lo <= s_axil_wdata;
          12'h014: csr_src_addr_hi <= s_axil_wdata;
          12'h018: csr_dst_addr_lo <= s_axil_wdata;
          12'h01C: csr_dst_addr_hi <= s_axil_wdata;
          12'h020: csr_xfer_bytes  <= s_axil_wdata;
          default: ;
        endcase
      end

      if (s_axil_bvalid && s_axil_bready)
        s_axil_bvalid <= 0;

      // Auto-clear start bit
      if (csr_ctrl[0]) csr_ctrl[0] <= 0;
      // IRQ clear
      if (csr_ctrl[2]) begin
        csr_status[2] <= 0;
        csr_ctrl[2]   <= 0;
      end
    end
  end

  // Read channel
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_axil_arready <= 1;
      s_axil_rvalid  <= 0;
      s_axil_rdata   <= 0;
    end else begin
      if (s_axil_arvalid && s_axil_arready) begin
        s_axil_arready <= 0;
        s_axil_rvalid  <= 1;
        case (s_axil_araddr)
          12'h000: s_axil_rdata <= csr_ctrl;
          12'h004: s_axil_rdata <= csr_status;
          12'h008: s_axil_rdata <= csr_version;
          12'h010: s_axil_rdata <= csr_src_addr_lo;
          12'h014: s_axil_rdata <= csr_src_addr_hi;
          12'h018: s_axil_rdata <= csr_dst_addr_lo;
          12'h01C: s_axil_rdata <= csr_dst_addr_hi;
          12'h020: s_axil_rdata <= csr_xfer_bytes;
          default: s_axil_rdata <= 32'hDEAD_BEEF;
        endcase
      end
      if (s_axil_rvalid && s_axil_rready) begin
        s_axil_rvalid  <= 0;
        s_axil_arready <= 1;
      end
    end
  end

  // ═══════════ DMA Copy Engine (Loopback) ═══════════
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_READ_REQ,
    ST_READ_DATA,
    ST_WRITE_REQ,
    ST_WRITE_DATA,
    ST_WRITE_RESP,
    ST_NEXT_BURST,
    ST_DONE
  } state_e;

  state_e state;
  logic [39:0] src_addr, dst_addr;
  logic [31:0] bytes_remaining;
  logic [255:0] data_buf [16]; // 16-beat burst buffer
  logic [3:0]   beat_cnt;
  logic [7:0]   burst_len;

  wire start_pulse = csr_ctrl[0];

  // AXI4 defaults
  assign m_axi_arsize  = 3'b101; // 32 bytes
  assign m_axi_arburst = 2'b01;  // INCR
  assign m_axi_awsize  = 3'b101;
  assign m_axi_awburst = 2'b01;
  assign m_axi_wstrb   = 32'hFFFF_FFFF;
  assign m_axi_bready  = 1'b1;

  assign irq = csr_status[2];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state           <= ST_IDLE;
      csr_status      <= 0;
      m_axi_arvalid   <= 0;
      m_axi_rready    <= 0;
      m_axi_awvalid   <= 0;
      m_axi_wvalid    <= 0;
      m_axi_wlast     <= 0;
      bytes_remaining <= 0;
      beat_cnt        <= 0;
    end else begin
      case (state)

        ST_IDLE: begin
          if (start_pulse) begin
            src_addr        <= {csr_src_addr_hi[7:0], csr_src_addr_lo};
            dst_addr        <= {csr_dst_addr_hi[7:0], csr_dst_addr_lo};
            bytes_remaining <= csr_xfer_bytes;
            csr_status      <= 32'h0000_0001; // busy
            state           <= ST_READ_REQ;
          end
        end

        ST_READ_REQ: begin
          if (bytes_remaining == 0) begin
            state <= ST_DONE;
          end else begin
            // Burst length: min(15, remaining_beats-1)
            automatic logic [31:0] remain_beats = bytes_remaining / 32;
            if (remain_beats > 16) burst_len = 8'd15;
            else burst_len = remain_beats[7:0] - 1;

            m_axi_araddr  <= src_addr;
            m_axi_arlen   <= burst_len;
            m_axi_arvalid <= 1;
            beat_cnt      <= 0;
            state         <= ST_READ_DATA;
          end
        end

        ST_READ_DATA: begin
          if (m_axi_arvalid && m_axi_arready)
            m_axi_arvalid <= 0;

          m_axi_rready <= 1;
          if (m_axi_rvalid && m_axi_rready) begin
            data_buf[beat_cnt] <= m_axi_rdata;
            beat_cnt <= beat_cnt + 1;
            if (m_axi_rlast) begin
              m_axi_rready <= 0;
              beat_cnt     <= 0;
              state        <= ST_WRITE_REQ;
            end
          end
        end

        ST_WRITE_REQ: begin
          m_axi_awaddr  <= dst_addr;
          m_axi_awlen   <= burst_len;
          m_axi_awvalid <= 1;
          beat_cnt      <= 0;
          state         <= ST_WRITE_DATA;
        end

        ST_WRITE_DATA: begin
          if (m_axi_awvalid && m_axi_awready)
            m_axi_awvalid <= 0;

          m_axi_wdata  <= data_buf[beat_cnt];
          m_axi_wvalid <= 1;
          m_axi_wlast  <= (beat_cnt == burst_len);

          if (m_axi_wvalid && m_axi_wready) begin
            beat_cnt <= beat_cnt + 1;
            if (beat_cnt == burst_len) begin
              m_axi_wvalid <= 0;
              m_axi_wlast  <= 0;
              state        <= ST_WRITE_RESP;
            end
          end
        end

        ST_WRITE_RESP: begin
          if (m_axi_bvalid) begin
            // Update addresses and remaining
            automatic logic [31:0] burst_bytes = (32'd1 + burst_len) * 32;
            src_addr        <= src_addr + burst_bytes;
            dst_addr        <= dst_addr + burst_bytes;
            bytes_remaining <= bytes_remaining - burst_bytes;
            state           <= ST_READ_REQ; // next burst
          end
        end

        ST_DONE: begin
          csr_status <= 32'h0000_0006; // done + irq
          state      <= ST_IDLE;
        end

      endcase
    end
  end

endmodule
```

## C.3. Block Design Connection Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Vivado Block Design                          │
│                                                                     │
│  ┌──────────────────────────────────┐                              │
│  │         XDMA IP                   │                              │
│  │  (DMA/Bridge for PCIe)           │                              │
│  │                                   │                              │
│  │  pcie_7x_mgt ←─── PCIe pins     │                              │
│  │  sys_clk     ←─── 100/250 MHz   │                              │
│  │  sys_rst_n   ←─── PERST#        │                              │
│  │                                   │                              │
│  │  M_AXI  (256b) ──→ S00 of Interconn (DMA data to DDR3)        │
│  │  M_AXI_LITE ──→ S01 of Interconn   (AXI-Lite bypass to Stub)  │
│  │  axi_aclk    ──→ shared clock     (user clock output)          │
│  │  axi_aresetn ──→ shared reset                                  │
│  │  usr_irq_req ←── irq from Stub IP                              │
│  └──────────────────────────────────┘                              │
│                                                                     │
│  ┌──────────────────────────────────┐                              │
│  │      AXI Interconnect            │                              │
│  │                                   │                              │
│  │  S00 ← XDMA M_AXI      (256b)  │                              │
│  │  S01 ← Stub  m_axi      (256b)  │  ★ Stub reads/writes DDR3  │
│  │                                   │                              │
│  │  M00 → MIG DDR3 S_AXI   (256b)  │  address range 0x0-0x3FFF.. │
│  │  M01 → Stub s_axil      (32b)   │  address range 0x4400_0000  │
│  └──────────────────────────────────┘                              │
│                                                                     │
│  ┌──────────────────────────────────┐                              │
│  │      MIG DDR3 Controller         │                              │
│  │  S_AXI ← M00 of Interconnect    │                              │
│  │  sys_clk ← 200 MHz ref          │                              │
│  │  DDR3 pins → board DDR3 SDRAM    │                              │
│  └──────────────────────────────────┘                              │
│                                                                     │
│  ┌──────────────────────────────────┐                              │
│  │      Stub IP (stub_accel_top)    │                              │
│  │  s_axil ← M01 of Interconnect   │  PC writes CSR here          │
│  │  m_axi  → S01 of Interconnect   │  Stub reads/writes DDR3      │
│  │  irq    → XDMA usr_irq_req[0]   │                              │
│  └──────────────────────────────────┘                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Address Map:
  0x0000_0000 — 0x3FFF_FFFF : DDR3 (via M00) — 1 GB
  0x4400_0000 — 0x4400_0FFF : Stub IP CSR (via M01) — 4 KB
```

## C.4. Generate Bitstream & Program

```
STEP 1: Validate Block Design
  → Tools → Validate Design (F5)
  → Fix any connection errors

STEP 2: Create HDL Wrapper
  → Right-click block design → Create HDL Wrapper
  → Let Vivado manage

STEP 3: Add Constraints (.xdc)
  → PCIe pins (from board schematic)
  → DDR3 pins (from board schematic)  
  → System clock, reset
  → Thường board có .xdc template file

STEP 4: Run Synthesis → Implementation → Generate Bitstream
  → Tốn 30-60 phút cho Virtex-7

STEP 5: Program FPGA
  → Hardware Manager → Open Target → Program Device
  → Load .bit file

STEP 6: Verify PCIe link
  → Trên PC: lspci | grep Xilinx (Linux)
  → Hoặc Device Manager (Windows)
  → Phải thấy Xilinx device
```

---

# PHẦN D: PC SIDE — XDMA DRIVER API + APPLICATION

## D.1. XDMA API Wrapper (C/C++)

File này wrap các thao tác đọc/ghi qua XDMA driver thành các function đơn giản.

```c
/* ═══════════════════════════════════════
 *  xdma_api.h — XDMA Driver Wrapper
 * ═══════════════════════════════════════ */
#ifndef XDMA_API_H
#define XDMA_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Khởi tạo — mở device files */
int xdma_init(void);

/* Giải phóng — đóng device files */
void xdma_cleanup(void);

/* ── DMA Bulk Transfer (DDR3 read/write) ── */

/* Ghi data từ PC buffer vào DDR3 trên FPGA
 * fpga_addr: DDR3 physical address (e.g. 0x00300000)
 * data:      pointer to PC buffer
 * size:      number of bytes to write
 * Returns:   0 on success, -1 on error */
int xdma_write_ddr3(uint64_t fpga_addr, const void *data, size_t size);

/* Đọc data từ DDR3 trên FPGA về PC buffer
 * fpga_addr: DDR3 physical address
 * data:      pointer to PC buffer (pre-allocated)
 * size:      number of bytes to read
 * Returns:   0 on success, -1 on error */
int xdma_read_ddr3(uint64_t fpga_addr, void *data, size_t size);

/* ── AXI-Lite Register Access (CSR read/write) ── */

/* Ghi 1 register 32-bit vào IP CSR
 * reg_offset: offset within CSR space (e.g. 0x000 for CTRL)
 * value:      32-bit value to write */
int xdma_csr_write(uint32_t reg_offset, uint32_t value);

/* Đọc 1 register 32-bit từ IP CSR
 * reg_offset: offset within CSR space
 * Returns:    32-bit register value */
uint32_t xdma_csr_read(uint32_t reg_offset);

#ifdef __cplusplus
}
#endif

#endif
```

## D.2. XDMA API Implementation

```c
/* ═══════════════════════════════════════
 *  xdma_api.c — Implementation
 * ═══════════════════════════════════════ */

/* ★★★ WINDOWS VERSION ★★★ */
#ifdef _WIN32

#include "xdma_api.h"
#include <windows.h>
#include <stdio.h>

static HANDLE h2c_handle = INVALID_HANDLE_VALUE;  /* Host→Card DMA */
static HANDLE c2h_handle = INVALID_HANDLE_VALUE;  /* Card→Host DMA */
static HANDLE usr_handle = INVALID_HANDLE_VALUE;  /* AXI-Lite bypass */

int xdma_init(void)
{
    h2c_handle = CreateFileA(
        "\\\\.\\xdma0_h2c_0",
        GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);

    c2h_handle = CreateFileA(
        "\\\\.\\xdma0_c2h_0",
        GENERIC_READ, 0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);

    usr_handle = CreateFileA(
        "\\\\.\\xdma0_user",
        GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, NULL);

    if (h2c_handle == INVALID_HANDLE_VALUE ||
        c2h_handle == INVALID_HANDLE_VALUE ||
        usr_handle == INVALID_HANDLE_VALUE) {
        printf("ERROR: Cannot open XDMA device files.\n");
        printf("  Check: 1) FPGA programmed? 2) Driver installed?\n");
        printf("  Check: 3) PCIe link up? 4) Device Manager?\n");
        xdma_cleanup();
        return -1;
    }

    printf("XDMA initialized successfully.\n");
    return 0;
}

void xdma_cleanup(void)
{
    if (h2c_handle != INVALID_HANDLE_VALUE) CloseHandle(h2c_handle);
    if (c2h_handle != INVALID_HANDLE_VALUE) CloseHandle(c2h_handle);
    if (usr_handle != INVALID_HANDLE_VALUE) CloseHandle(usr_handle);
    h2c_handle = c2h_handle = usr_handle = INVALID_HANDLE_VALUE;
}

int xdma_write_ddr3(uint64_t fpga_addr, const void *data, size_t size)
{
    LARGE_INTEGER offset;
    offset.QuadPart = fpga_addr;
    DWORD written = 0;

    SetFilePointerEx(h2c_handle, offset, NULL, FILE_BEGIN);
    BOOL ok = WriteFile(h2c_handle, data, (DWORD)size, &written, NULL);

    if (!ok || written != size) {
        printf("ERROR: DMA write failed at 0x%llX (wrote %d/%zu)\n",
               fpga_addr, written, size);
        return -1;
    }
    return 0;
}

int xdma_read_ddr3(uint64_t fpga_addr, void *data, size_t size)
{
    LARGE_INTEGER offset;
    offset.QuadPart = fpga_addr;
    DWORD read_bytes = 0;

    SetFilePointerEx(c2h_handle, offset, NULL, FILE_BEGIN);
    BOOL ok = ReadFile(c2h_handle, data, (DWORD)size, &read_bytes, NULL);

    if (!ok || read_bytes != size) {
        printf("ERROR: DMA read failed at 0x%llX (read %d/%zu)\n",
               fpga_addr, read_bytes, size);
        return -1;
    }
    return 0;
}

int xdma_csr_write(uint32_t reg_offset, uint32_t value)
{
    LARGE_INTEGER offset;
    offset.QuadPart = reg_offset;
    DWORD written = 0;

    SetFilePointerEx(usr_handle, offset, NULL, FILE_BEGIN);
    BOOL ok = WriteFile(usr_handle, &value, 4, &written, NULL);
    return (ok && written == 4) ? 0 : -1;
}

uint32_t xdma_csr_read(uint32_t reg_offset)
{
    LARGE_INTEGER offset;
    offset.QuadPart = reg_offset;
    uint32_t value = 0;
    DWORD read_bytes = 0;

    SetFilePointerEx(usr_handle, offset, NULL, FILE_BEGIN);
    ReadFile(usr_handle, &value, 4, &read_bytes, NULL);
    return value;
}

#endif /* _WIN32 */


/* ★★★ LINUX VERSION ★★★ */
#ifdef __linux__

#include "xdma_api.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>

static int h2c_fd = -1;
static int c2h_fd = -1;
static int usr_fd = -1;

int xdma_init(void)
{
    h2c_fd = open("/dev/xdma0_h2c_0", O_WRONLY);
    c2h_fd = open("/dev/xdma0_c2h_0", O_RDONLY);
    usr_fd = open("/dev/xdma0_user",   O_RDWR);

    if (h2c_fd < 0 || c2h_fd < 0 || usr_fd < 0) {
        perror("Cannot open XDMA device");
        xdma_cleanup();
        return -1;
    }

    printf("XDMA initialized successfully.\n");
    return 0;
}

void xdma_cleanup(void)
{
    if (h2c_fd >= 0) close(h2c_fd);
    if (c2h_fd >= 0) close(c2h_fd);
    if (usr_fd >= 0) close(usr_fd);
    h2c_fd = c2h_fd = usr_fd = -1;
}

int xdma_write_ddr3(uint64_t fpga_addr, const void *data, size_t size)
{
    if (lseek(h2c_fd, fpga_addr, SEEK_SET) < 0) return -1;
    ssize_t written = write(h2c_fd, data, size);
    return (written == (ssize_t)size) ? 0 : -1;
}

int xdma_read_ddr3(uint64_t fpga_addr, void *data, size_t size)
{
    if (lseek(c2h_fd, fpga_addr, SEEK_SET) < 0) return -1;
    ssize_t rd = read(c2h_fd, data, size);
    return (rd == (ssize_t)size) ? 0 : -1;
}

int xdma_csr_write(uint32_t reg_offset, uint32_t value)
{
    if (lseek(usr_fd, reg_offset, SEEK_SET) < 0) return -1;
    return (write(usr_fd, &value, 4) == 4) ? 0 : -1;
}

uint32_t xdma_csr_read(uint32_t reg_offset)
{
    uint32_t value = 0;
    if (lseek(usr_fd, reg_offset, SEEK_SET) < 0) return 0xDEAD;
    read(usr_fd, &value, 4);
    return value;
}

#endif /* __linux__ */
```

## D.3. Memory Map Header

```c
/* ═══════════════════════════════════════
 *  memory_map.h — DDR3 Address Definitions
 * ═══════════════════════════════════════ */
#ifndef MEMORY_MAP_H
#define MEMORY_MAP_H

#include <stdint.h>

/* DDR3 Memory Layout */
#define DDR3_DESC_BASE    0x00000000ULL  /* 1 MB  descriptors  */
#define DDR3_WEIGHT_BASE  0x00100000ULL  /* 2 MB  weights      */
#define DDR3_INPUT_BASE   0x00300000ULL  /* 1.23 MB input X_int8 */
#define DDR3_P3_BASE      0x00800000ULL  /* P3 [64,80,80]   = 409,600 B */
#define DDR3_P4_BASE      0x00A00000ULL  /* P4 [128,40,40]  = 204,800 B */
#define DDR3_P5_BASE      0x00B00000ULL  /* P5 [256,20,20]  = 102,400 B */
#define DDR3_SCRATCH_BASE 0x01000000ULL  /* scratch space for testing */

/* Sizes */
#define INPUT_SIZE   (3 * 640 * 640)    /* 1,228,800 bytes */
#define P3_SIZE      (64 * 80 * 80)     /*   409,600 bytes */
#define P4_SIZE      (128 * 40 * 40)    /*   204,800 bytes */
#define P5_SIZE      (256 * 20 * 20)    /*   102,400 bytes */
#define OUTPUT_TOTAL (P3_SIZE + P4_SIZE + P5_SIZE) /* 716,800 bytes */

/* CSR Offsets (within AXI-Lite bypass space) */
#define CSR_CTRL           0x000
#define CSR_STATUS         0x004
#define CSR_VERSION        0x008
#define CSR_SRC_ADDR_LO    0x010
#define CSR_SRC_ADDR_HI    0x014
#define CSR_DST_ADDR_LO    0x018
#define CSR_DST_ADDR_HI    0x01C
#define CSR_XFER_BYTES     0x020

#endif
```

---

# PHẦN E: STEP-BY-STEP TEST FLOW

## E.1. Milestone 0 — PCIe Link Up

```c
/* test_00_link.c — Verify PCIe connection */
#include "xdma_api.h"
#include "memory_map.h"
#include <stdio.h>

int main()
{
    printf("=== Milestone 0: PCIe Link Test ===\n");

    if (xdma_init() != 0) {
        printf("FAIL: Cannot open XDMA device.\n");
        printf("Checklist:\n");
        printf("  1. Is FPGA programmed with bitstream?\n");
        printf("  2. Is XDMA driver installed?\n");
        printf("  3. Does Device Manager show Xilinx device?\n");
        printf("  4. Try rebooting PC after programming FPGA.\n");
        return 1;
    }

    /* Read Stub IP version register */
    uint32_t version = xdma_csr_read(CSR_VERSION);
    printf("IP Version Register: 0x%08X\n", version);

    if ((version >> 16) == 0x594F) {
        printf("PASS: IP responds correctly (magic=YO)\n");
        if ((version & 0xFFFF) == 0xFFFF)
            printf("  → Stub IP detected (not real accelerator)\n");
        else
            printf("  → Real IP version: %d\n", version & 0xFFFF);
    } else {
        printf("FAIL: Unexpected version 0x%08X\n", version);
    }

    xdma_cleanup();
    return 0;
}
```

## E.2. Milestone 1 — DDR3 Write + Read Back

```c
/* test_01_ddr3.c — Verify DDR3 read/write via PCIe */
#include "xdma_api.h"
#include "memory_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main()
{
    printf("=== Milestone 1: DDR3 Read/Write Test ===\n\n");

    if (xdma_init() != 0) return 1;

    /* Test 1: Write 1 KB pattern, read back, compare */
    const size_t TEST_SIZE = 1024;
    uint8_t *wr_buf = (uint8_t *)malloc(TEST_SIZE);
    uint8_t *rd_buf = (uint8_t *)malloc(TEST_SIZE);

    /* Fill with known pattern */
    for (size_t i = 0; i < TEST_SIZE; i++)
        wr_buf[i] = (uint8_t)(i & 0xFF);

    printf("[1/3] Writing %zu bytes to DDR3 @ 0x%llX...\n",
           TEST_SIZE, DDR3_SCRATCH_BASE);
    xdma_write_ddr3(DDR3_SCRATCH_BASE, wr_buf, TEST_SIZE);

    printf("[2/3] Reading back %zu bytes...\n", TEST_SIZE);
    memset(rd_buf, 0, TEST_SIZE);
    xdma_read_ddr3(DDR3_SCRATCH_BASE, rd_buf, TEST_SIZE);

    printf("[3/3] Comparing...\n");
    int errors = 0;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (wr_buf[i] != rd_buf[i]) {
            if (errors < 10)
                printf("  MISMATCH at byte %zu: wrote=0x%02X read=0x%02X\n",
                       i, wr_buf[i], rd_buf[i]);
            errors++;
        }
    }

    if (errors == 0)
        printf("\nPASS: %zu bytes match perfectly!\n", TEST_SIZE);
    else
        printf("\nFAIL: %d mismatches out of %zu bytes\n",
               errors, TEST_SIZE);

    /* Test 2: Large transfer (1 MB) */
    printf("\n--- Large transfer test (1 MB) ---\n");
    const size_t BIG_SIZE = 1024 * 1024;
    uint8_t *big_wr = (uint8_t *)malloc(BIG_SIZE);
    uint8_t *big_rd = (uint8_t *)malloc(BIG_SIZE);

    for (size_t i = 0; i < BIG_SIZE; i++)
        big_wr[i] = (uint8_t)((i * 7 + 13) & 0xFF);

    printf("Writing 1 MB...\n");
    xdma_write_ddr3(DDR3_SCRATCH_BASE, big_wr, BIG_SIZE);

    printf("Reading 1 MB...\n");
    xdma_read_ddr3(DDR3_SCRATCH_BASE, big_rd, BIG_SIZE);

    errors = 0;
    for (size_t i = 0; i < BIG_SIZE; i++)
        if (big_wr[i] != big_rd[i]) errors++;

    if (errors == 0)
        printf("PASS: 1 MB transfer verified!\n");
    else
        printf("FAIL: %d errors in 1 MB\n", errors);

    free(wr_buf); free(rd_buf);
    free(big_wr); free(big_rd);
    xdma_cleanup();
    return (errors == 0) ? 0 : 1;
}
```

## E.3. Milestone 2 — Stub IP Round-Trip

```c
/* test_02_stub.c — Full round-trip: PC → DDR3 → Stub IP → DDR3 → PC */
#include "xdma_api.h"
#include "memory_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main()
{
    printf("=== Milestone 2: Stub IP Round-Trip Test ===\n\n");

    if (xdma_init() != 0) return 1;

    /* ── Step 1: Prepare test data (simulated X_int8) ── */
    const size_t XFER_SIZE = 4096;  /* start small */
    uint8_t *input = (uint8_t *)malloc(XFER_SIZE);
    uint8_t *output = (uint8_t *)malloc(XFER_SIZE);

    for (size_t i = 0; i < XFER_SIZE; i++)
        input[i] = (uint8_t)(i & 0xFF);

    /* ── Step 2: Write input to DDR3 (GĐ1 simulation) ── */
    printf("[1/5] Writing %zu bytes to DDR3 @ 0x%08X (input area)...\n",
           XFER_SIZE, (uint32_t)DDR3_INPUT_BASE);
    xdma_write_ddr3(DDR3_INPUT_BASE, input, XFER_SIZE);

    /* ── Step 3: Configure Stub IP via CSR ── */
    printf("[2/5] Configuring Stub IP CSR...\n");
    xdma_csr_write(CSR_SRC_ADDR_LO, (uint32_t)DDR3_INPUT_BASE);
    xdma_csr_write(CSR_SRC_ADDR_HI, 0);
    xdma_csr_write(CSR_DST_ADDR_LO, (uint32_t)DDR3_P3_BASE);
    xdma_csr_write(CSR_DST_ADDR_HI, 0);
    xdma_csr_write(CSR_XFER_BYTES,  (uint32_t)XFER_SIZE);

    /* ── Step 4: Start Stub IP ── */
    printf("[3/5] Starting Stub IP (CSR.start=1)...\n");
    clock_t t_start = clock();
    xdma_csr_write(CSR_CTRL, 0x1);

    /* ── Step 5: Poll for completion ── */
    printf("[4/5] Polling CSR.status for done...\n");
    uint32_t status;
    int timeout = 1000000;
    do {
        status = xdma_csr_read(CSR_STATUS);
        timeout--;
    } while (!(status & 0x2) && timeout > 0);

    clock_t t_end = clock();
    double elapsed_ms = (double)(t_end - t_start) / CLOCKS_PER_SEC * 1000.0;

    if (timeout == 0) {
        printf("FAIL: Timeout waiting for Stub IP done!\n");
        printf("  CSR_STATUS = 0x%08X\n", status);
        xdma_cleanup();
        return 1;
    }

    printf("  Stub IP completed in %.2f ms (status=0x%08X)\n",
           elapsed_ms, status);

    /* Clear IRQ */
    xdma_csr_write(CSR_CTRL, 0x4);

    /* ── Step 6: Read output from DDR3 ── */
    printf("[5/5] Reading output from DDR3 @ 0x%08X...\n",
           (uint32_t)DDR3_P3_BASE);
    memset(output, 0, XFER_SIZE);
    xdma_read_ddr3(DDR3_P3_BASE, output, XFER_SIZE);

    /* ── Step 7: Verify ── */
    /* Stub IP Level 1 = loopback → output should match input */
    int errors = 0;
    for (size_t i = 0; i < XFER_SIZE; i++) {
        if (input[i] != output[i]) {
            if (errors < 10)
                printf("  MISMATCH[%zu]: wrote=0x%02X got=0x%02X\n",
                       i, input[i], output[i]);
            errors++;
        }
    }

    printf("\n══════════════════════════════════════\n");
    if (errors == 0)
        printf("  PASS: Round-trip verified! (%zu bytes)\n", XFER_SIZE);
    else
        printf("  FAIL: %d mismatches out of %zu\n", errors, XFER_SIZE);
    printf("══════════════════════════════════════\n");

    free(input);
    free(output);
    xdma_cleanup();
    return (errors == 0) ? 0 : 1;
}
```

## E.4. Milestone 3+4+5 — GĐ1 + StubIP + GĐ3 Full Pipeline

```c
/* test_05_full_pipeline.c — Complete: Image → Preprocess → DDR3 →
   StubIP → DDR3 → Postprocess → BBox */
#include "xdma_api.h"
#include "memory_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simplified preprocess (no OpenCV dependency for basic test) */
void preprocess_test_image(int8_t *X_int8)
{
    /* Generate a synthetic test image pattern */
    for (int ch = 0; ch < 3; ch++)
        for (int h = 0; h < 640; h++)
            for (int w = 0; w < 640; w++)
                X_int8[ch * 640 * 640 + h * 640 + w] =
                    (int8_t)((ch * 50 + h + w) & 0x7F);
}

int main()
{
    printf("=== Milestone 5: Full Pipeline Test ===\n\n");

    if (xdma_init() != 0) return 1;

    /* ── GĐ1: Preprocess ── */
    printf("[GĐ1] Preprocessing image (1.23 MB)...\n");
    int8_t *X_int8 = (int8_t *)malloc(INPUT_SIZE);
    preprocess_test_image(X_int8);

    printf("[GĐ1] Writing X_int8 to DDR3 @ 0x%08X...\n",
           (uint32_t)DDR3_INPUT_BASE);
    xdma_write_ddr3(DDR3_INPUT_BASE, X_int8, INPUT_SIZE);
    printf("[GĐ1] Done: %.2f MB written.\n", INPUT_SIZE / 1e6);

    /* ── GĐ2: Trigger Stub IP ── */
    printf("\n[GĐ2] Configuring Stub IP...\n");
    xdma_csr_write(CSR_SRC_ADDR_LO, (uint32_t)DDR3_INPUT_BASE);
    xdma_csr_write(CSR_DST_ADDR_LO, (uint32_t)DDR3_P3_BASE);
    xdma_csr_write(CSR_XFER_BYTES,  OUTPUT_TOTAL);

    printf("[GĐ2] Starting Stub IP...\n");
    xdma_csr_write(CSR_CTRL, 0x1);

    /* Poll completion */
    uint32_t status;
    do { status = xdma_csr_read(CSR_STATUS); }
    while (!(status & 0x2));
    xdma_csr_write(CSR_CTRL, 0x4);
    printf("[GĐ2] Stub IP complete. (status=0x%08X)\n", status);

    /* ── GĐ3: Read outputs ── */
    printf("\n[GĐ3] Reading P3 (%d bytes)...\n", P3_SIZE);
    int8_t *P3 = (int8_t *)malloc(P3_SIZE);
    xdma_read_ddr3(DDR3_P3_BASE, P3, P3_SIZE);

    printf("[GĐ3] Reading P4 (%d bytes)...\n", P4_SIZE);
    int8_t *P4 = (int8_t *)malloc(P4_SIZE);
    xdma_read_ddr3(DDR3_P4_BASE, P4, P4_SIZE);

    printf("[GĐ3] Reading P5 (%d bytes)...\n", P5_SIZE);
    int8_t *P5 = (int8_t *)malloc(P5_SIZE);
    xdma_read_ddr3(DDR3_P5_BASE, P5, P5_SIZE);

    /* Verify data integrity (Stub=loopback → P3 should match first P3_SIZE bytes of input) */
    int errors = 0;
    for (int i = 0; i < P3_SIZE && i < INPUT_SIZE; i++) {
        if (P3[i] != X_int8[i]) errors++;
    }

    printf("\n[GĐ3] Postprocessing (placeholder)...\n");
    printf("  P3 first 8 bytes: ");
    for (int i = 0; i < 8; i++) printf("%3d ", P3[i]);
    printf("\n");
    printf("  P4 first 8 bytes: ");
    for (int i = 0; i < 8; i++) printf("%3d ", P4[i]);
    printf("\n");
    printf("  P5 first 8 bytes: ");
    for (int i = 0; i < 8; i++) printf("%3d ", P5[i]);
    printf("\n");

    printf("\n══════════════════════════════════════════════\n");
    if (errors == 0)
        printf("  PASS: Full pipeline round-trip OK!\n");
    else
        printf("  Note: %d mismatches (expected with Stub copy)\n", errors);
    printf("  Total transferred: %.2f MB (write) + %.2f MB (read)\n",
           INPUT_SIZE / 1e6, OUTPUT_TOTAL / 1e6);
    printf("══════════════════════════════════════════════\n");

    free(X_int8); free(P3); free(P4); free(P5);
    xdma_cleanup();
    return 0;
}
```

---

# PHẦN F: TÍCH HỢP REAL IP (Thay Thế Stub)

## F.1. Khi IP GĐ2 hoàn thành

```
BƯỚC 1: Thay Stub IP bằng accel_top trong Vivado Block Design
  → Xóa stub_accel_top instance
  → Thêm accel_top instance
  → Kết nối: s_axil ← AXI Interconnect M01
             m_axi  → AXI Interconnect S01
             irq    → XDMA usr_irq_req[0]
  → CSR address space giữ nguyên: 0x4400_0000

BƯỚC 2: Update CSR writes trong PC application
  → CSR_CTRL, CSR_STATUS giữ nguyên offset
  → Thêm: CSR_NET_DESC_LO/HI, CSR_LAYER_START/END
  → Xóa: CSR_SRC_ADDR/DST_ADDR (IP tự đọc từ descriptors)

BƯỚC 3: Load weights + descriptors vào DDR3
  → PC ghi weight_data vào DDR3_WEIGHT_BASE
  → PC ghi descriptors vào DDR3_DESC_BASE
  → Cần tool Python tạo descriptors từ model config

BƯỚC 4: Run real inference
  → GĐ1: preprocess → ghi X_int8
  → GĐ2: start IP → wait done
  → GĐ3: đọc P3/P4/P5 → postprocess → bbox

BƯỚC 5: Verify kết quả
  → So sánh P3/P4/P5 với Golden Python output
  → Nếu bit-exact → THÀNH CÔNG!
```

## F.2. CSR mapping thay đổi khi chuyển sang Real IP

```
              Stub IP CSR            Real accel_top CSR
              ─────────────          ──────────────────
  0x000       CTRL (start/reset)     CTRL (start/reset/irq_clr)  ← GIỐNG
  0x004       STATUS (busy/done)     STATUS (busy/done/irq/err)  ← GIỐNG
  0x008       VERSION               VERSION                      ← GIỐNG
  0x00C       (unused)              CAP0 (capabilities)          ← MỚI
  0x010       SRC_ADDR_LO           NET_DESC_LO                  ← ĐỔI TÊN
  0x014       SRC_ADDR_HI           NET_DESC_HI                  ← ĐỔI TÊN
  0x018       DST_ADDR_LO           LAYER_START                  ← ĐỔI
  0x01C       (unused)              LAYER_END                    ← MỚI
  0x020       XFER_BYTES            IRQ_MASK                     ← ĐỔI

★ CTRL[0]=start và STATUS[1]=done GIỮA NGUYÊN
→ PC code flow (start → poll done) KHÔNG ĐỔI
→ Chỉ đổi config registers trước khi start
```

---

# PHẦN G: CHECKLIST TỔNG HỢP

```
╔══════════════════════════════════════════════════════════════════════╗
║  PARALLEL DEVELOPMENT CHECKLIST                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  FPGA Side:                                                          ║
║  ☐ Vivado project created for Virtex-7                              ║
║  ☐ XDMA IP added + configured (PCIe Gen2 x4/x8)                   ║
║  ☐ MIG DDR3 added + configured (per board schematic)               ║
║  ☐ AXI Interconnect connects XDMA ↔ DDR3 ↔ Stub IP               ║
║  ☐ Stub IP (stub_accel_top.sv) added                               ║
║  ☐ Constraints file (.xdc) from board vendor                       ║
║  ☐ Bitstream generated + FPGA programmed                           ║
║  ☐ PCIe link up (PC sees device)                                   ║
║                                                                      ║
║  PC Side:                                                            ║
║  ☐ XDMA driver installed (Windows .sys or Linux .ko)               ║
║  ☐ Device files accessible (/dev/xdma0_* or \\.\xdma0_*)          ║
║  ☐ Visual Studio project created                                    ║
║  ☐ xdma_api.c/h compiled                                           ║
║  ☐ OpenCV installed (for GĐ1/GĐ3)                                 ║
║                                                                      ║
║  Milestones:                                                         ║
║  ☐ M0: PCIe link + CSR read (version register)                     ║
║  ☐ M1: DDR3 write 1 MB + read back → 0 errors                     ║
║  ☐ M2: Stub IP round-trip (write→copy→read) → verify              ║
║  ☐ M3: GĐ1 preprocess ảnh → ghi X_int8 vào DDR3                  ║
║  ☐ M4: GĐ3 đọc P3/P4/P5 từ DDR3 → postprocess                   ║
║  ☐ M5: Full pipeline: GĐ1 + Stub + GĐ3 end-to-end                ║
║  ☐ M6: Thay Stub bằng real accel_top → real inference               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

*Tài liệu này hướng dẫn phát triển song song PC software (GĐ1+GĐ3)
và FPGA IP (GĐ2), sử dụng Stub IP để verify đường truyền PCIe→DDR3→IP
trước khi tích hợp real accelerator.*
