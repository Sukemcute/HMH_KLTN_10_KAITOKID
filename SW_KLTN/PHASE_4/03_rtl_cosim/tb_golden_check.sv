`timescale 1ns/1ps
// Co-simulation testbench: loads golden hex data, feeds through accel_top,
// compares P3/P4/P5 output with golden reference.
module tb_golden_check;
  import accel_pkg::*;
  import desc_pkg::*;
  import csr_pkg::*;

  // ───── Parameters ─────
  localparam CLK_PERIOD   = 5;   // 200 MHz
  localparam DDR_SIZE     = 2**22; // 4M x 256-bit words (128 MB)
  localparam DDR_WORDS    = DDR_SIZE;
  localparam AXI_DW       = 256;
  localparam AXI_AW       = 40;

  // DDR3 base addresses (byte addresses, aligned to 32B)
  localparam [39:0] DESC_BASE       = 40'h00_0000_0000;
  localparam [39:0] WEIGHT_BASE     = 40'h00_0010_0000; // 1 MB
  localparam [39:0] INPUT_BASE      = 40'h00_0100_0000; // 16 MB
  localparam [39:0] OUTPUT_BASE     = 40'h00_0200_0000; // 32 MB
  localparam [39:0] GOLDEN_P3_BASE  = 40'h00_0300_0000; // 48 MB
  localparam [39:0] GOLDEN_P4_BASE  = 40'h00_0310_0000;
  localparam [39:0] GOLDEN_P5_BASE  = 40'h00_0320_0000;
  // Descriptor blobs (must match generate_descriptors.py memory_map)
  localparam [39:0] LAYER_TABLE_BASE = 40'h00_0000_0100;
  localparam [39:0] TILE_TABLE_BASE  = 40'h00_0001_0000;

  // Golden hex files — HWC format for input/output, OIHW for weights
  localparam string GOLDEN_DIR = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/02_golden_data/";
  localparam string G_INPUT       = {GOLDEN_DIR, "input_act_hwc.hex"};
  localparam string G_WEIGHTS     = {GOLDEN_DIR, "all_weights.hex"};
  localparam string G_DESC_NET    = {GOLDEN_DIR, "desc_net.hex"};
  localparam string G_DESC_LAYERS = {GOLDEN_DIR, "desc_layers.hex"};
  localparam string G_DESC_TILES  = {GOLDEN_DIR, "desc_tiles.hex"};
  localparam string G_P3_HEX     = {GOLDEN_DIR, "golden_P3_hwc.hex"};
  localparam string G_P4_HEX     = {GOLDEN_DIR, "golden_P4_hwc.hex"};
  localparam string G_P5_HEX     = {GOLDEN_DIR, "golden_P5_hwc.hex"};
  // P3/P4/P5 sizes (quint8, 32 B/line in .hex): match golden_outputs.json
  localparam int P3_HEX_LINES = (1 * 128 * 80 * 80) / 32;   // 25600
  localparam int P4_HEX_LINES = (1 * 256 * 40 * 40) / 32;  // 12800
  localparam int P5_HEX_LINES = (1 * 512 * 20 * 20) / 32;  // 6400
  localparam [39:0] P3_BYTES = 40'd819200;
  localparam [39:0] P4_BYTES = 40'd409600;

  // ───── Clock / Reset ─────
  logic clk = 0;
  logic rst_n;
  always #(CLK_PERIOD/2.0) clk = ~clk;

  // ───── DDR3 Memory Model ─────
  logic [AXI_DW-1:0] ddr_mem [0:DDR_WORDS-1];

  // ───── AXI-Lite signals ─────
  logic [11:0] s_axil_awaddr;
  logic        s_axil_awvalid;
  logic        s_axil_awready;
  logic [31:0] s_axil_wdata;
  logic        s_axil_wvalid;
  logic        s_axil_wready;
  logic [1:0]  s_axil_bresp;
  logic        s_axil_bvalid;
  logic        s_axil_bready;
  logic [11:0] s_axil_araddr;
  logic        s_axil_arvalid;
  logic        s_axil_arready;
  logic [31:0] s_axil_rdata;
  logic        s_axil_rvalid;
  logic        s_axil_rready;

  // ───── AXI4 Master signals ─────
  logic [AXI_AW-1:0] m_axi_araddr;
  logic [7:0]        m_axi_arlen;
  logic              m_axi_arvalid;
  logic              m_axi_arready;
  logic [AXI_DW-1:0] m_axi_rdata;
  logic              m_axi_rvalid;
  logic              m_axi_rlast;
  logic              m_axi_rready;
  logic [AXI_AW-1:0] m_axi_awaddr;
  logic [7:0]        m_axi_awlen;
  logic              m_axi_awvalid;
  logic              m_axi_awready;
  logic [AXI_DW-1:0] m_axi_wdata;
  logic              m_axi_wvalid;
  logic              m_axi_wlast;
  logic              m_axi_wready;
  logic [1:0]        m_axi_bresp;
  logic              m_axi_bvalid;
  logic              m_axi_bready;
  logic              irq;

`ifdef COSIM_DMA_AUDIT
  // Byte range covering L0 act1 arena (must cover compare base + L0 tensor size)
  localparam [39:0] DMA_AUDIT_L0_LO = 40'h00_0180_0000;
  localparam [39:0] DMA_AUDIT_L0_HI = 40'h00_01A0_0000;
  int unsigned      dma_audit_aw_total;
  int unsigned      dma_audit_aw_l0;
  int unsigned      dma_audit_w_l0;
  logic [AXI_DW-1:0] dma_audit_first_wdata_l0;
  logic              dma_audit_seen_first_wdata_l0;
`endif

  // ───── PPU Parameter Signals (active layer) ─────
  logic signed [31:0] ppu_bias    [32];
  logic signed [31:0] ppu_m_int   [32];
  logic        [5:0]  ppu_shift   [32];
  logic signed [7:0]  ppu_zp_out;
  logic signed [7:0]  ppu_silu_lut [256];

  // Pre-loaded PPU params for all layers (up to 24)
  localparam int MAX_LAYERS = 24;
  logic signed [31:0] all_ppu_bias   [MAX_LAYERS][32];
  logic signed [31:0] all_ppu_m_int  [MAX_LAYERS][32];
  logic        [5:0]  all_ppu_shift  [MAX_LAYERS][32];
  logic signed [7:0]  all_ppu_zp_out [MAX_LAYERS];
  logic signed [7:0]  all_ppu_silu   [MAX_LAYERS][256];

  // Initialize with safe defaults, then load from hex in initial block
  initial begin
    for (int lay = 0; lay < MAX_LAYERS; lay++) begin
      for (int ch = 0; ch < 32; ch++) begin
        all_ppu_bias[lay][ch]  = 32'sd0;
        all_ppu_m_int[lay][ch] = 32'sd1;
        all_ppu_shift[lay][ch] = 6'd0;
      end
      all_ppu_zp_out[lay] = 8'sd0;
      for (int i = 0; i < 256; i++)
        all_ppu_silu[lay][i] = $signed(8'(i));
    end
  end

  // Load PPU params for one layer from hex files
  task automatic load_ppu_hex_for_layer(input int layer_idx);
    string bias_file, m_file, shift_file, zp_file, lut_file;
    integer fd;
    string line;
    logic [31:0] val32;
    logic [7:0]  val8;

    $sformat(bias_file,  "%sppu_bias_L%02d.hex", GOLDEN_DIR, layer_idx);
    $sformat(m_file,     "%sppu_m_int_L%02d.hex", GOLDEN_DIR, layer_idx);
    $sformat(shift_file, "%sppu_shift_L%02d.hex", GOLDEN_DIR, layer_idx);
    $sformat(zp_file,    "%sppu_zp_out_L%02d.hex", GOLDEN_DIR, layer_idx);
    $sformat(lut_file,   "%ssilu_lut_L%02d.hex", GOLDEN_DIR, layer_idx);

    // Load bias (INT32, 8 values per line)
    fd = $fopen(bias_file, "r");
    if (fd != 0) begin
      for (int ch = 0; ch < 32; ch += 8) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 8 && (ch+j) < 32; j++) begin
            val32 = '0;
            for (int k = 0; k < 8; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*8 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              val32 = {val32[27:0], nib};
            end
            all_ppu_bias[layer_idx][ch+j] = $signed(val32);
          end
        end
      end
      $fclose(fd);
    end

    // Load m_int (same format as bias)
    fd = $fopen(m_file, "r");
    if (fd != 0) begin
      for (int ch = 0; ch < 32; ch += 8) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 8 && (ch+j) < 32; j++) begin
            val32 = '0;
            for (int k = 0; k < 8; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*8 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              val32 = {val32[27:0], nib};
            end
            all_ppu_m_int[layer_idx][ch+j] = $signed(val32);
          end
        end
      end
      $fclose(fd);
    end

    // Load shift (UINT8, 32 values per line)
    fd = $fopen(shift_file, "r");
    if (fd != 0) begin
      if ($fgets(line, fd)) begin
        for (int ch = 0; ch < 32; ch++) begin
          val8 = '0;
          for (int k = 0; k < 2; k++) begin
            automatic logic [3:0] nib;
            automatic byte c = line.getc(ch*2 + k);
            if      (c >= "0" && c <= "9") nib = c - "0";
            else if (c >= "A" && c <= "F") nib = c - "A" + 10;
            else if (c >= "a" && c <= "f") nib = c - "a" + 10;
            else nib = 0;
            val8 = {val8[3:0], nib};
          end
          all_ppu_shift[layer_idx][ch] = val8[5:0];
        end
      end
      $fclose(fd);
    end

    // Load zp_out (single INT8)
    fd = $fopen(zp_file, "r");
    if (fd != 0) begin
      if ($fgets(line, fd)) begin
        val8 = '0;
        for (int k = 0; k < 2; k++) begin
          automatic logic [3:0] nib;
          automatic byte c = line.getc(k);
          if      (c >= "0" && c <= "9") nib = c - "0";
          else if (c >= "A" && c <= "F") nib = c - "A" + 10;
          else if (c >= "a" && c <= "f") nib = c - "a" + 10;
          else nib = 0;
          val8 = {val8[3:0], nib};
        end
        all_ppu_zp_out[layer_idx] = $signed(val8);
      end
      $fclose(fd);
    end

    // Load SiLU LUT (256 UINT8 values)
    fd = $fopen(lut_file, "r");
    if (fd != 0) begin
      for (int i = 0; i < 256; i += 32) begin
        if (!$feof(fd) && $fgets(line, fd)) begin
          for (int j = 0; j < 32 && (i+j) < 256; j++) begin
            val8 = '0;
            for (int k = 0; k < 2; k++) begin
              automatic logic [3:0] nib;
              automatic byte c = line.getc(j*2 + k);
              if      (c >= "0" && c <= "9") nib = c - "0";
              else if (c >= "A" && c <= "F") nib = c - "A" + 10;
              else if (c >= "a" && c <= "f") nib = c - "a" + 10;
              else nib = 0;
              val8 = {val8[3:0], nib};
            end
            all_ppu_silu[layer_idx][i+j] = $signed(val8);
          end
        end
      end
      $fclose(fd);
    end

    $display("  PPU params loaded for L%0d", layer_idx);
  endtask

  // Dynamically update PPU params when layer changes
  logic [4:0] ppu_cur_layer;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ppu_cur_layer <= 5'h1F;
      for (int ch = 0; ch < 32; ch++) begin
        ppu_bias[ch]  <= 32'sd0;
        ppu_m_int[ch] <= 32'sd1;
        ppu_shift[ch] <= 6'd0;
      end
      ppu_zp_out <= 8'sd0;
      for (int i = 0; i < 256; i++)
        ppu_silu_lut[i] <= $signed(8'(i));
    end else begin
      automatic logic [4:0] active_layer;
      active_layer = u_dut.u_ctrl.u_fetch.current_layer;
      if (active_layer != ppu_cur_layer && active_layer < MAX_LAYERS) begin
        ppu_cur_layer <= active_layer;
        for (int ch = 0; ch < 32; ch++) begin
          ppu_bias[ch]  <= all_ppu_bias[active_layer][ch];
          ppu_m_int[ch] <= all_ppu_m_int[active_layer][ch];
          ppu_shift[ch] <= all_ppu_shift[active_layer][ch];
        end
        ppu_zp_out <= all_ppu_zp_out[active_layer];
        for (int i = 0; i < 256; i++)
          ppu_silu_lut[i] <= all_ppu_silu[active_layer][i];
        $display("[TB-PPU] @%0t Layer switch: L%0d -> L%0d | bias[0]=%0d m_int[0]=%0d shift[0]=%0d zp=%0d",
                 $time, ppu_cur_layer, active_layer,
                 all_ppu_bias[active_layer][0],
                 all_ppu_m_int[active_layer][0],
                 all_ppu_shift[active_layer][0],
                 all_ppu_zp_out[active_layer]);
      end
    end
  end

  // ───── DUT ─────
  accel_top u_dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axil_awaddr    (s_axil_awaddr),
    .s_axil_awvalid   (s_axil_awvalid),
    .s_axil_awready   (s_axil_awready),
    .s_axil_wdata     (s_axil_wdata),
    .s_axil_wvalid    (s_axil_wvalid),
    .s_axil_wready    (s_axil_wready),
    .s_axil_bresp     (s_axil_bresp),
    .s_axil_bvalid    (s_axil_bvalid),
    .s_axil_bready    (s_axil_bready),
    .s_axil_araddr    (s_axil_araddr),
    .s_axil_arvalid   (s_axil_arvalid),
    .s_axil_arready   (s_axil_arready),
    .s_axil_rdata     (s_axil_rdata),
    .s_axil_rvalid    (s_axil_rvalid),
    .s_axil_rready    (s_axil_rready),
    .m_axi_araddr     (m_axi_araddr),
    .m_axi_arlen      (m_axi_arlen),
    .m_axi_arvalid    (m_axi_arvalid),
    .m_axi_arready    (m_axi_arready),
    .m_axi_rdata      (m_axi_rdata),
    .m_axi_rvalid     (m_axi_rvalid),
    .m_axi_rlast      (m_axi_rlast),
    .m_axi_rready     (m_axi_rready),
    .m_axi_awaddr     (m_axi_awaddr),
    .m_axi_awlen      (m_axi_awlen),
    .m_axi_awvalid    (m_axi_awvalid),
    .m_axi_awready    (m_axi_awready),
    .m_axi_wdata      (m_axi_wdata),
    .m_axi_wvalid     (m_axi_wvalid),
    .m_axi_wlast      (m_axi_wlast),
    .m_axi_wready     (m_axi_wready),
    .m_axi_bresp      (m_axi_bresp),
    .m_axi_bvalid     (m_axi_bvalid),
    .m_axi_bready     (m_axi_bready),
    .irq              (irq),
    .ppu_bias         (ppu_bias),
    .ppu_m_int        (ppu_m_int),
    .ppu_shift        (ppu_shift),
    .ppu_zp_out       (ppu_zp_out),
    .ppu_silu_lut     (ppu_silu_lut)
  );

  // ═══════════ AXI4 Memory Slave Model ═══════════
  // Handles read and write transactions to ddr_mem[]
  logic [7:0]  rd_burst_cnt;
  logic [39:0] rd_burst_addr;
  logic        rd_burst_active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axi_arready   <= 1'b1;
      m_axi_rvalid    <= 1'b0;
      m_axi_rlast     <= 1'b0;
      m_axi_rdata     <= '0;
      rd_burst_active  <= 1'b0;
      rd_burst_cnt     <= '0;
      rd_burst_addr    <= '0;
    end else begin
      if (m_axi_arvalid && m_axi_arready && !rd_burst_active) begin
        rd_burst_addr   <= m_axi_araddr;
        rd_burst_cnt    <= m_axi_arlen;
        rd_burst_active <= 1'b1;
        m_axi_arready   <= 1'b0;
        m_axi_rvalid    <= 1'b1;
        m_axi_rdata     <= ddr_mem[m_axi_araddr[26:5]];
        m_axi_rlast     <= (m_axi_arlen == 0);
      end else if (rd_burst_active && m_axi_rready && m_axi_rvalid) begin
        if (m_axi_rlast) begin
          rd_burst_active <= 1'b0;
          m_axi_rvalid    <= 1'b0;
          m_axi_rlast     <= 1'b0;
          m_axi_arready   <= 1'b1;
        end else begin
          rd_burst_addr <= rd_burst_addr + 32;
          rd_burst_cnt  <= rd_burst_cnt - 1;
          m_axi_rdata   <= ddr_mem[(rd_burst_addr + 32) >> 5];
          m_axi_rlast   <= (rd_burst_cnt == 1);
        end
      end
    end
  end

  // AXI Write channel
  logic [39:0] wr_burst_addr;
  logic [7:0]  wr_burst_cnt;
  logic        wr_burst_active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axi_awready   <= 1'b1;
      m_axi_wready    <= 1'b0;
      m_axi_bvalid    <= 1'b0;
      m_axi_bresp     <= 2'b00;
      wr_burst_active  <= 1'b0;
      wr_burst_addr    <= '0;
      wr_burst_cnt     <= '0;
`ifdef COSIM_DMA_AUDIT
      dma_audit_w_l0                <= 0;
      dma_audit_first_wdata_l0      <= '0;
      dma_audit_seen_first_wdata_l0 <= 1'b0;
`endif
    end else begin
      if (m_axi_awvalid && m_axi_awready && !wr_burst_active) begin
        wr_burst_addr   <= m_axi_awaddr;
        wr_burst_cnt    <= m_axi_awlen;
        wr_burst_active <= 1'b1;
        m_axi_awready   <= 1'b0;
        m_axi_wready    <= 1'b1;
      end else if (wr_burst_active && m_axi_wvalid && m_axi_wready) begin
`ifdef COSIM_DMA_AUDIT
        if (wr_burst_addr >= DMA_AUDIT_L0_LO && wr_burst_addr < DMA_AUDIT_L0_HI) begin
          dma_audit_w_l0 <= dma_audit_w_l0 + 1;
          if (!dma_audit_seen_first_wdata_l0) begin
            dma_audit_first_wdata_l0      <= m_axi_wdata;
            dma_audit_seen_first_wdata_l0 <= 1'b1;
          end
        end
`endif
        ddr_mem[wr_burst_addr[26:5]] <= m_axi_wdata;
        if (m_axi_wlast) begin
          wr_burst_active <= 1'b0;
          m_axi_wready    <= 1'b0;
          m_axi_bvalid    <= 1'b1;
          m_axi_awready   <= 1'b1;
        end else begin
          wr_burst_addr <= wr_burst_addr + 32;
          wr_burst_cnt  <= wr_burst_cnt - 1;
        end
      end
      if (m_axi_bvalid && m_axi_bready)
        m_axi_bvalid <= 1'b0;
    end
  end

`ifdef COSIM_DMA_AUDIT
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dma_audit_aw_total <= 0;
      dma_audit_aw_l0    <= 0;
    end else if (m_axi_awvalid && m_axi_awready) begin
      dma_audit_aw_total <= dma_audit_aw_total + 1;
      if (m_axi_awaddr >= DMA_AUDIT_L0_LO && m_axi_awaddr < DMA_AUDIT_L0_HI)
        dma_audit_aw_l0 <= dma_audit_aw_l0 + 1;
    end
  end
`endif

  // ═══════════ AXI-Lite Helper Tasks ═══════════
  task automatic axil_write(input logic [11:0] addr, input logic [31:0] data);
    @(negedge clk);
    s_axil_awaddr  = addr;
    s_axil_awvalid = 1'b1;
    s_axil_wdata   = data;
    s_axil_wvalid  = 1'b1;
    s_axil_bready  = 1'b1;
    fork
      begin
        wait (s_axil_awready);
        @(posedge clk); #1;
        s_axil_awvalid = 1'b0;
      end
      begin
        wait (s_axil_wready);
        @(posedge clk); #1;
        s_axil_wvalid = 1'b0;
      end
    join
    wait (s_axil_bvalid);
    @(posedge clk); #1;
    s_axil_bready = 1'b0;
  endtask

  task automatic axil_read(input logic [11:0] addr, output logic [31:0] data);
    @(negedge clk);
    s_axil_araddr  = addr;
    s_axil_arvalid = 1'b1;
    s_axil_rready  = 1'b1;
    wait (s_axil_arready);
    @(posedge clk); #1;
    s_axil_arvalid = 1'b0;
    wait (s_axil_rvalid);
    data = s_axil_rdata;
    @(posedge clk); #1;
    s_axil_rready = 1'b0;
  endtask

  // ═══════════ DDR Memory Load Helpers ═══════════
  task automatic load_hex_to_ddr(
    input string filename,
    input logic [39:0] base_addr
  );
    integer fd, line_num, word_idx;
    string line;
    logic [AXI_DW-1:0] word_val;

    fd = $fopen(filename, "r");
    if (fd == 0) begin
      $display("ERROR: Cannot open %s", filename);
      return;
    end

    line_num = 0;
    word_idx = base_addr[26:5];
    while (!$feof(fd)) begin
      if ($fgets(line, fd)) begin
        if (line.len() >= 64) begin
          word_val = '0;
          for (int i = 0; i < 64; i++) begin
            automatic logic [3:0] nibble;
            automatic byte c = line.getc(i);
            if      (c >= "0" && c <= "9") nibble = c - "0";
            else if (c >= "A" && c <= "F") nibble = c - "A" + 10;
            else if (c >= "a" && c <= "f") nibble = c - "a" + 10;
            else nibble = 0;
            word_val = {word_val[AXI_DW-5:0], nibble};
          end
          ddr_mem[word_idx] = word_val;
          word_idx = word_idx + 1;
          line_num = line_num + 1;
        end
      end
    end
    $fclose(fd);
    $display("  Loaded %0d lines from %s to DDR[0x%08h]", line_num, filename, base_addr);
  endtask

  // ═══════════ Compare DDR region with golden hex ═══════════
  task automatic compare_ddr_with_golden(
    input string golden_file,
    input logic [39:0] ddr_addr,
    input int num_lines,
    input string feature_name,
    output int mismatches
  );
    integer fd;
    string line;
    logic [AXI_DW-1:0] golden_word, ddr_word;
    int word_idx, line_cnt;

    mismatches = 0;
    fd = $fopen(golden_file, "r");
    if (fd == 0) begin
      $display("ERROR: Cannot open golden file %s", golden_file);
      mismatches = -1;
      return;
    end

    word_idx = ddr_addr[26:5];
    line_cnt = 0;
    while (!$feof(fd) && line_cnt < num_lines) begin
      if ($fgets(line, fd)) begin
        if (line.len() >= 64) begin
          golden_word = '0;
          for (int i = 0; i < 64; i++) begin
            automatic logic [3:0] nibble;
            automatic byte c = line.getc(i);
            if      (c >= "0" && c <= "9") nibble = c - "0";
            else if (c >= "A" && c <= "F") nibble = c - "A" + 10;
            else if (c >= "a" && c <= "f") nibble = c - "a" + 10;
            else nibble = 0;
            golden_word = {golden_word[AXI_DW-5:0], nibble};
          end

          ddr_word = ddr_mem[word_idx];
          if (ddr_word !== golden_word) begin
            if (mismatches < 20)
              $display("  MISMATCH %s line[%0d] byte_addr=0x%010h: RTL=%h GOLDEN=%h",
                       feature_name, line_cnt,
                       ddr_addr + 40'(line_cnt) * 40'd32, ddr_word, golden_word);
            mismatches = mismatches + 1;
          end
          word_idx = word_idx + 1;
          line_cnt = line_cnt + 1;
        end
      end
    end
    $fclose(fd);
  endtask

  // ═══════════ Main Test Sequence ═══════════
  int p3_errors, p4_errors, p5_errors;
  logic [31:0] rd_val;
  // Plusargs (must be module scope: Vivado xvlog rejects mid-initial declarations)
  int cosim_layer_start_plusarg;
  int cosim_layer_end_plusarg;

  initial begin
    $display("+====================================================+");
    $display("| PHASE 4: Golden Co-Simulation Testbench            |");
    $display("+====================================================+");

    // Init signals
    rst_n          = 1'b0;
    s_axil_awaddr  = '0;
    s_axil_awvalid = 1'b0;
    s_axil_wdata   = '0;
    s_axil_wvalid  = 1'b0;
    s_axil_bready  = 1'b0;
    s_axil_araddr  = '0;
    s_axil_arvalid = 1'b0;
    s_axil_rready  = 1'b0;

    // Clear DDR
    for (int i = 0; i < DDR_WORDS; i++) ddr_mem[i] = '0;

    // Reset
    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    // ═══════════ STEP 1: Load golden data into DDR ═══════════
    $display("\n=== STEP 1: Loading golden data into DDR model ===");

    // Input activations (HWC format)
    load_hex_to_ddr(G_INPUT,  INPUT_BASE);
    // All layer weights (OIHW, concatenated + aligned)
    load_hex_to_ddr(G_WEIGHTS, WEIGHT_BASE);
    // Descriptor tables
    load_hex_to_ddr(G_DESC_NET,    DESC_BASE);
    load_hex_to_ddr(G_DESC_LAYERS, LAYER_TABLE_BASE);
    load_hex_to_ddr(G_DESC_TILES,  TILE_TABLE_BASE);

    // Golden P3/P4/P5 outputs (HWC format for comparison)
    load_hex_to_ddr(G_P3_HEX, GOLDEN_P3_BASE);
    load_hex_to_ddr(G_P4_HEX, GOLDEN_P4_BASE);
    load_hex_to_ddr(G_P5_HEX, GOLDEN_P5_BASE);

    // Load PPU parameters for all layers
    $display("\n  Loading PPU parameters per layer...");
    for (int li = 0; li < 23; li++)
      load_ppu_hex_for_layer(li);

    // ═══════════ STEP 2: Configure accelerator via CSR ═══════════
    $display("\n=== STEP 2: Configuring accelerator via AXI-Lite CSR ===");

    // Read VERSION register
    axil_read(CSR_VERSION, rd_val);
    $display("  VERSION = 0x%08h", rd_val);

    // Write NET_DESC base address
    axil_write(CSR_NET_DESC_LO, DESC_BASE[31:0]);
    axil_write(CSR_NET_DESC_HI, {24'b0, DESC_BASE[39:32]});
    $display("  NET_DESC_BASE = 0x%010h", DESC_BASE);

    // Layer range: +COSIM_LAYER_START=N +COSIM_LAYER_END=M overrides default (YOLOv10n per-layer sweep).
    cosim_layer_start_plusarg = -1;
    cosim_layer_end_plusarg   = -1;
    if ($value$plusargs("COSIM_LAYER_START=%d", cosim_layer_start_plusarg)) begin end
    if ($value$plusargs("COSIM_LAYER_END=%d", cosim_layer_end_plusarg)) begin end
    if (cosim_layer_start_plusarg >= 0 && cosim_layer_end_plusarg >= 0) begin
      axil_write(CSR_LAYER_START, cosim_layer_start_plusarg);
      axil_write(CSR_LAYER_END, cosim_layer_end_plusarg);
      $display("  Layer range: %0d..%0d (+COSIM_LAYER_START/END)",
               cosim_layer_start_plusarg, cosim_layer_end_plusarg);
    end else begin
      axil_write(CSR_LAYER_START, 32'd0);
`ifdef COSIM_FULL_MODEL
      axil_write(CSR_LAYER_END, 32'd22);
      $display("  Layer range: 0 to 22 (full model)");
`else
      axil_write(CSR_LAYER_END, 32'd0);
      $display("  Layer range: 0 to 0 (L0 single-layer verify)");
`endif
    end

    // Enable IRQ
    axil_write(CSR_IRQ_MASK, 32'h0000_0001);

    // ═══════════ STEP 3: Start inference ═══════════
    $display("\n=== STEP 3: Starting inference (CTRL.START=1) ===");
    axil_write(CSR_CTRL, 32'h0000_0001);

    // Poll STATUS until DONE or IRQ.
    // Note: IRQ fires when the descriptor-fetch walk completes (DF_DONE).
    // SC tiles may still be computing + writing back via DMA, so we add
    // a generous post-IRQ window for all DMA writes to finish.
    $display("  Waiting for completion...");
    rd_val = 0;
    fork
      begin : wait_irq
        @(posedge irq);
        $display("  IRQ asserted at time %0t (desc walk done, waiting for SC compute)", $time);
        repeat (2_000_000) @(posedge clk);
        $display("  Post-IRQ compute window elapsed at %0t", $time);
      end
      begin : timeout_check
        repeat (20_000_000) @(posedge clk);
`ifdef ACCEL_PHASE_A_DBG
        phase_a_print_snapshot("TIMEOUT_SNAPSHOT");
`endif
        $display("  TIMEOUT after 20M cycles!");
      end
    join_any
    disable wait_irq;
    disable timeout_check;

    // Read final status
    axil_read(CSR_STATUS, rd_val);
    $display("  STATUS = 0x%08h", rd_val);

    // Read performance counters
    axil_read(CSR_PERF_CYCLE_LO, rd_val);
    $display("  PERF_CYCLES = %0d", rd_val);
    axil_read(CSR_PERF_TILE_DONE, rd_val);
    $display("  TILES_DONE = %0d", rd_val);

`ifdef COSIM_DMA_AUDIT
    $display("  [COSIM_DMA_AUDIT] AXI AW (all masters) total=%0d", dma_audit_aw_total);
    $display("  [COSIM_DMA_AUDIT] AW with addr in L0 act1 [%010h .. %010h) = %0d",
             DMA_AUDIT_L0_LO, DMA_AUDIT_L0_HI, dma_audit_aw_l0);
    $display("  [COSIM_DMA_AUDIT] W beats committed to DDR in that range = %0d (first_wdata=%h)",
             dma_audit_w_l0, dma_audit_first_wdata_l0);
    if (dma_audit_aw_l0 == 0)
      $display("  [COSIM_DMA_AUDIT] No writes to L0 act1 — check tile spill (dma_wr), ext_wr_grant, tensor_dma, bh_act_out.");
`endif

    // ═══════════ STEP 4: Compare outputs ═══════════
    $display("\n=== STEP 4: Comparing RTL output with golden reference ===");

`ifdef COSIM_FULL_MODEL
    // Full model: compare P3/P4/P5
    compare_ddr_with_golden(
      G_P3_HEX,
      OUTPUT_BASE, P3_HEX_LINES, "P3", p3_errors
    );
    compare_ddr_with_golden(
      G_P4_HEX,
      OUTPUT_BASE + P3_BYTES, P4_HEX_LINES, "P4", p4_errors
    );
    compare_ddr_with_golden(
      G_P5_HEX,
      OUTPUT_BASE + P3_BYTES + P4_BYTES, P5_HEX_LINES, "P5", p5_errors
    );
`else
    // L0 single-layer: compare output at act1 arena (ping-pong output)
    // L0: cout=16, hout=320, wout=320 → 16*320*320 = 1,638,400 bytes
    //     = 51200 × 32B lines
    begin
      localparam [39:0] L0_OUT_BASE = 40'h00_0180_0000;  // act1_arena
      localparam int L0_OUT_LINES = (16 * 320 * 320) / 32;  // 51200
      string l0_golden_hex;
      l0_golden_hex = {GOLDEN_DIR, "layer_by_layer/L00_output_hwc.hex"};
      compare_ddr_with_golden(
        l0_golden_hex,
        L0_OUT_BASE, L0_OUT_LINES, "L0_output", p3_errors
      );
    end
    p4_errors = 0;
    p5_errors = 0;
`endif

    // ═══════════ STEP 5: Report ═══════════
    $display("\n====================================================");
`ifdef COSIM_FULL_MODEL
    $display("  P3 mismatches: %0d / %0d", p3_errors, P3_HEX_LINES);
    $display("  P4 mismatches: %0d / %0d", p4_errors, P4_HEX_LINES);
    $display("  P5 mismatches: %0d / %0d", p5_errors, P5_HEX_LINES);
`else
    $display("  L0 output mismatches: %0d / %0d", p3_errors, (16*320*320)/32);
`endif
    if (p3_errors == 0 && p4_errors == 0 && p5_errors == 0)
      $display("  ALL OUTPUTS BIT-EXACT");
    else
      $display("  MISMATCHES FOUND - debug layer-by-layer");
    $display("====================================================");

    repeat (100) @(posedge clk);
    $finish;
  end

  // ═══════════ DMA Write Activity Monitor ═══════════
  always_ff @(posedge clk) begin
    if (rst_n && wr_burst_active && m_axi_wvalid && m_axi_wready) begin
      if (m_axi_wlast)
        $display("[DMA-WR] @%0t addr=0x%010h data[31:0]=%08h (LAST)",
                 $time, wr_burst_addr, m_axi_wdata[31:0]);
    end
  end

  // ═══════════ Per-Layer Completion Tracker ═══════════
  logic [7:0] layer_done_tracker;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      layer_done_tracker <= 8'hFF;
    else begin
      automatic logic [7:0] cur = u_dut.u_ctrl.u_fetch.current_layer;
      if (cur != layer_done_tracker && layer_done_tracker != 8'hFF)
        $display("[LAYER-DONE] @%0t Layer L%0d completed, now starting L%0d",
                 $time, layer_done_tracker, cur);
      layer_done_tracker <= cur;
    end
  end

  // ═══════════ Phase A: hierarchical deadlock observability (compile TB with -d ACCEL_PHASE_A_DBG) ═══════════
`ifdef ACCEL_PHASE_A_DBG
  localparam int PHASE_A_STALL_THRESHOLD = 10_000;

  function automatic string phase_a_df_str(input int s);
    case (s)
      0:  return "DF_IDLE";
      1:  return "DF_FETCH_NET_AR";
      2:  return "DF_FETCH_NET_R";
      3:  return "DF_PARSE_NET";
      4:  return "DF_FETCH_LAYER_AR";
      5:  return "DF_FETCH_LAYER_R";
      6:  return "DF_PARSE_LAYER";
      7:  return "DF_FETCH_TILE_AR";
      8:  return "DF_FETCH_TILE_R";
      9:  return "DF_DISPATCH_TILE";
      10: return "DF_NEXT_TILE";
      11: return "DF_NEXT_LAYER";
      12: return "DF_DONE";
      default: return $sformatf("DF_?(%0d)", s);
    endcase
  endfunction

  function automatic string phase_a_gs_str(input int s);
    case (s)
      0: return "GS_IDLE";
      1: return "GS_DISPATCH";
      2: return "GS_WAIT_ACCEPT";
      3: return "GS_DONE";
      default: return $sformatf("GS_?(%0d)", s);
    endcase
  endfunction

  task automatic phase_a_print_snapshot(input string tag);
    int df_s, gs_s;
    logic [3:0] sc_tv, sc_acc;
    begin
      df_s = int'(u_dut.u_ctrl.u_fetch.state);
      gs_s = int'(u_dut.u_ctrl.u_sched.state);
      sc_acc = {u_dut.gen_sc[3].u_sc.tile_accept, u_dut.gen_sc[2].u_sc.tile_accept,
                u_dut.gen_sc[1].u_sc.tile_accept, u_dut.gen_sc[0].u_sc.tile_accept};
      sc_tv  = {u_dut.gen_sc[3].u_sc.tile_valid, u_dut.gen_sc[2].u_sc.tile_valid,
                u_dut.gen_sc[1].u_sc.tile_valid, u_dut.gen_sc[0].u_sc.tile_valid};
      $display("[PHASE-A] %s @%0t DF=%s GS=%s tile_v=%b tile_rdy=%b sc_mask=%b sc_disp=%b tile_id=%0h",
               tag, $time, phase_a_df_str(df_s), phase_a_gs_str(gs_s),
               u_dut.u_ctrl.u_fetch.tile_desc_valid, u_dut.u_ctrl.u_fetch.tile_desc_ready,
               u_dut.u_ctrl.u_sched.sc_mask, u_dut.u_ctrl.u_sched.sc_dispatched,
               u_dut.u_ctrl.u_sched.tile_reg.tile_id);
      $display("[PHASE-A]   L/tc/tt fetch: L=%0d tc=%0d tt=%0d | gs: L=%0d td=%0d tt=%0d",
               u_dut.u_ctrl.u_fetch.current_layer,
               u_dut.u_ctrl.u_fetch.tile_cnt, u_dut.u_ctrl.u_fetch.tile_total,
               u_dut.u_ctrl.u_sched.layer_id_reg,
               u_dut.u_ctrl.u_sched.tiles_dispatched, u_dut.u_ctrl.u_sched.tiles_total);
      $display("[PHASE-A]   AXI(fetch) arv=%b arr=%b rv=%b rr=%b rlast=%b | top ctrl_rd_busy=%b m_arv=%b m_arr=%b m_rv=%b m_rr=%b m_rlast=%b",
               u_dut.u_ctrl.u_fetch.axi_arvalid, u_dut.u_ctrl.u_fetch.axi_arready,
               u_dut.u_ctrl.u_fetch.axi_rvalid, u_dut.u_ctrl.u_fetch.axi_rready,
               u_dut.u_ctrl.u_fetch.axi_rlast,
               u_dut.ctrl_rd_busy,
               m_axi_arvalid, m_axi_arready, m_axi_rvalid, m_axi_rready, m_axi_rlast);
      $display("[PHASE-A]   SC tile_valid=%b tile_accept=%b", sc_tv, sc_acc);
    end
  endtask

  int                phase_a_prev_df;
  int                phase_a_df_stall;
  int                phase_a_prev_gs;
  logic [4:0] phase_a_prev_layer;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      phase_a_prev_df   <= -1;
      phase_a_df_stall  <= 0;
    end else begin
      if (int'(u_dut.u_ctrl.u_fetch.state) != phase_a_prev_df) begin
        $display("[PHASE-A] DF_STATE @%0t %s", $time,
                 phase_a_df_str(int'(u_dut.u_ctrl.u_fetch.state)));
        phase_a_prev_df  <= int'(u_dut.u_ctrl.u_fetch.state);
        phase_a_df_stall <= 0;
      end else begin
        // desc_fetch_engine: DF_IDLE=0, DF_DONE=12 — idle after walk is not a control stall.
        if (int'(u_dut.u_ctrl.u_fetch.state) != 0 && int'(u_dut.u_ctrl.u_fetch.state) != 12) begin
          phase_a_df_stall <= phase_a_df_stall + 1;
          if (phase_a_df_stall == PHASE_A_STALL_THRESHOLD) begin
            phase_a_print_snapshot("STALL");
            phase_a_df_stall <= 0;
          end
        end else
          phase_a_df_stall <= 0;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      phase_a_prev_gs <= -1;
    else if (int'(u_dut.u_ctrl.u_sched.state) != phase_a_prev_gs) begin
      $display("[PHASE-A] GS_STATE @%0t %s", $time,
               phase_a_gs_str(int'(u_dut.u_ctrl.u_sched.state)));
      phase_a_prev_gs <= int'(u_dut.u_ctrl.u_sched.state);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      phase_a_prev_layer <= 5'h1f;
    else if (u_dut.u_ctrl.u_fetch.current_layer != phase_a_prev_layer) begin
      $display("[PHASE-A] LAYER_ID @%0t fetch_current_layer=%0d", $time,
               u_dut.u_ctrl.u_fetch.current_layer);
      phase_a_prev_layer <= u_dut.u_ctrl.u_fetch.current_layer;
    end
  end
`endif

endmodule
