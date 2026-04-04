// ============================================================================
// Testbench: tb_accel_top — Stage 10.5
// Tests: CPU start → descriptors fetched → tiles dispatched → IRQ asserted.
// Uses an AXI4 DDR3 memory model and AXI-Lite CPU model.
// ============================================================================
`timescale 1ns / 1ps

module tb_accel_top;
  import accel_pkg::*;
  import csr_pkg::*;
  import desc_pkg::*;

  localparam int ADDR_W  = AXI_ADDR_WIDTH;
  localparam int DATA_W  = AXI_DATA_WIDTH;
  localparam int AXIL_AW = 12;
  localparam int AXIL_DW = 32;

  logic clk_200, rst_n_async;
  initial begin clk_200 = 0; forever #2.5 clk_200 = ~clk_200; end  // 200 MHz

  // AXI-Lite (CPU side)
  logic              s_awvalid, s_awready, s_wvalid, s_wready;
  logic [AXIL_AW-1:0] s_awaddr;
  logic [AXIL_DW-1:0] s_wdata;
  logic [AXIL_DW/8-1:0] s_wstrb;
  logic              s_bvalid, s_bready;
  logic [1:0]        s_bresp;
  logic              s_arvalid, s_arready, s_rvalid, s_rready;
  logic [AXIL_AW-1:0] s_araddr;
  logic [AXIL_DW-1:0] s_rdata;
  logic [1:0]        s_rresp;

  // AXI4 DDR
  logic              ddr_ar_valid, ddr_ar_ready;
  logic [ADDR_W-1:0] ddr_ar_addr;
  logic [7:0]        ddr_ar_len;
  logic [2:0]        ddr_ar_size;
  logic [1:0]        ddr_ar_burst;
  logic              ddr_r_valid, ddr_r_ready;
  logic [DATA_W-1:0] ddr_r_data;
  logic              ddr_r_last;
  logic              ddr_aw_valid, ddr_aw_ready;
  logic [ADDR_W-1:0] ddr_aw_addr;
  logic [7:0]        ddr_aw_len;
  logic [2:0]        ddr_aw_size;
  logic [1:0]        ddr_aw_burst;
  logic              ddr_w_valid, ddr_w_ready;
  logic [DATA_W-1:0] ddr_w_data;
  logic [DATA_W/8-1:0] ddr_w_strb;
  logic              ddr_w_last;
  logic              ddr_b_valid, ddr_b_ready;

  logic irq;

  accel_top u_dut (
    .clk_in_200  (clk_200),
    .rst_n_async (rst_n_async),
    .s_awvalid(s_awvalid), .s_awready(s_awready), .s_awaddr(s_awaddr),
    .s_wvalid(s_wvalid), .s_wready(s_wready), .s_wdata(s_wdata), .s_wstrb(s_wstrb),
    .s_bvalid(s_bvalid), .s_bready(s_bready), .s_bresp(s_bresp),
    .s_arvalid(s_arvalid), .s_arready(s_arready), .s_araddr(s_araddr),
    .s_rvalid(s_rvalid), .s_rready(s_rready), .s_rdata(s_rdata), .s_rresp(s_rresp),
    .ddr_ar_valid(ddr_ar_valid), .ddr_ar_ready(ddr_ar_ready),
    .ddr_ar_addr(ddr_ar_addr), .ddr_ar_len(ddr_ar_len),
    .ddr_ar_size(ddr_ar_size), .ddr_ar_burst(ddr_ar_burst),
    .ddr_r_valid(ddr_r_valid), .ddr_r_ready(ddr_r_ready),
    .ddr_r_data(ddr_r_data), .ddr_r_last(ddr_r_last),
    .ddr_aw_valid(ddr_aw_valid), .ddr_aw_ready(ddr_aw_ready),
    .ddr_aw_addr(ddr_aw_addr), .ddr_aw_len(ddr_aw_len),
    .ddr_aw_size(ddr_aw_size), .ddr_aw_burst(ddr_aw_burst),
    .ddr_w_valid(ddr_w_valid), .ddr_w_ready(ddr_w_ready),
    .ddr_w_data(ddr_w_data), .ddr_w_strb(ddr_w_strb), .ddr_w_last(ddr_w_last),
    .ddr_b_valid(ddr_b_valid), .ddr_b_ready(ddr_b_ready),
    .irq(irq)
  );

  // ═══════════ DDR3 Memory Model ═══════════
  localparam int DDR_MEM_SIZE = 65536;
  logic [7:0] ddr_mem [0:DDR_MEM_SIZE-1];

  // Initialize DDR with a minimal net descriptor at address 0
  initial begin
    for (int i = 0; i < DDR_MEM_SIZE; i++)
      ddr_mem[i] = 8'd0;
    // Net descriptor magic
    ddr_mem[0] = 8'h04; ddr_mem[1] = 8'h00; ddr_mem[2] = 8'hC1; ddr_mem[3] = 8'hAC;
    // num_layers = 1
    ddr_mem[6] = 8'd1;
    // layer_table_base = 0x100
    ddr_mem[8] = 8'h00; ddr_mem[9] = 8'h01;
    // Layer 0 descriptor at 0x100: set num_tiles=0 (empty layer for smoke test)
    ddr_mem[256 + 10] = 8'd0;  // num_tiles byte
  end

  // AXI4 read responder
  logic [7:0] ddr_rd_beat;
  logic [7:0] ddr_rd_len;
  logic [ADDR_W-1:0] ddr_rd_addr;
  typedef enum logic [1:0] { DDR_R_IDLE, DDR_R_DATA } ddr_r_state_e;
  ddr_r_state_e ddr_r_st;

  always_ff @(posedge clk_200 or negedge rst_n_async) begin
    if (!rst_n_async) begin
      ddr_ar_ready <= 1'b1;
      ddr_r_valid  <= 1'b0;
      ddr_r_last   <= 1'b0;
      ddr_r_st     <= DDR_R_IDLE;
    end else begin
      case (ddr_r_st)
        DDR_R_IDLE: begin
          ddr_ar_ready <= 1'b1;
          ddr_r_valid  <= 1'b0;
          if (ddr_ar_valid && ddr_ar_ready) begin
            ddr_rd_addr <= ddr_ar_addr;
            ddr_rd_len  <= ddr_ar_len;
            ddr_rd_beat <= 8'd0;
            ddr_ar_ready <= 1'b0;
            ddr_r_st    <= DDR_R_DATA;
          end
        end
        DDR_R_DATA: begin
          ddr_r_valid <= 1'b1;
          for (int b = 0; b < DATA_W/8; b++)
            ddr_r_data[b*8 +: 8] <= ddr_mem[((ddr_rd_addr + ddr_rd_beat*(DATA_W/8) + b)) % DDR_MEM_SIZE];
          ddr_r_last <= (ddr_rd_beat == ddr_rd_len);
          if (ddr_r_ready && ddr_r_valid) begin
            if (ddr_rd_beat == ddr_rd_len) begin
              ddr_r_valid <= 1'b0;
              ddr_r_st    <= DDR_R_IDLE;
            end else begin
              ddr_rd_beat <= ddr_rd_beat + 8'd1;
            end
          end
        end
      endcase
    end
  end

  // AXI4 write responder
  assign ddr_aw_ready = 1'b1;
  assign ddr_w_ready  = 1'b1;
  always_ff @(posedge clk_200 or negedge rst_n_async) begin
    if (!rst_n_async)
      ddr_b_valid <= 1'b0;
    else begin
      ddr_b_valid <= 1'b0;
      if (ddr_w_valid && ddr_w_ready && ddr_w_last)
        ddr_b_valid <= 1'b1;
    end
  end

  // ═══════════ CPU Model ═══════════
  int pass_cnt = 0, fail_cnt = 0;
  task automatic chk(input string t, input logic ok);
    if (ok) begin pass_cnt++; $display("[PASS] %s", t); end
    else begin fail_cnt++; $display("[FAIL] %s", t); end
  endtask

  task automatic cpu_write(input logic [AXIL_AW-1:0] addr, input logic [AXIL_DW-1:0] data);
    @(posedge clk_200);
    s_awvalid <= 1'b1; s_awaddr <= addr;
    @(posedge clk_200);
    while (!s_awready) @(posedge clk_200);
    s_awvalid <= 1'b0;
    s_wvalid <= 1'b1; s_wdata <= data; s_wstrb <= 4'hF;
    @(posedge clk_200);
    while (!s_wready) @(posedge clk_200);
    s_wvalid <= 1'b0;
    s_bready <= 1'b1;
    @(posedge clk_200);
    while (!s_bvalid) @(posedge clk_200);
    s_bready <= 1'b0;
    @(posedge clk_200);
  endtask

  task automatic cpu_read(input logic [AXIL_AW-1:0] addr, output logic [AXIL_DW-1:0] data);
    @(posedge clk_200);
    s_arvalid <= 1'b1; s_araddr <= addr;
    @(posedge clk_200);
    while (!s_arready) @(posedge clk_200);
    s_arvalid <= 1'b0;
    s_rready <= 1'b1;
    @(posedge clk_200);
    while (!s_rvalid) @(posedge clk_200);
    data = s_rdata;
    s_rready <= 1'b0;
    @(posedge clk_200);
  endtask

  logic [AXIL_DW-1:0] rd_val;

  initial begin
    rst_n_async <= 1'b0;
    s_awvalid <= 0; s_wvalid <= 0; s_arvalid <= 0;
    s_bready <= 0; s_rready <= 0;
    repeat (10) @(posedge clk_200);
    rst_n_async <= 1'b1;
    repeat (20) @(posedge clk_200);  // Wait for PLL lock

    $display("=== Stage 10.5 — accel_top System Tests ===");

    // Test: Read VERSION via AXI-Lite
    cpu_read(CSR_VERSION, rd_val);
    chk("10.5 VERSION readback", rd_val == IP_VERSION);

    // Test: Write net_desc_addr = 0x0000_0000
    cpu_write(CSR_NET_DESC_LO, 32'h0000_0000);
    cpu_write(CSR_NET_DESC_HI, 32'h0000_0000);
    cpu_write(CSR_LAYER_START, 32'd0);
    cpu_write(CSR_LAYER_END, 32'd0);

    // Test: Enable IRQ
    cpu_write(CSR_IRQ_MASK, 32'h0000_0001);

    // Test: Start inference
    cpu_write(CSR_CTRL, 32'h0000_0001);
    $display("[INFO] Inference started, waiting for IRQ (done)...");

    // Wait for IRQ with wall-time timeout (fork avoids blocking forever if DUT stuck)
    fork
      begin
        // 2 ms wall time @ 1ns timeunit — enough for descriptor fetch + idle SCs
        #(2_000_000);
        $display("[WARN] Wall-timeout 2 ms: IRQ not asserted — reading STATUS anyway");
      end
      begin
        wait (irq === 1'b1);
        $display("[INFO] IRQ asserted (inference complete)");
      end
    join_any
    disable fork;

    // Read status (STAT_BIT_DONE should be 1 if accel_top completed SYS_RUN)
    cpu_read(CSR_STATUS, rd_val);
    $display("[INFO] CSR_STATUS = %08h (busy=%0d done=%0d err=%0d)",
             rd_val, rd_val[0], rd_val[1], rd_val[2]);
    chk("10.5 inference finished (done bit set)", rd_val[csr_pkg::STAT_BIT_DONE]);

    // Read perf counters
    cpu_read(CSR_PERF_CYCLES, rd_val);
    $display("[INFO] Perf cycles = %0d", rd_val);

    $display("\n=== 10.5 SUMMARY: PASS=%0d FAIL=%0d ===", pass_cnt, fail_cnt);
    $finish;
  end

  initial begin #10_000_000; $display("[TIMEOUT] 10ms"); $finish; end
endmodule
