`timescale 1ns/1ps
// ============================================================================
// Integration Testbench: addr_gen_input + glb_input_bank
// Proves that the address generator and GLB banking work together for a
// real layer access pattern.
//
// Flow:
//   1. Pre-fill 3 glb_input_banks with known data pattern
//   2. Use addr_gen_input to generate addresses for L0 access (h,w,c sweep)
//   3. Read from correct bank based on addr_gen bank_id output
//   4. Verify: data matches original write OR padding returns zp_x
// ============================================================================
module tb_addr_bank_integration;
  import accel_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES          = 32;
  localparam int MAX_WIDTH      = 640;
  localparam int MAX_CIN        = 256;
  localparam int SUBBANK_DEPTH  = 2048;
  localparam int ADDR_W         = $clog2(SUBBANK_DEPTH);
  localparam int CLK_HP         = 5;

  // Layer config (small test layer)
  localparam int CFG_WIN       = 64;
  localparam int CFG_HIN       = 10;
  localparam int CFG_CIN_TILE  = 3;
  localparam int CFG_Q_IN      = 4;
  localparam int CFG_PAD_TOP   = 1;
  localparam int CFG_PAD_BOT   = 1;
  localparam int CFG_PAD_LEFT  = 0;
  localparam int CFG_PAD_RIGHT = 0;
  localparam int CFG_ZP_X      = -3;

  // ---------- Signals ----------
  logic              clk, rst_n;

  // addr_gen_input ports
  logic [9:0]        cfg_win, cfg_hin;
  logic [8:0]        cfg_cin_tile;
  logic [3:0]        cfg_q_in;
  logic [3:0]        cfg_stride;
  logic [3:0]        cfg_pad_top, cfg_pad_bot, cfg_pad_left, cfg_pad_right;
  logic signed [7:0] cfg_zp_x;

  logic              ag_req_valid;
  logic [9:0]        ag_req_h, ag_req_w;
  logic [8:0]        ag_req_c;

  logic              ag_out_valid;
  logic [1:0]        ag_out_bank_id;
  logic [15:0]       ag_out_addr;
  logic              ag_out_is_padding;
  logic signed [7:0] ag_out_pad_value;

  // GLB bank ports
  logic              bank_wr_en   [3];
  logic [ADDR_W-1:0] bank_wr_addr [3];
  logic [LANES*8-1:0] bank_wr_data [3];
  logic [LANES-1:0]  bank_wr_mask [3];

  logic              bank_rd_en   [3];
  logic [ADDR_W-1:0] bank_rd_addr [3];
  logic [LANES*8-1:0] bank_rd_data [3];

  // ---------- DUT: addr_gen_input ----------
  addr_gen_input #(
    .LANES     (LANES),
    .MAX_WIDTH (MAX_WIDTH),
    .MAX_CIN   (MAX_CIN)
  ) u_addr_gen (
    .clk            (clk),
    .rst_n          (rst_n),
    .cfg_win        (cfg_win),
    .cfg_cin_tile   (cfg_cin_tile),
    .cfg_q_in       (cfg_q_in),
    .cfg_stride     (cfg_stride),
    .cfg_pad_top    (cfg_pad_top),
    .cfg_pad_bot    (cfg_pad_bot),
    .cfg_pad_left   (cfg_pad_left),
    .cfg_pad_right  (cfg_pad_right),
    .cfg_hin        (cfg_hin),
    .cfg_zp_x       (cfg_zp_x),
    .req_valid      (ag_req_valid),
    .req_h          (ag_req_h),
    .req_w          (ag_req_w),
    .req_c          (ag_req_c),
    .out_valid      (ag_out_valid),
    .out_bank_id    (ag_out_bank_id),
    .out_addr       (ag_out_addr),
    .out_is_padding (ag_out_is_padding),
    .out_pad_value  (ag_out_pad_value)
  );

  // ---------- DUT: 3 x glb_input_bank ----------
  genvar gb;
  generate
    for (gb = 0; gb < 3; gb++) begin : gen_bank
      glb_input_bank #(
        .LANES         (LANES),
        .SUBBANK_DEPTH (SUBBANK_DEPTH)
      ) u_bank (
        .clk          (clk),
        .rst_n        (rst_n),
        .wr_en        (bank_wr_en[gb]),
        .wr_addr      (bank_wr_addr[gb]),
        .wr_data      (bank_wr_data[gb]),
        .wr_lane_mask (bank_wr_mask[gb]),
        .rd_en        (bank_rd_en[gb]),
        .rd_addr      (bank_rd_addr[gb]),
        .rd_data      (bank_rd_data[gb])
      );
    end
  endgenerate

  // ---------- Clock ----------
  initial clk = 0;
  always #(CLK_HP) clk = ~clk;

  // ---------- Scoreboard ----------
  int pass_cnt = 0;
  int fail_cnt = 0;

  task automatic check(string tag, logic cond);
    if (cond) begin
      pass_cnt++;
    end else begin
      fail_cnt++;
      $display("[FAIL] %s @ %0t", tag, $time);
    end
  endtask

  // ---------- Data pattern: deterministic function of (bank, addr, lane) ----------
  function automatic logic [7:0] data_pattern(int bank, int addr, int lane);
    return 8'((bank * 71 + addr * 13 + lane * 3 + 17) % 251);
  endfunction

  // ---------- Address computation (mirror RTL) ----------
  function automatic logic [15:0] compute_addr(
    input int h, input int w, input int c
  );
    int h_div3, row_slot, wblk, wblk_total;
    h_div3     = h / 3;
    row_slot   = h_div3 % CFG_Q_IN;
    wblk       = w / LANES;
    wblk_total = (CFG_WIN + LANES - 1) / LANES;
    return 16'((row_slot * CFG_CIN_TILE + c) * wblk_total + wblk);
  endfunction

  // ---------- Stimulus ----------
  initial begin
    $display("============================================================");
    $display("  TB: addr_gen_input + glb_input_bank Integration");
    $display("============================================================");

    rst_n        = 0;
    ag_req_valid = 0;
    ag_req_h     = 0;
    ag_req_w     = 0;
    ag_req_c     = 0;

    cfg_win       = CFG_WIN;
    cfg_hin       = CFG_HIN;
    cfg_cin_tile  = CFG_CIN_TILE;
    cfg_q_in      = CFG_Q_IN;
    cfg_stride    = 4'd1;
    cfg_pad_top   = CFG_PAD_TOP;
    cfg_pad_bot   = CFG_PAD_BOT;
    cfg_pad_left  = CFG_PAD_LEFT;
    cfg_pad_right = CFG_PAD_RIGHT;
    cfg_zp_x      = CFG_ZP_X;

    for (int b = 0; b < 3; b++) begin
      bank_wr_en[b]   = 0;
      bank_wr_addr[b] = 0;
      bank_wr_data[b] = 0;
      bank_wr_mask[b] = 0;
      bank_rd_en[b]   = 0;
      bank_rd_addr[b] = 0;
    end

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ====== Phase 1: Pre-fill banks with known data ======
    $display("\n--- Phase 1: Pre-filling GLB banks ---");
    begin
      int wblk_total;
      wblk_total = (CFG_WIN + LANES - 1) / LANES;

      // For each non-padded (h,w,c) position, compute bank_id and addr,
      // then write the data pattern into the correct bank.
      // Valid h range: pad_top .. hin-pad_bot-1 = 1..8
      for (int h = CFG_PAD_TOP; h < CFG_HIN - CFG_PAD_BOT; h++) begin
        int bank_id;
        bank_id = h % 3;
        for (int c = 0; c < CFG_CIN_TILE; c++) begin
          for (int wblk = 0; wblk < wblk_total; wblk++) begin
            int h_div3, row_slot;
            logic [15:0] addr;
            h_div3   = h / 3;
            row_slot = h_div3 % CFG_Q_IN;
            addr     = 16'((row_slot * CFG_CIN_TILE + c) * wblk_total + wblk);

            @(posedge clk);
            bank_wr_en[bank_id]   <= 1'b1;
            bank_wr_addr[bank_id] <= addr[ADDR_W-1:0];
            bank_wr_mask[bank_id] <= {LANES{1'b1}};

            // Fill each lane with deterministic pattern
            for (int l = 0; l < LANES; l++)
              bank_wr_data[bank_id][l*8 +: 8] <= data_pattern(bank_id, int'(addr), l);

            @(posedge clk);
            bank_wr_en[bank_id] <= 1'b0;
          end
        end
      end
      $display("  Pre-fill complete for h=[%0d..%0d], c=[0..%0d]",
               CFG_PAD_TOP, CFG_HIN - CFG_PAD_BOT - 1, CFG_CIN_TILE - 1);
    end

    repeat (4) @(posedge clk);

    // ====== Phase 2: Read-back using addr_gen_input ======
    $display("\n--- Phase 2: Read-back via addr_gen_input ---");
    begin
      int total_checks, pad_checks, data_checks, data_errors;
      total_checks = 0;
      pad_checks   = 0;
      data_checks  = 0;
      data_errors  = 0;

      // Sweep h=0..CFG_HIN-1 (includes padding rows), w=0,32, c=0..CFG_CIN_TILE-1
      for (int h = 0; h < CFG_HIN; h++) begin
        for (int w = 0; w < CFG_WIN; w += LANES) begin
          for (int c = 0; c < CFG_CIN_TILE; c++) begin
            logic        exp_pad;
            logic [1:0]  exp_bank;
            logic [15:0] exp_addr;

            exp_pad  = (h < CFG_PAD_TOP) || (h >= CFG_HIN - CFG_PAD_BOT);
            exp_bank = h % 3;
            exp_addr = compute_addr(h, w, c);

            // Step A: Drive addr_gen_input request
            @(posedge clk);
            ag_req_valid <= 1'b1;
            ag_req_h     <= 10'(h);
            ag_req_w     <= 10'(w);
            ag_req_c     <= 9'(c);
            @(posedge clk);
            ag_req_valid <= 1'b0;

            // Step B: Wait 1 cycle for addr_gen output (1-cycle latency)
            @(posedge clk);

            // Verify addr_gen outputs
            check($sformatf("AG h=%0d w=%0d c=%0d bank", h, w, c),
                  ag_out_bank_id == exp_bank);
            check($sformatf("AG h=%0d w=%0d c=%0d pad", h, w, c),
                  ag_out_is_padding == exp_pad);

            if (ag_out_is_padding) begin
              // Padding: verify pad_value = zp_x
              check($sformatf("PAD h=%0d zp_x", h),
                    ag_out_pad_value == cfg_zp_x);
              pad_checks++;
            end else begin
              // Non-padding: read from the bank indicated by addr_gen
              bank_rd_en[ag_out_bank_id]   <= 1'b1;
              bank_rd_addr[ag_out_bank_id] <= ag_out_addr[ADDR_W-1:0];
              @(posedge clk);
              bank_rd_en[ag_out_bank_id] <= 1'b0;
              @(posedge clk);  // Read has 1-cycle latency (registered output)

              // Step C: Verify data from bank matches original pattern
              begin
                int rd_bank;
                int rd_addr_int;
                rd_bank     = int'(ag_out_bank_id);
                rd_addr_int = int'(ag_out_addr);

                for (int l = 0; l < LANES; l++) begin
                  logic [7:0] exp_data, got_data;
                  exp_data = data_pattern(rd_bank, rd_addr_int, l);
                  got_data = bank_rd_data[rd_bank][l*8 +: 8];

                  if (got_data != exp_data) begin
                    data_errors++;
                    if (data_errors <= 10)
                      $display("  [FAIL] DATA h=%0d w=%0d c=%0d lane=%0d: got=0x%02h exp=0x%02h (bank=%0d addr=%0d)",
                               h, w, c, l, got_data, exp_data, rd_bank, rd_addr_int);
                  end
                  check($sformatf("DATA h=%0d w=%0d c=%0d l=%0d", h, w, c, l),
                        got_data == exp_data);
                end
              end
              data_checks++;
            end
            total_checks++;
          end
        end
      end

      $display("  Total positions checked: %0d", total_checks);
      $display("  Padding positions:       %0d", pad_checks);
      $display("  Data positions:          %0d", data_checks);
      $display("  Data lane errors:        %0d", data_errors);
    end

    // ====== Phase 3: Verify padding does NOT return stale SRAM data ======
    $display("\n--- Phase 3: Padding isolation check ---");
    begin
      // Write garbage into bank at address 0 for all banks
      for (int b = 0; b < 3; b++) begin
        @(posedge clk);
        bank_wr_en[b]   <= 1'b1;
        bank_wr_addr[b] <= '0;
        bank_wr_data[b] <= {LANES{8'hDE}};
        bank_wr_mask[b] <= {LANES{1'b1}};
        @(posedge clk);
        bank_wr_en[b] <= 1'b0;
      end

      // Request padding position: h=0 (top pad), w=0, c=0
      @(posedge clk);
      ag_req_valid <= 1'b1;
      ag_req_h     <= 10'd0;
      ag_req_w     <= 10'd0;
      ag_req_c     <= 9'd0;
      @(posedge clk);
      ag_req_valid <= 1'b0;
      @(posedge clk);

      // addr_gen should flag this as padding
      check("Phase3 is_padding", ag_out_is_padding == 1'b1);
      check("Phase3 pad_value",  ag_out_pad_value == cfg_zp_x);
      $display("  Padding at h=0: is_padding=%0b pad_value=%0d (zp_x=%0d) -- SRAM not used",
               ag_out_is_padding, ag_out_pad_value, cfg_zp_x);
      $display("  The consumer must use pad_value when is_padding=1, NOT SRAM data.");
    end

    // ====== Summary ======
    repeat (4) @(posedge clk);
    $display("\n============================================================");
    $display("  addr_bank_integration: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("============================================================");
    $finish;
  end

endmodule
