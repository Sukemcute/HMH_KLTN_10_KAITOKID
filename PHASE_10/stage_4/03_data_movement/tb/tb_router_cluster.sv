`timescale 1ns/1ps
// ============================================================================
// Testbench: router_cluster
// Verifies RIN (activation), RWT (weight), RPS (psum->output), and bypass
// routing paths.
// ============================================================================
module tb_router_cluster;
  import accel_pkg::*;
  import desc_pkg::*;

  // ---------- Parameters ----------
  localparam int LANES  = 32;
  localparam int CLK_HP = 5;

  // ---------- Signals ----------
  logic                     clk, rst_n;
  router_profile_t          cfg_profile;
  pe_mode_e                 cfg_mode;

  // RIN
  logic [LANES*8-1:0]      bank_input_rd [3];
  logic signed [7:0]       pe_act [3][LANES];

  // RWT
  logic [LANES*8-1:0]      bank_weight_rd [3];
  logic signed [7:0]       pe_wgt [3][LANES];

  // RPS
  logic signed [31:0]      pe_psum [4][LANES];
  logic                    psum_valid;
  logic [LANES*32-1:0]     bank_output_wr [4];
  logic                    bank_output_wr_en [4];

  // Bypass
  logic                    bypass_en;
  logic [LANES*8-1:0]     bypass_data_in;
  logic [LANES*8-1:0]     bypass_data_out;
  logic                    bypass_valid;

  // ---------- DUT ----------
  router_cluster #(.LANES(LANES)) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .cfg_profile      (cfg_profile),
    .cfg_mode         (cfg_mode),
    .bank_input_rd    (bank_input_rd),
    .pe_act           (pe_act),
    .bank_weight_rd   (bank_weight_rd),
    .pe_wgt           (pe_wgt),
    .pe_psum          (pe_psum),
    .psum_valid       (psum_valid),
    .bank_output_wr   (bank_output_wr),
    .bank_output_wr_en(bank_output_wr_en),
    .bypass_en        (bypass_en),
    .bypass_data_in   (bypass_data_in),
    .bypass_data_out  (bypass_data_out),
    .bypass_valid     (bypass_valid)
  );

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

  // ---------- Helper: fill bank_input_rd with known pattern ----------
  // bank b, lane l -> value = (b+1)*100 + l (truncated to 8 bits)
  task automatic fill_input_banks();
    for (int b = 0; b < 3; b++)
      for (int l = 0; l < LANES; l++)
        bank_input_rd[b][l*8 +: 8] = 8'(((b + 1) * 100 + l) % 256);
  endtask

  // ---------- Helper: fill bank_weight_rd with known pattern ----------
  // bank b, lane l -> value = (b+1)*50 + l
  task automatic fill_weight_banks();
    for (int b = 0; b < 3; b++)
      for (int l = 0; l < LANES; l++)
        bank_weight_rd[b][l*8 +: 8] = 8'(((b + 1) * 50 + l) % 256);
  endtask

  // ---------- Stimulus ----------
  initial begin
    $display("============================================================");
    $display("  TB: router_cluster");
    $display("============================================================");

    rst_n        = 0;
    cfg_mode     = PE_RS3;
    cfg_profile  = '0;
    psum_valid   = 0;
    bypass_en    = 0;
    bypass_data_in = '0;
    for (int b = 0; b < 3; b++) begin
      bank_input_rd[b]  = '0;
      bank_weight_rd[b] = '0;
    end
    for (int c = 0; c < 4; c++)
      for (int l = 0; l < LANES; l++)
        pe_psum[c][l] = '0;

    repeat (4) @(posedge clk);
    rst_n = 1;
    repeat (2) @(posedge clk);

    // ====== T1: RIN straight routing: rin_src[row]=row ======
    $display("\n--- T1: RIN straight routing ---");
    begin
      fill_input_banks();

      // Configure: rin_src[0]=0, rin_src[1]=1, rin_src[2]=2
      cfg_profile.rin_src[0] = 3'd0;
      cfg_profile.rin_src[1] = 3'd1;
      cfg_profile.rin_src[2] = 3'd2;
      @(posedge clk);
      #1;  // combinational settle

      for (int row = 0; row < 3; row++) begin
        for (int l = 0; l < LANES; l++) begin
          logic signed [7:0] exp_val;
          exp_val = $signed(bank_input_rd[row][l*8 +: 8]);
          check($sformatf("T1 RIN row=%0d lane=%0d", row, l),
                pe_act[row][l] == exp_val);
        end
      end
      $display("  T1: RIN straight routing (src[r]=r) -> pe_act[0][0]=%0d pe_act[1][0]=%0d pe_act[2][0]=%0d",
               pe_act[0][0], pe_act[1][0], pe_act[2][0]);
      $display("      Expected: %0d %0d %0d",
               $signed(bank_input_rd[0][7:0]),
               $signed(bank_input_rd[1][7:0]),
               $signed(bank_input_rd[2][7:0]));
    end

    // ====== T2: RIN cross-routing: rin_src[0]=2, src[1]=0, src[2]=1 ======
    $display("\n--- T2: RIN cross-routing ---");
    begin
      fill_input_banks();

      cfg_profile.rin_src[0] = 3'd2;
      cfg_profile.rin_src[1] = 3'd0;
      cfg_profile.rin_src[2] = 3'd1;
      @(posedge clk);
      #1;

      // pe_act[0] should get bank_input_rd[2]
      for (int l = 0; l < LANES; l++) begin
        check($sformatf("T2 row0 from bank2 lane=%0d", l),
              pe_act[0][l] == $signed(bank_input_rd[2][l*8 +: 8]));
      end
      // pe_act[1] should get bank_input_rd[0]
      for (int l = 0; l < LANES; l++) begin
        check($sformatf("T2 row1 from bank0 lane=%0d", l),
              pe_act[1][l] == $signed(bank_input_rd[0][l*8 +: 8]));
      end
      // pe_act[2] should get bank_input_rd[1]
      for (int l = 0; l < LANES; l++) begin
        check($sformatf("T2 row2 from bank1 lane=%0d", l),
              pe_act[2][l] == $signed(bank_input_rd[1][l*8 +: 8]));
      end
      $display("  T2: Cross-routing: pe_act[0][0]=%0d (from bank2=%0d) pe_act[1][0]=%0d (from bank0=%0d)",
               pe_act[0][0], $signed(bank_input_rd[2][7:0]),
               pe_act[1][0], $signed(bank_input_rd[0][7:0]));
    end

    // ====== T3: RWT routing ======
    $display("\n--- T3: RWT routing ---");
    begin
      fill_weight_banks();

      // Straight: rwt_src[0]=0, rwt_src[1]=1, rwt_src[2]=2
      cfg_profile.rwt_src[0] = 3'd0;
      cfg_profile.rwt_src[1] = 3'd1;
      cfg_profile.rwt_src[2] = 3'd2;
      @(posedge clk);
      #1;

      for (int row = 0; row < 3; row++) begin
        for (int l = 0; l < LANES; l++) begin
          logic signed [7:0] exp_val;
          exp_val = $signed(bank_weight_rd[row][l*8 +: 8]);
          check($sformatf("T3 RWT straight row=%0d lane=%0d", row, l),
                pe_wgt[row][l] == exp_val);
        end
      end
      $display("  T3a: RWT straight: pe_wgt[0][0]=%0d pe_wgt[1][0]=%0d pe_wgt[2][0]=%0d",
               pe_wgt[0][0], pe_wgt[1][0], pe_wgt[2][0]);

      // Cross: rwt_src[0]=1, rwt_src[1]=2, rwt_src[2]=0
      cfg_profile.rwt_src[0] = 3'd1;
      cfg_profile.rwt_src[1] = 3'd2;
      cfg_profile.rwt_src[2] = 3'd0;
      @(posedge clk);
      #1;

      for (int l = 0; l < LANES; l++) begin
        check($sformatf("T3 cross row0 lane=%0d", l),
              pe_wgt[0][l] == $signed(bank_weight_rd[1][l*8 +: 8]));
        check($sformatf("T3 cross row1 lane=%0d", l),
              pe_wgt[1][l] == $signed(bank_weight_rd[2][l*8 +: 8]));
        check($sformatf("T3 cross row2 lane=%0d", l),
              pe_wgt[2][l] == $signed(bank_weight_rd[0][l*8 +: 8]));
      end
      $display("  T3b: RWT cross: pe_wgt[0][0]=%0d (from bank1=%0d)",
               pe_wgt[0][0], $signed(bank_weight_rd[1][7:0]));
    end

    // ====== T4: RPS packing ======
    $display("\n--- T4: RPS psum packing ---");
    begin
      // Fill pe_psum with known values: pe_psum[col][lane] = col*1000 + lane
      for (int c = 0; c < 4; c++)
        for (int l = 0; l < LANES; l++)
          pe_psum[c][l] = 32'(c * 1000 + l);

      psum_valid = 1'b1;
      @(posedge clk);
      #1;

      // Verify packing: bank_output_wr[col][lane*32 +: 32] = pe_psum[col][lane]
      for (int c = 0; c < 4; c++) begin
        check($sformatf("T4 wr_en[%0d]", c), bank_output_wr_en[c] == 1'b1);
        for (int l = 0; l < LANES; l++) begin
          logic [31:0] packed_val;
          packed_val = bank_output_wr[c][l*32 +: 32];
          check($sformatf("T4 col=%0d lane=%0d", c, l),
                $signed(packed_val) == pe_psum[c][l]);
        end
      end
      $display("  T4: bank_output_wr[0][0]=%0d (exp %0d) bank_output_wr[3][31]=%0d (exp %0d)",
               $signed(bank_output_wr[0][31:0]), pe_psum[0][0],
               $signed(bank_output_wr[3][31*32 +: 32]), pe_psum[3][31]);

      // Deassert psum_valid -> wr_en should go low
      psum_valid = 1'b0;
      @(posedge clk);
      #1;
      for (int c = 0; c < 4; c++)
        check($sformatf("T4 wr_en[%0d] deasserted", c), bank_output_wr_en[c] == 1'b0);
      $display("  T4: psum_valid=0 -> all wr_en deasserted");
    end

    // ====== T5: Bypass path ======
    $display("\n--- T5: Bypass path ---");
    begin
      // Fill bypass data
      for (int l = 0; l < LANES; l++)
        bypass_data_in[l*8 +: 8] = 8'(l + 42);

      bypass_en = 1'b0;
      @(posedge clk);
      #1;
      check("T5 bypass disabled -> out=0", bypass_data_out == '0);
      check("T5 bypass disabled -> valid=0", bypass_valid == 1'b0);
      $display("  T5: bypass_en=0 -> data_out=0x%0h valid=%0b", bypass_data_out[31:0], bypass_valid);

      bypass_en = 1'b1;
      @(posedge clk);
      #1;
      check("T5 bypass enabled -> valid=1", bypass_valid == 1'b1);
      check("T5 bypass data pass-through", bypass_data_out == bypass_data_in);
      $display("  T5: bypass_en=1 -> data passes through, valid=%0b", bypass_valid);

      // Spot-check a few lanes
      for (int l = 0; l < LANES; l++) begin
        check($sformatf("T5 bypass lane=%0d", l),
              bypass_data_out[l*8 +: 8] == bypass_data_in[l*8 +: 8]);
      end

      bypass_en = 1'b0;
      @(posedge clk);
    end

    // ====== Summary ======
    repeat (4) @(posedge clk);
    $display("\n============================================================");
    $display("  router_cluster: %0d PASSED, %0d FAILED", pass_cnt, fail_cnt);
    if (fail_cnt == 0)
      $display("  >>> ALL TESTS PASSED <<<");
    else
      $display("  >>> SOME TESTS FAILED <<<");
    $display("============================================================");
    $finish;
  end

endmodule
