// ============================================================================
// Testbench : tb_router_cluster_v2
// Project   : YOLOv10n INT8 Accelerator — V4-VC707
// Tests     : T5.1.1 RIN multicast — bank0 → pe_act[0]=pe_act[1]=pe_act[2]
//             T5.1.2 RWT per-column weight — 4 DIFFERENT patterns per col
//             T5.1.3 Bypass path — bypass_en=1 → bypass_out=bypass_in
//             T5.1.4 Mode switching — RS3 → PASS → DW3 verify each
// ============================================================================
`timescale 1ns / 1ps

module tb_router_cluster_v2;
  import accel_pkg::*;

  // ──────────────────────────────────────────────────────────────
  //  Parameters
  // ──────────────────────────────────────────────────────────────
  localparam int LANES    = accel_pkg::LANES;    // 20
  localparam int PE_ROWS_ = accel_pkg::PE_ROWS;  // 3
  localparam int PE_COLS_ = accel_pkg::PE_COLS;  // 4
  localparam int CLK_NS   = 4;                   // 250 MHz

  // ──────────────────────────────────────────────────────────────
  //  DUT signals
  // ──────────────────────────────────────────────────────────────
  logic        clk, rst_n;
  pe_mode_e    cfg_pe_mode;

  // RIN
  int8_t       glb_in_data   [3][LANES];
  int8_t       pe_act        [PE_ROWS_][LANES];

  // RWT
  int8_t       glb_wgt_data  [3][PE_COLS_][LANES];
  int8_t       pe_wgt        [PE_ROWS_][PE_COLS_][LANES];

  // RPS
  int32_t      pe_psum_in    [PE_COLS_][LANES];
  logic        psum_valid;
  int32_t      glb_out_psum  [PE_COLS_][LANES];
  logic        glb_out_wr_en [PE_COLS_];

  // BYPASS
  int8_t       bypass_in     [LANES];
  int8_t       bypass_out    [LANES];
  logic        bypass_en;
  logic [1:0]  rin_bank_sel;

  // ──────────────────────────────────────────────────────────────
  //  DUT instantiation
  // ──────────────────────────────────────────────────────────────
  router_cluster_v2 #(
    .LANES  (LANES),
    .PE_ROWS(PE_ROWS_),
    .PE_COLS(PE_COLS_)
  ) u_dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .cfg_pe_mode   (cfg_pe_mode),
    .rin_bank_sel  (rin_bank_sel),
    .glb_in_data   (glb_in_data),
    .pe_act        (pe_act),
    .glb_wgt_data  (glb_wgt_data),
    .pe_wgt        (pe_wgt),
    .pe_psum_in    (pe_psum_in),
    .psum_valid    (psum_valid),
    .glb_out_psum  (glb_out_psum),
    .glb_out_wr_en (glb_out_wr_en),
    .bypass_in     (bypass_in),
    .bypass_out    (bypass_out),
    .bypass_en     (bypass_en)
  );

  // ──────────────────────────────────────────────────────────────
  //  Clock generation: 250 MHz (4 ns period)
  // ──────────────────────────────────────────────────────────────
  initial clk = 1'b0;
  always #(CLK_NS/2) clk = ~clk;

  // ──────────────────────────────────────────────────────────────
  //  Scoreboard
  // ──────────────────────────────────────────────────────────────
  int pass_cnt = 0;
  int fail_cnt = 0;
  int test_cnt = 0;

  task automatic check(string tag, logic cond, string msg);
    test_cnt++;
    if (cond) begin
      pass_cnt++;
    end else begin
      fail_cnt++;
      $display("[FAIL] %s : %s", tag, msg);
    end
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Helper: zero all inputs
  // ──────────────────────────────────────────────────────────────
  task automatic zero_inputs();
    for (int b = 0; b < 3; b++)
      for (int l = 0; l < LANES; l++)
        glb_in_data[b][l] = 8'sd0;

    for (int b = 0; b < 3; b++)
      for (int c = 0; c < PE_COLS_; c++)
        for (int l = 0; l < LANES; l++)
          glb_wgt_data[b][c][l] = 8'sd0;

    for (int c = 0; c < PE_COLS_; c++)
      for (int l = 0; l < LANES; l++)
        pe_psum_in[c][l] = 32'sd0;

    for (int l = 0; l < LANES; l++)
      bypass_in[l] = 8'sd0;

    psum_valid    = 1'b0;
    bypass_en     = 1'b0;
    rin_bank_sel  = 2'd0;
  endtask

  // ──────────────────────────────────────────────────────────────
  //  Main test sequence
  // ──────────────────────────────────────────────────────────────
  initial begin
    $display("===========================================================");
    $display(" tb_router_cluster_v2 — START");
    $display("===========================================================");

    // Reset
    rst_n = 1'b0;
    cfg_pe_mode = PE_RS3;
    zero_inputs();
    repeat (4) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);

    // ════════════════════════════════════════════════════════════
    //  T5.1.1: RIN multicast (RS3 mode)
    //   Set bank0 with known pattern. In RS3 mode:
    //     pe_act[row][ln] = glb_in_data[row][ln]
    //   So bank 0 → row 0, bank 1 → row 1, bank 2 → row 2
    //   Test: set all 3 banks to same pattern → all rows match
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.1.1: RIN multicast (RS3) ---");
    cfg_pe_mode = PE_RS3;
    zero_inputs();

    // Fill bank 0 with pattern 10+lane, bank 1 with 30+lane, bank 2 with 50+lane
    for (int l = 0; l < LANES; l++) begin
      glb_in_data[0][l] = 8'(10 + l);
      glb_in_data[1][l] = 8'(30 + l);
      glb_in_data[2][l] = 8'(50 + l);
    end

    @(posedge clk);
    #1;

    // In RS3: pe_act[row] = glb_in_data[row]
    // Verify row 0 = bank 0 pattern
    begin
      automatic logic all_ok = 1'b1;
      for (int l = 0; l < LANES; l++) begin
        if (pe_act[0][l] != 8'(10 + l)) all_ok = 1'b0;
        if (pe_act[1][l] != 8'(30 + l)) all_ok = 1'b0;
        if (pe_act[2][l] != 8'(50 + l)) all_ok = 1'b0;
      end
      check("T5.1.1_rs3", all_ok,
            "RS3: pe_act[row] should match glb_in_data[row]");
    end

    // OS1 mode: all rows get bank 0
    cfg_pe_mode = PE_OS1;
    @(posedge clk);
    #1;

    begin
      automatic logic all_ok = 1'b1;
      for (int row = 0; row < PE_ROWS_; row++)
        for (int l = 0; l < LANES; l++)
          if (pe_act[row][l] != 8'(10 + l)) all_ok = 1'b0;
      check("T5.1.1_os1", all_ok,
            "OS1: all 3 rows should see bank0 data (multicast)");
    end

    // ════════════════════════════════════════════════════════════
    //  T5.1.2: RWT per-column weight
    //   Set 4 DIFFERENT weight patterns per column.
    //   Verify pe_wgt[row][col] differ per col.
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.1.2: RWT per-column weight ---");
    cfg_pe_mode = PE_RS3;
    zero_inputs();

    // bank=row, each col gets unique pattern
    for (int bank = 0; bank < 3; bank++)
      for (int col = 0; col < PE_COLS_; col++)
        for (int l = 0; l < LANES; l++)
          glb_wgt_data[bank][col][l] = 8'(bank * 40 + col * 10 + l);

    @(posedge clk);
    #1;

    // Verify each row/col receives correct data
    begin
      automatic logic all_ok = 1'b1;
      for (int row = 0; row < PE_ROWS_; row++)
        for (int col = 0; col < PE_COLS_; col++)
          for (int l = 0; l < LANES; l++)
            if (pe_wgt[row][col][l] != 8'(row * 40 + col * 10 + l))
              all_ok = 1'b0;
      check("T5.1.2_match", all_ok,
            "pe_wgt[row][col] should match glb_wgt_data[row][col]");
    end

    // ★ Verify columns are DIFFERENT
    begin
      automatic logic cols_differ = 1'b1;
      for (int row = 0; row < PE_ROWS_; row++) begin
        for (int c1 = 0; c1 < PE_COLS_; c1++) begin
          for (int c2 = c1 + 1; c2 < PE_COLS_; c2++) begin
            automatic logic same = 1'b1;
            for (int l = 0; l < LANES; l++)
              if (pe_wgt[row][c1][l] != pe_wgt[row][c2][l]) same = 1'b0;
            if (same) cols_differ = 1'b0;
          end
        end
      end
      check("T5.1.2_diff", cols_differ,
            "All columns must carry DIFFERENT weight data");
    end

    // Verify PE_PASS mode zeros weights
    cfg_pe_mode = PE_PASS;
    @(posedge clk);
    #1;
    begin
      automatic logic all_zero = 1'b1;
      for (int r = 0; r < PE_ROWS_; r++)
        for (int c = 0; c < PE_COLS_; c++)
          for (int l = 0; l < LANES; l++)
            if (pe_wgt[r][c][l] != 8'sd0) all_zero = 1'b0;
      check("T5.1.2_pass_zero", all_zero,
            "PE_PASS: weights should all be zero");
    end

    // ════════════════════════════════════════════════════════════
    //  T5.1.3: Bypass path
    //   bypass_en=1 → bypass_out = bypass_in
    //   bypass_en=0 → bypass_out = 0
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.1.3: Bypass path ---");
    cfg_pe_mode = PE_PASS;
    zero_inputs();

    // Load bypass pattern
    for (int l = 0; l < LANES; l++)
      bypass_in[l] = 8'(l * 3 + 7);

    // bypass_en = 1
    bypass_en = 1'b1;
    @(posedge clk);
    #1;

    begin
      automatic logic all_ok = 1'b1;
      for (int l = 0; l < LANES; l++)
        if (bypass_out[l] != 8'(l * 3 + 7)) all_ok = 1'b0;
      check("T5.1.3_en", all_ok,
            "bypass_en=1: bypass_out should equal bypass_in");
    end

    // bypass_en = 0 → output must be zero
    bypass_en = 1'b0;
    @(posedge clk);
    #1;

    begin
      automatic logic all_ok = 1'b1;
      for (int l = 0; l < LANES; l++)
        if (bypass_out[l] != 8'sd0) all_ok = 1'b0;
      check("T5.1.3_dis", all_ok,
            "bypass_en=0: bypass_out should be zero");
    end

    // ════════════════════════════════════════════════════════════
    //  T5.1.4: Mode switching — RS3 → PASS → DW3
    //   Verify correct routing at each mode transition.
    // ════════════════════════════════════════════════════════════
    $display("\n--- T5.1.4: Mode switching ---");
    zero_inputs();

    // Set up known data for all paths
    for (int b = 0; b < 3; b++)
      for (int l = 0; l < LANES; l++)
        glb_in_data[b][l] = 8'(b * 20 + l);

    for (int b = 0; b < 3; b++)
      for (int c = 0; c < PE_COLS_; c++)
        for (int l = 0; l < LANES; l++)
          glb_wgt_data[b][c][l] = 8'(100 + b * 30 + c * 5 + l);

    for (int l = 0; l < LANES; l++)
      bypass_in[l] = 8'(l + 50);
    bypass_en = 1'b1;

    // --- RS3 ---
    cfg_pe_mode = PE_RS3;
    @(posedge clk);
    #1;

    // RS3: pe_act[row]=glb_in_data[row], weights routed, bypass=passthrough
    check("T5.1.4_rs3_act", pe_act[0][0] == 8'(0) && pe_act[1][0] == 8'(20),
          $sformatf("RS3 act: row0[0]=%0d row1[0]=%0d", pe_act[0][0], pe_act[1][0]));
    check("T5.1.4_rs3_wgt", pe_wgt[0][0][0] == 8'(100) && pe_wgt[0][1][0] == 8'(105),
          $sformatf("RS3 wgt: r0c0=%0d r0c1=%0d", pe_wgt[0][0][0], pe_wgt[0][1][0]));
    check("T5.1.4_rs3_byp", bypass_out[0] == 8'(50),
          $sformatf("RS3 bypass: out[0]=%0d", bypass_out[0]));

    // --- PASS ---
    cfg_pe_mode = PE_PASS;
    @(posedge clk);
    #1;

    // PASS: pe_act = 0, pe_wgt = 0, bypass = bypass_in
    check("T5.1.4_pass_act", pe_act[0][0] == 8'sd0,
          $sformatf("PASS act: should be 0, got %0d", pe_act[0][0]));
    check("T5.1.4_pass_wgt", pe_wgt[0][0][0] == 8'sd0,
          $sformatf("PASS wgt: should be 0, got %0d", pe_wgt[0][0][0]));
    check("T5.1.4_pass_byp", bypass_out[5] == 8'(55),
          $sformatf("PASS bypass: out[5]=%0d expected=55", bypass_out[5]));

    // --- DW3 ---
    cfg_pe_mode = PE_DW3;
    @(posedge clk);
    #1;

    // DW3: same as RS3 for act routing (row-matched), weights routed
    check("T5.1.4_dw3_act", pe_act[0][0] == glb_in_data[0][0],
          $sformatf("DW3 act: row0[0]=%0d expected=%0d",
                    pe_act[0][0], glb_in_data[0][0]));
    check("T5.1.4_dw3_wgt", pe_wgt[1][2][0] == glb_wgt_data[1][2][0],
          $sformatf("DW3 wgt: r1c2[0]=%0d expected=%0d",
                    pe_wgt[1][2][0], glb_wgt_data[1][2][0]));

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    $display("\n===========================================================");
    $display(" tb_router_cluster_v2 — RESULTS");
    $display("   Total : %0d", test_cnt);
    $display("   PASS  : %0d", pass_cnt);
    $display("   FAIL  : %0d", fail_cnt);
    if (fail_cnt == 0)
      $display("   >>> ALL TESTS PASSED <<<");
    else
      $display("   >>> SOME TESTS FAILED <<<");
    $display("===========================================================");
    $finish;
  end

endmodule
