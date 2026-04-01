// ============================================================================
// Testbench: tb_pe_unit
// Project:   YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Verification of pe_unit with LANES=20, 10 dsp_pairs.
//   Tests: PE_RS3, PE_OS1, PE_DW3 modes, enable gating, clear.
//   PASS CRITERIA: 0 errors across all modes and all 20 lanes.
// ============================================================================
`timescale 1ns / 1ps

module tb_pe_unit;
  import accel_pkg::*;

  localparam real CLK_PERIOD = 4.0;
  localparam int  L = LANES;          // 20
  localparam int  PIPE = DSP_PIPE_DEPTH;  // 5

  logic              clk   = 1'b0;
  logic              rst_n = 1'b0;
  pe_mode_e          pe_mode;
  logic signed [7:0] x_in  [L];
  logic signed [7:0] w_in  [L];
  logic              pe_enable;
  logic              clear_acc;
  logic signed [31:0] psum_out [L];
  logic              psum_valid;

  always #(CLK_PERIOD / 2.0) clk = ~clk;

  // ─────────────────────────────────────────────────────────────
  //  DUT
  // ─────────────────────────────────────────────────────────────
  pe_unit #(.LANES(L)) u_dut (
    .clk       (clk),
    .rst_n     (rst_n),
    .pe_mode   (pe_mode),
    .x_in      (x_in),
    .w_in      (w_in),
    .pe_enable (pe_enable),
    .clear_acc (clear_acc),
    .psum_out  (psum_out),
    .psum_valid(psum_valid)
  );

  integer total_tests = 0, total_errors = 0;

  task automatic do_reset();
    rst_n     = 1'b0;
    pe_enable = 1'b0;
    clear_acc = 1'b0;
    pe_mode   = PE_RS3;
    for (int l = 0; l < L; l++) begin x_in[l] = 0; w_in[l] = 0; end
    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    @(posedge clk);
  endtask

  task automatic wait_pipe();
    repeat (PIPE + 2) @(posedge clk);
  endtask

  // ─────────────────────────────────────────────────────────────
  //  Main
  // ─────────────────────────────────────────────────────────────
  initial begin
    $display("══════════════════════════════════════════════════════════");
    $display(" TB: pe_unit — V4 (LANES=%0d, 5-stage, 250 MHz)", L);
    $display("══════════════════════════════════════════════════════════");

    // ══════════════════════════════════════════════════════════
    //  TEST 1: PE_RS3 — 9-cycle accumulation, per-lane weight
    // ══════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] golden [L];
      logic signed [7:0]  xa_seq [9][L], w_seq [9][L];
      $display("\n── TEST 1: PE_RS3 (9-cycle, per-lane) ──");
      do_reset();
      pe_mode = PE_RS3;

      // Generate random test data
      for (int k = 0; k < 9; k++)
        for (int l = 0; l < L; l++) begin
          xa_seq[k][l] = $random;
          w_seq[k][l]  = $random;
        end

      // Golden computation
      for (int l = 0; l < L; l++) begin
        golden[l] = 0;
        for (int k = 0; k < 9; k++)
          golden[l] += 32'(xa_seq[k][l]) * 32'(w_seq[k][l]);
      end

      // Drive 9 beats
      for (int k = 0; k < 9; k++) begin
        @(posedge clk);
        pe_enable <= 1'b1;
        clear_acc <= (k == 0);
        for (int l = 0; l < L; l++) begin
          x_in[l] <= xa_seq[k][l];
          w_in[l] <= w_seq[k][l];
        end
      end
      @(posedge clk);
      pe_enable <= 1'b0; clear_acc <= 1'b0;
      for (int l = 0; l < L; l++) begin x_in[l] <= 0; w_in[l] <= 0; end
      wait_pipe();

      // Compare all 20 lanes
      for (int l = 0; l < L; l++) begin
        if (psum_out[l] !== golden[l]) begin
          $display("  ERR lane %0d: got=%0d exp=%0d", l, psum_out[l], golden[l]);
          t_err++;
        end
      end

      total_tests++; total_errors += t_err;
      $display("  TEST 1 %s (%0d lane errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ══════════════════════════════════════════════════════════
    //  TEST 2: PE_OS1 — broadcast weight, 20-cycle accumulation
    // ══════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] golden [L];
      logic signed [7:0]  xa_v [20][L], w_scalar [20];
      $display("\n── TEST 2: PE_OS1 (broadcast weight) ──");
      do_reset();
      pe_mode = PE_OS1;

      for (int k = 0; k < 20; k++) begin
        w_scalar[k] = $random;
        for (int l = 0; l < L; l++)
          xa_v[k][l] = $random;
      end

      // Golden: each lane accumulates x[l][k] * w_scalar[k]
      for (int l = 0; l < L; l++) begin
        golden[l] = 0;
        for (int k = 0; k < 20; k++)
          golden[l] += 32'(xa_v[k][l]) * 32'(w_scalar[k]);
      end

      // Drive 20 beats with broadcast weight
      for (int k = 0; k < 20; k++) begin
        @(posedge clk);
        pe_enable <= 1'b1;
        clear_acc <= (k == 0);
        for (int l = 0; l < L; l++) begin
          x_in[l] <= xa_v[k][l];
          w_in[l] <= w_scalar[k];  // Same weight for all lanes
        end
      end
      @(posedge clk);
      pe_enable <= 1'b0; clear_acc <= 1'b0;
      for (int l = 0; l < L; l++) begin x_in[l] <= 0; w_in[l] <= 0; end
      wait_pipe();

      for (int l = 0; l < L; l++) begin
        if (psum_out[l] !== golden[l]) begin
          $display("  ERR lane %0d: got=%0d exp=%0d", l, psum_out[l], golden[l]);
          t_err++;
        end
      end

      total_tests++; total_errors += t_err;
      $display("  TEST 2 %s (%0d lane errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ══════════════════════════════════════════════════════════
    //  TEST 3: PE_DW3 — per-channel, 3-cycle (kw=0,1,2)
    // ══════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] golden [L];
      logic signed [7:0]  xa_v [3][L], w_v [3][L];
      $display("\n── TEST 3: PE_DW3 (per-channel, 3 cycles) ──");
      do_reset();
      pe_mode = PE_DW3;

      for (int k = 0; k < 3; k++)
        for (int l = 0; l < L; l++) begin
          xa_v[k][l] = $random;
          w_v[k][l]  = $random;
        end

      // Golden: each lane independently sums 3 products
      for (int l = 0; l < L; l++) begin
        golden[l] = 0;
        for (int k = 0; k < 3; k++)
          golden[l] += 32'(xa_v[k][l]) * 32'(w_v[k][l]);
      end

      // Drive
      for (int k = 0; k < 3; k++) begin
        @(posedge clk);
        pe_enable <= 1'b1;
        clear_acc <= (k == 0);
        for (int l = 0; l < L; l++) begin
          x_in[l] <= xa_v[k][l];
          w_in[l] <= w_v[k][l];
        end
      end
      @(posedge clk);
      pe_enable <= 1'b0; clear_acc <= 1'b0;
      for (int l = 0; l < L; l++) begin x_in[l] <= 0; w_in[l] <= 0; end
      wait_pipe();

      for (int l = 0; l < L; l++) begin
        if (psum_out[l] !== golden[l]) begin
          $display("  ERR lane %0d: got=%0d exp=%0d", l, psum_out[l], golden[l]);
          t_err++;
        end
      end

      total_tests++; total_errors += t_err;
      $display("  TEST 3 %s (%0d lane errors)", t_err == 0 ? "PASS" : "FAIL", t_err);
    end

    // ══════════════════════════════════════════════════════════
    //  TEST 4: Enable Gating — psum holds when disabled
    // ══════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] snap [L];
      $display("\n── TEST 4: Enable Gating ──");
      do_reset();
      pe_mode = PE_RS3;

      // Accumulate 3 beats
      for (int k = 0; k < 3; k++) begin
        @(posedge clk);
        pe_enable <= 1'b1; clear_acc <= (k == 0);
        for (int l = 0; l < L; l++) begin
          x_in[l] <= (k * 10 + l);
          w_in[l] <= (k + 1);
        end
      end
      @(posedge clk);
      pe_enable <= 1'b0; clear_acc <= 1'b0;
      for (int l = 0; l < L; l++) begin x_in[l] <= 0; w_in[l] <= 0; end
      wait_pipe();

      // Snapshot
      for (int l = 0; l < L; l++) snap[l] = psum_out[l];

      // Drive non-zero with en=0 for 15 cycles
      for (int l = 0; l < L; l++) begin x_in[l] <= 8'sd100; w_in[l] <= 8'sd50; end
      repeat (15) @(posedge clk);
      for (int l = 0; l < L; l++) begin x_in[l] <= 0; w_in[l] <= 0; end
      wait_pipe();

      // Verify unchanged
      for (int l = 0; l < L; l++) begin
        if (psum_out[l] !== snap[l]) begin
          $display("  ERR lane %0d: changed %0d->%0d", l, snap[l], psum_out[l]);
          t_err++;
        end
      end

      total_tests++; total_errors += t_err;
      $display("  TEST 4 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ══════════════════════════════════════════════════════════
    //  TEST 5: Clear + Immediate Start
    // ══════════════════════════════════════════════════════════
    begin
      integer t_err = 0;
      logic signed [31:0] golden [L];
      $display("\n── TEST 5: Clear + Immediate Start ──");
      do_reset();
      pe_mode = PE_RS3;

      // Pre-fill with garbage (5 beats)
      for (int k = 0; k < 5; k++) begin
        @(posedge clk);
        pe_enable <= 1'b1; clear_acc <= (k == 0);
        for (int l = 0; l < L; l++) begin x_in[l] <= $random; w_in[l] <= $random; end
      end

      // Now clear and accumulate known data (3 beats)
      for (int l = 0; l < L; l++) golden[l] = 0;
      for (int k = 0; k < 3; k++) begin
        @(posedge clk);
        pe_enable <= 1'b1; clear_acc <= (k == 0);
        for (int l = 0; l < L; l++) begin
          automatic logic signed [7:0] xv = 10 + l;
          automatic logic signed [7:0] wv = k + 1;
          x_in[l] <= xv; w_in[l] <= wv;
          golden[l] += 32'(xv) * 32'(wv);
        end
      end
      @(posedge clk);
      pe_enable <= 1'b0; clear_acc <= 1'b0;
      for (int l = 0; l < L; l++) begin x_in[l] <= 0; w_in[l] <= 0; end
      wait_pipe();

      for (int l = 0; l < L; l++) begin
        if (psum_out[l] !== golden[l]) begin
          $display("  ERR lane %0d: got=%0d exp=%0d", l, psum_out[l], golden[l]);
          t_err++;
        end
      end

      total_tests++; total_errors += t_err;
      $display("  TEST 5 %s", t_err == 0 ? "PASS" : "FAIL");
    end

    // ══════════════════════════════════════════════════════════
    //  SUMMARY
    // ══════════════════════════════════════════════════════════
    $display("\n══════════════════════════════════════════════════════════");
    $display(" FINAL: %0d tests, %0d errors", total_tests, total_errors);
    if (total_errors == 0)
      $display(" ★★★ ALL PASS — pe_unit VERIFIED ★★★");
    else
      $display(" ✗✗✗ FAIL — %0d errors ✗✗✗", total_errors);
    $display("══════════════════════════════════════════════════════════\n");
    $finish;
  end

  initial begin #200_000_000; $display("TIMEOUT"); $finish; end

endmodule
