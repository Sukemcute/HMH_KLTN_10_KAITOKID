`timescale 1ns/1ps
// Full PE array: 3 rows x 4 cols x 32 lanes = 12 PE units + column_reduce + comparator_tree.
module pe_cluster #(
  parameter int LANES   = 32,
  parameter int PE_ROWS = 3,
  parameter int PE_COLS = 4
)(
  input  logic                clk,
  input  logic                rst_n,
  input  logic                en,
  input  logic                clear_psum,
  input  accel_pkg::pe_mode_e mode,

  // Activation taps from window_gen (per PE row)
  input  logic signed [7:0]  act_taps [PE_ROWS][LANES],
  // Weight data from router (per PE row, broadcast to PE cols)
  input  logic signed [7:0]  wgt_data [PE_ROWS][LANES],

  // Psum input from bank_output (multi-pass accumulation)
  input  logic signed [31:0] psum_in [PE_COLS][LANES],
  input  logic               psum_in_valid,

  // Psum output after column reduction
  output logic signed [31:0] psum_out [PE_COLS][LANES],
  output logic               psum_out_valid,

  // MAXPOOL output (bypass psum path)
  input  logic signed [7:0]  pool_data_in [25][LANES],
  input  logic               pool_en,
  output logic signed [7:0]  pool_out [LANES],
  output logic               pool_out_valid
);
  import accel_pkg::*;

  // ───── PE Array (3x4 grid) ─────
  logic signed [31:0] pe_psum [PE_ROWS][PE_COLS][LANES];
  logic               pe_valid [PE_ROWS][PE_COLS];

  genvar r, c;
  generate
    for (r = 0; r < PE_ROWS; r++) begin : gen_row
      for (c = 0; c < PE_COLS; c++) begin : gen_col
        // Each PE in a column gets the same weight (broadcast)
        // Each PE in a row gets the same activation (shared taps)
        pe_unit #(.LANES(LANES)) u_pe (
          .clk        (clk),
          .rst_n      (rst_n),
          .en         (en),
          .clear_psum (clear_psum),
          .mode       (mode),
          .x_in       (act_taps[r]),
          .w_in       (wgt_data[r]),
          .psum_out   (pe_psum[r][c]),
          .psum_valid (pe_valid[r][c])
        );
      end
    end
  endgenerate

  // ───── Column Reduce (sum 3 rows → 1 psum per col) ─────
  logic signed [31:0] col_psum [PE_COLS][LANES];
  logic               col_valid;

  column_reduce #(
    .LANES  (LANES),
    .PE_ROWS(PE_ROWS),
    .PE_COLS(PE_COLS)
  ) u_col_reduce (
    .clk      (clk),
    .rst_n    (rst_n),
    .en       (en),
    .mode     (mode),
    .pe_psum  (pe_psum),
    .col_psum (col_psum),
    .col_valid(col_valid)
  );

  // ───── Multi-pass Accumulation ─────
  // When psum_in_valid, add previous partial sum to current result
  always_comb begin
    for (int cc = 0; cc < PE_COLS; cc++) begin
      for (int ll = 0; ll < LANES; ll++) begin
        if (psum_in_valid)
          psum_out[cc][ll] = col_psum[cc][ll] + psum_in[cc][ll];
        else
          psum_out[cc][ll] = col_psum[cc][ll];
      end
    end
  end

  assign psum_out_valid = col_valid;

  // ───── Comparator Tree (MAXPOOL mode) ─────
  comparator_tree #(
    .LANES     (LANES),
    .NUM_INPUTS(25)
  ) u_comp_tree (
    .clk      (clk),
    .rst_n    (rst_n),
    .en       (pool_en),
    .data_in  (pool_data_in),
    .max_out  (pool_out),
    .max_valid(pool_out_valid)
  );

endmodule
