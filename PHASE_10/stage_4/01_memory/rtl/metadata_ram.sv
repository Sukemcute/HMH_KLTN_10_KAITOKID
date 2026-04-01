`timescale 1ns/1ps

// Slot validity and ring buffer pointer management for input/output banks.
module metadata_ram #(
  parameter int NUM_SLOTS = 16,
  parameter int META_BITS = 32,
  parameter int SLOT_W    = $clog2(NUM_SLOTS)
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    clear_all,

  // Set valid
  input  logic                    set_valid,
  input  logic [SLOT_W-1:0]      set_slot_id,
  input  logic [META_BITS-1:0]   set_meta,

  // Query
  input  logic [SLOT_W-1:0]      query_slot_id,
  output logic                    query_valid,
  output logic [META_BITS-1:0]   query_meta,

  // Ring management
  input  logic                    advance_ring,
  output logic [SLOT_W-1:0]      ring_head,
  output logic [SLOT_W-1:0]      ring_tail,
  output logic                    ring_full,
  output logic                    ring_empty
);

  logic                  valid_bits [NUM_SLOTS];
  logic [META_BITS-1:0]  meta_store [NUM_SLOTS];
  logic [SLOT_W:0]       head_ptr, tail_ptr;

  wire [SLOT_W:0] occupancy = tail_ptr - head_ptr;

  assign ring_head  = head_ptr[SLOT_W-1:0];
  assign ring_tail  = tail_ptr[SLOT_W-1:0];
  assign ring_full  = (occupancy == NUM_SLOTS[SLOT_W:0]);
  assign ring_empty = (occupancy == 0);

  // Query output (combinational)
  assign query_valid = valid_bits[query_slot_id];
  assign query_meta  = meta_store[query_slot_id];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n || clear_all) begin
      head_ptr <= '0;
      tail_ptr <= '0;
      for (int i = 0; i < NUM_SLOTS; i++) begin
        valid_bits[i] <= 1'b0;
        meta_store[i] <= '0;
      end
    end else begin
      if (set_valid) begin
        valid_bits[set_slot_id] <= 1'b1;
        meta_store[set_slot_id] <= set_meta;
        tail_ptr <= tail_ptr + 1;
      end

      if (advance_ring && !ring_empty) begin
        valid_bits[head_ptr[SLOT_W-1:0]] <= 1'b0;
        head_ptr <= head_ptr + 1;
      end
    end
  end

endmodule
