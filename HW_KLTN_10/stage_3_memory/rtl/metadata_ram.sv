// ============================================================================
// Module : metadata_ram
// Project: YOLOv10n INT8 Accelerator — V4-VC707
// Description:
//   Slot validity and ring buffer management for tile descriptors.
//   NUM_SLOTS entries, each holding META_BITS of metadata and a valid flag.
//
//   Producer interface: set_valid + set_slot_id + set_meta marks a slot valid.
//   Consumer interface: query_slot_id returns query_valid + query_meta.
//   Ring management:    head/tail pointers with full/empty flags.
//     - advance_ring: consume and invalidate the head slot.
//     - clear_all:    reset all validity, head, tail, and count.
// ============================================================================
`timescale 1ns / 1ps

module metadata_ram #(
  parameter NUM_SLOTS = 16,
  parameter META_BITS = 32
)(
  input  logic                          clk,
  input  logic                          rst_n,

  // Producer — mark slot valid with metadata
  input  logic                          set_valid,
  input  logic [$clog2(NUM_SLOTS)-1:0]  set_slot_id,
  input  logic [META_BITS-1:0]          set_meta,

  // Consumer — query slot status
  input  logic [$clog2(NUM_SLOTS)-1:0]  query_slot_id,
  output logic                          query_valid,
  output logic [META_BITS-1:0]          query_meta,

  // Ring management
  input  logic                          advance_ring,
  input  logic                          clear_all,

  // Ring status
  output logic [$clog2(NUM_SLOTS)-1:0]  head_ptr,
  output logic [$clog2(NUM_SLOTS)-1:0]  tail_ptr,
  output logic                          ring_full,
  output logic                          ring_empty
);

  import accel_pkg::*;

  localparam SLOT_BITS = $clog2(NUM_SLOTS);

  // --------------------------------------------------------------------------
  //  Storage arrays
  // --------------------------------------------------------------------------
  logic [META_BITS-1:0] meta_store [NUM_SLOTS];
  logic                 valid_store [NUM_SLOTS];

  // --------------------------------------------------------------------------
  //  Ring count (tracks number of valid entries in ring order)
  // --------------------------------------------------------------------------
  logic [SLOT_BITS:0] ring_count;  // extra bit to distinguish full vs empty

  assign ring_full  = (ring_count == NUM_SLOTS[SLOT_BITS:0]);
  assign ring_empty = (ring_count == '0);

  // --------------------------------------------------------------------------
  //  Head / tail pointer management
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      head_ptr   <= '0;
      tail_ptr   <= '0;
      ring_count <= '0;
    end else if (clear_all) begin
      head_ptr   <= '0;
      tail_ptr   <= '0;
      ring_count <= '0;
    end else begin
      // Producer enqueue — advances tail
      if (set_valid && !ring_full) begin
        tail_ptr   <= (tail_ptr == SLOT_BITS'(NUM_SLOTS - 1)) ? '0 : tail_ptr + 1'b1;
        ring_count <= ring_count + 1'b1;
      end

      // Consumer dequeue — advances head
      if (advance_ring && !ring_empty) begin
        head_ptr   <= (head_ptr == SLOT_BITS'(NUM_SLOTS - 1)) ? '0 : head_ptr + 1'b1;
        ring_count <= ring_count - 1'b1;
      end

      // Simultaneous enqueue + dequeue — count unchanged, both pointers advance
      if (set_valid && !ring_full && advance_ring && !ring_empty) begin
        ring_count <= ring_count;  // net zero change
      end
    end
  end

  // --------------------------------------------------------------------------
  //  Validity and metadata storage
  // --------------------------------------------------------------------------
  genvar s;
  generate
    for (s = 0; s < NUM_SLOTS; s++) begin : gen_slot

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          valid_store[s] <= 1'b0;
          meta_store[s]  <= '0;
        end else if (clear_all) begin
          valid_store[s] <= 1'b0;
          meta_store[s]  <= '0;
        end else begin
          // Producer sets validity + metadata at the addressed slot
          if (set_valid && (set_slot_id == SLOT_BITS'(s)))  begin
            valid_store[s] <= 1'b1;
            meta_store[s]  <= set_meta;
          end

          // Consumer invalidates head slot on advance
          if (advance_ring && !ring_empty && (head_ptr == SLOT_BITS'(s))) begin
            valid_store[s] <= 1'b0;
          end
        end
      end

    end // gen_slot
  endgenerate

  // --------------------------------------------------------------------------
  //  Consumer query — combinational read (registered externally if needed)
  // --------------------------------------------------------------------------
  assign query_valid = valid_store[query_slot_id];
  assign query_meta  = meta_store[query_slot_id];

endmodule
