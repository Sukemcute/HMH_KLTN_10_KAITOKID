// ============================================================================
// Module : desc_fetch_engine
// Project: YOLOv10n INT8 Accelerator — V4-VC707
//
// Fetches NET_DESC → LAYER_DESC[0..22] → TILE_DESC from DDR3 via AXI4.
// Parses binary blobs into structured desc_pkg types for the scheduler.
// ============================================================================
`timescale 1ns / 1ps

module desc_fetch_engine
  import accel_pkg::*;
  import desc_pkg::*;
#(
  parameter int ADDR_W = AXI_ADDR_WIDTH,
  parameter int DATA_W = AXI_DATA_WIDTH
)(
  input  logic              clk,
  input  logic              rst_n,

  // ═══════════ CONTROL (from CSR) ═══════════
  input  logic              start,
  input  logic [ADDR_W-1:0] net_desc_addr,
  input  logic [7:0]        layer_start,  // first layer to execute
  input  logic [7:0]        layer_end,    // last layer to execute
  output logic              busy,
  output logic              done,

  // ═══════════ OUTPUT: parsed descriptors → scheduler ═══════════
  output logic              desc_valid,
  output layer_desc_t       desc_layer,
  output tile_desc_t        desc_tile,
  input  logic              desc_ready,

  // ═══════════ AXI4 READ (desc memory) ═══════════
  output logic              axi_ar_valid,
  input  logic              axi_ar_ready,
  output logic [ADDR_W-1:0] axi_ar_addr,
  output logic [7:0]        axi_ar_len,
  output logic [2:0]        axi_ar_size,
  output logic [1:0]        axi_ar_burst,

  input  logic              axi_r_valid,
  output logic              axi_r_ready,
  input  logic [DATA_W-1:0] axi_r_data,
  input  logic              axi_r_last
);

  assign axi_ar_size  = 3'b101;  // 32 bytes
  assign axi_ar_burst = 2'b01;   // INCR

  typedef enum logic [3:0] {
    DF_IDLE,
    DF_FETCH_NET,
    DF_WAIT_NET,
    DF_FETCH_LAYER,
    DF_WAIT_LAYER,
    DF_FETCH_TILE,
    DF_WAIT_TILE,
    DF_EMIT_TILE,
    DF_NEXT_LAYER,
    DF_DONE
  } df_state_e;
  df_state_e state;

  // Parsed net descriptor
  net_desc_t   net_desc_r;
  layer_desc_t current_layer;
  tile_desc_t  current_tile;

  logic [7:0]  cur_layer_id;
  logic [7:0]  cur_tile_idx;
  logic [7:0]  total_tiles_in_layer;

  logic [ADDR_W-1:0] layer_table_addr;
  logic [ADDR_W-1:0] tile_base_addr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= DF_IDLE;
      busy         <= 1'b0;
      done         <= 1'b0;
      desc_valid   <= 1'b0;
      axi_ar_valid <= 1'b0;
      axi_r_ready  <= 1'b0;
      cur_layer_id <= 8'd0;
      cur_tile_idx <= 8'd0;
    end else begin
      case (state)
        DF_IDLE: begin
          done <= 1'b0;
          if (start) begin
            busy         <= 1'b1;
            cur_layer_id <= layer_start;
            state        <= DF_FETCH_NET;
          end
        end

        // ─── Fetch NET descriptor ───
        DF_FETCH_NET: begin
          axi_ar_valid <= 1'b1;
          axi_ar_addr  <= net_desc_addr;
          axi_ar_len   <= 8'd0;  // 1 beat
          if (axi_ar_ready) begin
            axi_ar_valid <= 1'b0;
            axi_r_ready  <= 1'b1;
            state        <= DF_WAIT_NET;
          end
        end

        DF_WAIT_NET: begin
          if (axi_r_valid) begin
            axi_r_ready <= 1'b0;
            // Parse net descriptor from AXI data (simplified)
            net_desc_r.magic            <= axi_r_data[31:0];
            net_desc_r.num_layers       <= axi_r_data[55:48];
            net_desc_r.layer_table_base <= axi_r_data[127:64];
            layer_table_addr            <= axi_r_data[127:64];
            state <= DF_FETCH_LAYER;
          end
        end

        // ─── Fetch LAYER descriptor ───
        DF_FETCH_LAYER: begin
          axi_ar_valid <= 1'b1;
          axi_ar_addr  <= layer_table_addr + (ADDR_W'(cur_layer_id) * 32);
          axi_ar_len   <= 8'd0;
          if (axi_ar_ready) begin
            axi_ar_valid <= 1'b0;
            axi_r_ready  <= 1'b1;
            state        <= DF_WAIT_LAYER;
          end
        end

        DF_WAIT_LAYER: begin
          if (axi_r_valid) begin
            axi_r_ready <= 1'b0;
            // Parse layer descriptor (simplified: take fields from data)
            current_layer.layer_id   <= axi_r_data[4:0];
            current_layer.pe_mode    <= pe_mode_e'(axi_r_data[8:5]);
            current_layer.activation <= act_mode_e'(axi_r_data[10:9]);
            current_layer.cin        <= axi_r_data[20:11];
            current_layer.cout       <= axi_r_data[30:21];
            current_layer.hin        <= axi_r_data[40:31];
            current_layer.win        <= axi_r_data[50:41];
            current_layer.hout       <= axi_r_data[60:51];
            current_layer.wout       <= axi_r_data[70:61];
            current_layer.kh         <= axi_r_data[74:71];
            current_layer.kw         <= axi_r_data[78:75];
            current_layer.stride     <= axi_r_data[81:79];
            current_layer.padding    <= axi_r_data[84:82];
            current_layer.num_tiles  <= axi_r_data[92:85];
            current_layer.num_k_pass <= axi_r_data[96:93];
            current_layer.swizzle    <= swizzle_mode_e'(axi_r_data[98:97]);
            total_tiles_in_layer     <= axi_r_data[92:85];
            cur_tile_idx             <= 8'd0;
            // Compute tile base address (after layer table)
            tile_base_addr <= layer_table_addr + (ADDR_W'(32) * 32);
            state <= DF_FETCH_TILE;
          end
        end

        // ─── Fetch TILE descriptors for current layer ───
        DF_FETCH_TILE: begin
          if (cur_tile_idx >= total_tiles_in_layer) begin
            state <= DF_NEXT_LAYER;
          end else begin
            axi_ar_valid <= 1'b1;
            axi_ar_addr  <= tile_base_addr + (ADDR_W'(cur_tile_idx) * 32);
            axi_ar_len   <= 8'd0;
            if (axi_ar_ready) begin
              axi_ar_valid <= 1'b0;
              axi_r_ready  <= 1'b1;
              state        <= DF_WAIT_TILE;
            end
          end
        end

        DF_WAIT_TILE: begin
          if (axi_r_valid) begin
            axi_r_ready <= 1'b0;
            // Parse tile descriptor
            current_tile.tile_id    <= axi_r_data[15:0];
            current_tile.layer_id   <= cur_layer_id[4:0];
            current_tile.valid_h    <= axi_r_data[21:16];
            current_tile.valid_w    <= axi_r_data[27:22];
            current_tile.first_tile <= (cur_tile_idx == 0);
            current_tile.last_tile  <= (cur_tile_idx == total_tiles_in_layer - 1);
            state <= DF_EMIT_TILE;
          end
        end

        DF_EMIT_TILE: begin
          desc_valid <= 1'b1;
          desc_layer <= current_layer;
          desc_tile  <= current_tile;
          if (desc_ready) begin
            desc_valid   <= 1'b0;
            cur_tile_idx <= cur_tile_idx + 8'd1;
            state        <= DF_FETCH_TILE;
          end
        end

        // ─── Next layer ───
        DF_NEXT_LAYER: begin
          if (cur_layer_id >= layer_end)
            state <= DF_DONE;
          else begin
            cur_layer_id <= cur_layer_id + 8'd1;
            state        <= DF_FETCH_LAYER;
          end
        end

        DF_DONE: begin
          busy <= 1'b0;
          done <= 1'b1;
          state <= DF_IDLE;
        end

        default: state <= DF_IDLE;
      endcase
    end
  end

endmodule
