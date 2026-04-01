`timescale 1ns/1ps

// ================================================================
//  PPU Golden Verification Testbench
//  Feeds pre-computed INT32 psums through the RTL PPU and compares
//  output with golden data (from export_rtl_params.py).
//
//  Tests: Bias Add -> Requantize (m_int, shift) -> SiLU LUT -> zp_out
//
//  Compile & run (from Vivado Tcl, cd to SW_KLTN):
//    source PHASE_4/03_rtl_cosim/run_ppu_golden.tcl
//    run_ppu_test 0    ;# or 1, 3, 17
// ================================================================

module tb_ppu_golden #(
    parameter int LAYER_IDX    = 0,
    parameter int NUM_POSITIONS = 4
);
    import accel_pkg::*;
    import desc_pkg::*;

    localparam int LANES   = 32;
    localparam int MAX_CH  = 512;
    localparam int PIPE_DEPTH = 8;  // extra margin for 4-stage pipeline

    // ---- Clock & reset ----
    logic clk, rst_n;
    initial begin clk = 0; forever #2.5 clk = ~clk; end

    // ---- PPU ports ----
    logic en;
    post_profile_t cfg_post;
    pe_mode_e      cfg_mode;

    logic signed [31:0] psum_in    [LANES];
    logic               psum_valid;
    logic signed [31:0] bias_val   [LANES];
    logic signed [31:0] m_int_r    [LANES];
    logic [5:0]         shift_r    [LANES];
    logic signed [7:0]  zp_out;
    logic signed [7:0]  silu_lut   [256];
    logic signed [7:0]  ewise_in   [LANES];
    logic               ewise_valid;
    logic signed [7:0]  act_out    [LANES];
    logic               act_valid;

    // ---- DUT ----
    ppu #(.LANES(LANES)) u_ppu (
        .clk           (clk),
        .rst_n         (rst_n),
        .en            (en),
        .cfg_post      (cfg_post),
        .cfg_mode      (cfg_mode),
        .psum_in       (psum_in),
        .psum_valid    (psum_valid),
        .bias_val      (bias_val),
        .m_int         (m_int_r),
        .shift         (shift_r),
        .zp_out        (zp_out),
        .silu_lut_data (silu_lut),
        .ewise_in      (ewise_in),
        .ewise_valid   (ewise_valid),
        .act_out       (act_out),
        .act_valid     (act_valid)
    );

    // ---- Hex data memories ----
    reg [31:0] mem_m_int   [0:MAX_CH-1];
    reg [7:0]  mem_shift   [0:MAX_CH-1];
    reg [7:0]  mem_silu    [0:255];
    reg [31:0] mem_bias    [0:MAX_CH-1];
    reg [7:0]  mem_zp      [0:0];
    reg [31:0] mem_psum    [0:MAX_CH-1];
    reg [7:0]  mem_golden  [0:MAX_CH-1];

    // ---- Output capture (latch on act_valid) ----
    logic [7:0] cap_out [LANES];
    logic       cap_flag;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cap_flag <= 1'b0;
        end else if (act_valid) begin
            cap_flag <= 1'b1;
            for (int l = 0; l < LANES; l++)
                cap_out[l] <= act_out[l];
        end else begin
            cap_flag <= 1'b0;
        end
    end

    // ---- Variables ----
    int cout;
    int total_errors, total_checked, pos_errors;
    string fpath;

    initial begin
        $display("");
        $display("================================================================");
        $display("  PPU Golden Verification - Layer %0d", LAYER_IDX);
        $display("================================================================");

        // ---- Load fixed params (use $sformatf for each path) ----
        fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/m_int.hex", LAYER_IDX);
        $readmemh(fpath, mem_m_int);
        fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/shift.hex", LAYER_IDX);
        $readmemh(fpath, mem_shift);
        fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/silu_lut.hex", LAYER_IDX);
        $readmemh(fpath, mem_silu);
        fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/bias.hex", LAYER_IDX);
        $readmemh(fpath, mem_bias);
        fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/zp_out.hex", LAYER_IDX);
        $readmemh(fpath, mem_zp);

        // cout per layer (must match Python export)
        case (LAYER_IDX)
            0:  cout = 16;
            1:  cout = 32;
            3:  cout = 64;
            17: cout = 64;
            default: cout = 32;
        endcase

        // Copy SiLU LUT (interpret as signed int8)
        for (int i = 0; i < 256; i++)
            silu_lut[i] = $signed(mem_silu[i]);

        zp_out = $signed(mem_zp[0]);

        $display("  cout=%0d  zp_out=%0d  m_int[0]=%0d  shift[0]=%0d",
                 cout, zp_out, $signed(mem_m_int[0]), mem_shift[0]);

        // ---- PPU static configuration ----
        cfg_post.bias_en           = 1'b1;
        cfg_post.quant_mode        = QMODE_PER_CHANNEL;
        cfg_post.act_mode          = ACT_SILU;
        cfg_post.ewise_en          = 1'b0;
        cfg_post.bias_scale_offset = '0;
        cfg_post.concat_ch_offset  = '0;
        cfg_post.upsample_factor   = '0;
        cfg_mode = PE_RS3;

        en          = 1'b1;
        psum_valid  = 1'b0;
        ewise_valid = 1'b0;
        for (int l = 0; l < LANES; l++) begin
            psum_in[l]  = '0;
            bias_val[l] = '0;
            m_int_r[l]  = '0;
            shift_r[l]  = '0;
            ewise_in[l] = '0;
        end

        // ---- Reset ----
        rst_n = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (3) @(posedge clk);

        total_errors  = 0;
        total_checked = 0;

        // ============================================
        //  Run NUM_POSITIONS test points per layer
        // ============================================
        for (int pos = 0; pos < NUM_POSITIONS; pos++) begin
            fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/psum_pos%0d.hex", LAYER_IDX, pos);
            $readmemh(fpath, mem_psum);
            fpath = $sformatf("E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L%02d/golden_pos%0d.hex", LAYER_IDX, pos);
            $readmemh(fpath, mem_golden);

            $display("\n  --- Position %0d ---", pos);
            pos_errors = 0;

            // Process in groups of LANES
            for (int ch0 = 0; ch0 < cout; ch0 += LANES) begin
                automatic int n_ch = ((cout - ch0) < LANES) ? (cout - ch0) : LANES;

                // Setup per-lane parameters
                @(negedge clk);
                for (int l = 0; l < LANES; l++) begin
                    if (l < n_ch) begin
                        psum_in[l]  = $signed(mem_psum[ch0 + l]);
                        bias_val[l] = $signed(mem_bias[ch0 + l]);
                        m_int_r[l]  = $signed(mem_m_int[ch0 + l]);
                        shift_r[l]  = mem_shift[ch0 + l][5:0];
                    end else begin
                        psum_in[l]  = '0;
                        bias_val[l] = '0;
                        m_int_r[l]  = '0;
                        shift_r[l]  = '0;
                    end
                end

                // Pulse psum_valid for exactly 1 cycle
                @(negedge clk);
                psum_valid = 1'b1;
                @(negedge clk);
                psum_valid = 1'b0;

                // Wait for pipeline to flush (4 stages + margin)
                repeat (PIPE_DEPTH) @(posedge clk);
                #1;  // let NBA settle

                // Compare captured output
                for (int l = 0; l < n_ch; l++) begin
                    automatic logic [7:0] got;
                    automatic logic [7:0] exp;
                    got = cap_out[l];
                    exp = mem_golden[ch0 + l];
                    total_checked++;
                    if (got !== exp) begin
                        total_errors++;
                        pos_errors++;
                        if (pos_errors <= 5)
                            $display("    FAIL ch[%0d]: got=0x%02X(%0d) exp=0x%02X(%0d)",
                                     ch0+l, got, got, exp, exp);
                    end
                end
            end

            if (pos_errors == 0)
                $display("    PASS: %0d channels correct", cout);
            else
                $display("    FAIL: %0d/%0d channel errors", pos_errors, cout);
        end

        // ---- Summary ----
        $display("");
        $display("================================================================");
        if (total_errors == 0) begin
            $display("  [PASS] Layer %0d PPU: ALL %0d values bit-exact", LAYER_IDX, total_checked);
        end else begin
            $display("  [FAIL] Layer %0d PPU: %0d/%0d errors", LAYER_IDX, total_errors, total_checked);
        end
        $display("================================================================");

        #100;
        $finish;
    end

endmodule
