`timescale 1ns/1ps
// Layer 0: SV reference must match verify_conv_layer.py (quant + bias + requant + SiLU).
// Not "ảnh cũ" — old TB used naive INT8 MAC+clamp (wrong). Uses same hex as layer_by_layer export.
module tb_single_layer;

  // XSIM: use absolute paths. Edit if project root moves.
  localparam string G_INPUT    = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/02_golden_data/layer_by_layer/act_L00_input.hex";
  localparam string G_WEIGHT   = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/02_golden_data/layer_by_layer/weight_L00_conv.hex";
  localparam string G_BIAS     = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/02_golden_data/bias_L0_conv.hex";
  localparam string G_PY_OUT   = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/02_golden_data/layer_by_layer/act_L00_output.hex";
  // Same SiLU table as verify_conv_layer.py (export: python export_rtl_params.py --layer 0)
  localparam string G_SILU_QUANT = "E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/rtl_test_L00/silu_lut_quant.hex";

  localparam CLK_PERIOD = 5;
  localparam int CIN  = 3;
  localparam int COUT = 16;
  localparam int KH   = 3;
  localparam int KW   = 3;
  localparam int SH   = 2;
  localparam int SW   = 2;
  localparam int HIN  = 640;
  localparam int WIN  = 640;
  localparam int HOUT = 320;
  localparam int WOUT = 320;

  localparam int INPUT_SIZE  = CIN * HIN * WIN;
  localparam int WEIGHT_SIZE = COUT * CIN * KH * KW;
  localparam int OUTPUT_SIZE = COUT * HOUT * WOUT;
  localparam int MEM_MAX     = (INPUT_SIZE > OUTPUT_SIZE) ? INPUT_SIZE : OUTPUT_SIZE;
  localparam int MEM_MAX_ALL = (WEIGHT_SIZE > MEM_MAX) ? WEIGHT_SIZE : MEM_MAX;

  // Quant metadata — must match layer_summary.json layer 0 (export_layer_by_layer)
  localparam real INP_SCALE  = 0.007874015718698502;
  localparam int  INP_ZP     = 0;
  localparam real OUT_SCALE  = 0.35273128747940063;
  localparam int  OUT_ZP     = 62;

  localparam int PAD = 1;

  logic clk = 0;
  logic rst_n;
  always #(CLK_PERIOD/2.0) clk = ~clk;

  logic signed [7:0] input_mem  [0:MEM_MAX_ALL-1];
  logic signed [7:0] weight_mem [0:MEM_MAX_ALL-1];
  logic [7:0]        sv_ref     [0:MEM_MAX_ALL-1];
  logic signed [7:0] py_golden  [0:MEM_MAX_ALL-1];
  int                bias_i32   [0:COUT-1];
  logic [7:0]        silu_lut   [0:255];

  // Per-channel weight scales (from quant export, layer 0 conv)
  function automatic real wscale(input int co);
    case (co)
      0:  return 0.035110294818878174;
      1:  return 0.011435432359576225;
      2:  return 0.045557599514722824;
      3:  return 0.028462009504437447;
      4:  return 0.06397058814764023;
      5:  return 0.0208639707416296;
      6:  return 0.038204655051231384;
      7:  return 0.035140931606292725;
      8:  return 0.03308854252099991;
      9:  return 0.028768228366971016;
      10: return 0.027757352218031883;
      11: return 0.030912989750504494;
      12: return 0.035171568393707275;
      13: return 0.03422181308269501;
      14: return 0.039368871599435806;
      15: return 0.05039828270673752;
      default: return 1.0;
    endcase
  endfunction

  function automatic int hex_char_to_nibble(input byte c);
    if (c >= "0" && c <= "9") return c - "0";
    if (c >= "A" && c <= "F") return c - "A" + 10;
    if (c >= "a" && c <= "f") return c - "a" + 10;
    return -1;
  endfunction

  // Python 3 round(x): nearest int, ties to even — NOT $rtoi(x+0.5). Widen |t-0.5|<eps for float noise.
  function automatic int round_py3(input real x);
    real ax, f, t;
    int fl, res, sgn;
    sgn = (x >= 0.0) ? 1 : -1;
    ax  = (x >= 0.0) ? x : -x;
    f   = $floor(ax);
    t   = ax - f;
    if (t < 0.5 - 1e-7)
      fl = $rtoi(f);
    else if (t > 0.5 + 1e-7)
      fl = $rtoi(f + 1.0);
    else begin
      fl = $rtoi(f);
      if ((fl % 2) == 0)
        res = fl;
      else
        res = fl + 1;
      fl = res;
    end
    return sgn * fl;
  endfunction

  task automatic load_int8_hex(
    input string filename,
    ref   logic signed [7:0] mem [0:MEM_MAX_ALL-1],
    input int max_elements
  );
    integer fd, idx, hn;
    string line;
    fd = $fopen(filename, "r");
    if (fd == 0) begin
      $display("ERROR: Cannot open %s", filename);
      return;
    end
    idx = 0;
    while (!$feof(fd) && idx < max_elements) begin
      if ($fgets(line, fd)) begin
        for (int i = 0; i < line.len() - 1 && idx < max_elements; i += 2) begin
          automatic int h0 = hex_char_to_nibble(line.getc(i));
          automatic int h1 = hex_char_to_nibble(line.getc(i + 1));
          if (h0 >= 0 && h1 >= 0) begin
            mem[idx] = {h0[3:0], h1[3:0]};
            idx = idx + 1;
          end
        end
      end
    end
    $fclose(fd);
    $display("  Loaded %0d int8 values from %s", idx, filename);
  endtask

  task automatic load_bias_int32_hex(input string filename);
    integer fd, bi, i, j, hn;
    string line;
    logic [31:0] wv;
    fd = $fopen(filename, "r");
    if (fd == 0) begin
      $display("ERROR: Cannot open bias %s", filename);
      for (int b = 0; b < COUT; b++) bias_i32[b] = 0;
      return;
    end
    bi = 0;
    while (!$feof(fd) && bi < COUT) begin
      if ($fgets(line, fd)) begin
        for (i = 0; i + 8 <= line.len() - 1 && bi < COUT; i += 8) begin
          wv = 32'h0;
          for (j = 0; j < 8; j++) begin
            hn = hex_char_to_nibble(line.getc(i + j));
            if (hn < 0) break;
            wv = (wv << 4) | hn[3:0];
          end
          bias_i32[bi] = $signed(wv);
          bi++;
        end
      end
    end
    $fclose(fd);
    $display("  Loaded %0d INT32 biases from %s", bi, filename);
  endtask

  task automatic build_silu_lut;
    real x, silu_val, qf;
    int  qi;
    for (int q = 0; q < 256; q++) begin
      x = (real'(q) - real'(OUT_ZP)) * OUT_SCALE;
      if (x > -1e-12 && x < 1e-12)
        silu_val = 0.0;
      else
        silu_val = x / (1.0 + $exp(-x));
      qf = silu_val / OUT_SCALE;
      qi = round_py3(qf) + OUT_ZP;
      if (qi < 0) qi = 0;
      if (qi > 255) qi = 255;
      silu_lut[q] = qi[7:0];
    end
  endtask

  task automatic compute_l0_like_python;
    int co, ci, oh, ow, kh_i, kw_i, ih, iw;
    longint acc;
    int act_z, w_z;
    real fv;
    int pre_silu, qi;
    $display("  Computing L0 reference (MAC + bias + requant + SiLU LUT)...");
    for (co = 0; co < COUT; co++) begin
      for (oh = 0; oh < HOUT; oh++) begin
        for (ow = 0; ow < WOUT; ow++) begin
          acc = 0;
          for (ci = 0; ci < CIN; ci++) begin
            for (kh_i = 0; kh_i < KH; kh_i++) begin
              for (kw_i = 0; kw_i < KW; kw_i++) begin
                ih = oh * SH + kh_i - PAD;
                iw = ow * SW + kw_i - PAD;
                if (ih >= 0 && ih < HIN && iw >= 0 && iw < WIN) begin
                  act_z = input_mem[ci * HIN * WIN + ih * WIN + iw] - INP_ZP;
                  w_z   = weight_mem[co * CIN * KH * KW + ci * KH * KW + kh_i * KW + kw_i]; // w_zp=0
                end else begin
                  act_z = 0;
                  w_z   = weight_mem[co * CIN * KH * KW + ci * KH * KW + kh_i * KW + kw_i];
                end
                acc = acc + act_z * w_z;
              end
            end
          end
          acc = acc + bias_i32[co];
          fv = real'(acc) * INP_SCALE * wscale(co);
          qi = round_py3(fv / OUT_SCALE);
          pre_silu = qi + OUT_ZP;
          if (pre_silu < 0) pre_silu = 0;
          if (pre_silu > 255) pre_silu = 255;
          sv_ref[co * HOUT * WOUT + oh * WOUT + ow] = silu_lut[pre_silu];
        end
      end
      if (co % 4 == 0)
        $display("    channel %0d/%0d done", co, COUT);
    end
    $display("  L0 reference complete.");
  endtask

  task automatic compare_outputs(input int total_elements, output int errors);
    errors = 0;
    for (int i = 0; i < total_elements; i++) begin
      if (py_golden[i] !== sv_ref[i]) begin
        if (errors < 20)
          $display("  MISMATCH [%0d]: PY_HEX=%0d SV_REF=%0d", i, py_golden[i], sv_ref[i]);
        errors = errors + 1;
      end
    end
  endtask

  int total_errors;

  initial begin
    $display("================================================================");
    $display(" PHASE 4: Single Layer L0 (SV ref == verify_conv_layer.py math)");
    $display("================================================================");

    rst_n = 0;
    repeat (10) @(posedge clk);
    rst_n = 1;

    $display("\n=== Loading hex (layer_by_layer + bias) ===");
    $display("  (input = act_L00_input.hex — same as Python verify_conv)");
    load_int8_hex(G_INPUT, input_mem, INPUT_SIZE);
    load_int8_hex(G_WEIGHT, weight_mem, WEIGHT_SIZE);
    load_bias_int32_hex(G_BIAS);
    $readmemh(G_SILU_QUANT, silu_lut);
    $display("  Loaded SiLU LUT from %s (Python-quant-domain, matches verify_conv_layer)", G_SILU_QUANT);

    $display("\n=== SV reference conv ===");
    compute_l0_like_python();

    $display("\n=== Loading Python golden act_L00_output.hex ===");
    load_int8_hex(G_PY_OUT, py_golden, OUTPUT_SIZE);

    $display("\n=== Compare PY golden vs SV reference ===");
    compare_outputs(OUTPUT_SIZE, total_errors);

    $display("\n================================================================");
    if (total_errors == 0)
      $display("  [PASS] LAYER 0 bit-exact (SV ref matches exported hex)");
    else
      $display("  [FAIL] %0d mismatches / %0d (check scales in TB vs layer_summary.json)", total_errors, OUTPUT_SIZE);
    $display("================================================================\n");

    $finish;
  end

endmodule
