// ============================================================================
// Package: stage8_pkg — Stage 8 checkpoints & SW-aligned reference nuggets
// Ref: FUNCTION/SW_KLTN/documentation (INT8, ZP_hw = ZP_torch - 128, ReLU)
// ============================================================================
`timescale 1ns / 1ps

package stage8_pkg;
  import accel_pkg::*;

  // ═══════════════════════════════════════════════════════════════════
  // Checkpoint: tag + time (compare waves with Python golden offline)
  // ═══════════════════════════════════════════════════════════════════
  task automatic checkpoint(input string tag);
    $display("[STAGE8_CP] %0t | %s", $time, tag);
  endtask

  // ═══════════════════════════════════════════════════════════════════
  // RULE 2: half-up requant (bias already in INT32 domain)
  // ═══════════════════════════════════════════════════════════════════
  function automatic int32_t half_up_requant(
    input int64_t acc,
    input uint32_t m_int,
    input logic [7:0] sh
  );
    int64_t prod;
    int64_t rnd;
    prod = acc * int64_t'(unsigned'(m_int));
    if (sh == 0)
      return int32_t'(prod);
    rnd = 64'sd1 <<< (sh - 1);
    return int32_t'((prod + rnd) >>> sh);
  endfunction

  // RULE 4 + RULE 10: ReLU then add zp_out, clamp INT8
  function automatic int8_t relu_zp_clamp(input int32_t x, input int8_t zp);
    int32_t v;
    v = (x > 0) ? x : 32'sd0;
    v = v + 32'signed'({{24{zp[7]}}, zp});
    if (v > 127)
      return 8'sd127;
    if (v < -128)
      return -8'sd128;
    return int8_t'(v);
  endfunction

  // Signed INT8 max of 25 entries (MP5)
  function automatic int8_t max25(input int8_t v[25]);
    int8_t m;
    m = v[0];
    for (int i = 1; i < 25; i++)
      if (v[i] > m)
        m = v[i];
    return m;
  endfunction

endpackage
