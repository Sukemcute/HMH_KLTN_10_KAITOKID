# xsim wave add — Phase A (paste in xsim Tcl console after simulation loads design)
# Usage: source E:/KLTN_HMH_FINAL/SW_KLTN/PHASE_4/03_rtl_cosim/wave_phase_a.tcl

set tb tb_golden_check
add_wave_divider "Phase A — Fetch / GS / AXI"
add_wave /${tb}/clk
add_wave /${tb}/rst_n
add_wave /${tb}/u_dut/u_ctrl/u_fetch/state
add_wave /${tb}/u_dut/u_ctrl/u_sched/state
add_wave /${tb}/u_dut/u_ctrl/fetched_tile_valid
add_wave /${tb}/u_dut/u_ctrl/fetched_tile_ready
add_wave /${tb}/u_dut/u_ctrl/u_fetch/layer_id
add_wave /${tb}/u_dut/u_ctrl/u_fetch/tile_cnt
add_wave /${tb}/u_dut/u_ctrl/u_fetch/tile_total
add_wave_divider "AXI read (TB ↔ DUT)"
add_wave /${tb}/m_axi_arvalid
add_wave /${tb}/m_axi_arready
add_wave /${tb}/m_axi_araddr
add_wave /${tb}/m_axi_rvalid
add_wave /${tb}/m_axi_rready
add_wave /${tb}/m_axi_rlast
