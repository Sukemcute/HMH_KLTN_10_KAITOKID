`timescale 1ns/1ps
// =============================================================================
// PHASE_5 — accel_top (bản phát triển)
//
// Khác PHASE_3: không có cổng input ppu_* trên top. Tham số PPU nằm trong bank nội bộ;
// sau này nạp qua cùng không gian DDR / cùng master AXI4 (module loader + mux — TODO).
//
// CSR gợi ý: CSR_PPU_TABLE_LO/HI, CSR_PPU_CTRL, CSR_PPU_STATUS trong PHASE_5 csr_pkg.sv
// (controller hiện tại vẫn ở PHASE_3 thì chưa decode — copy/sửa controller khi cần).
//
// Ghép compile: thêm PHASE_5/packages/*.sv và RTL PHASE_3 (controller_system,
// supercluster_wrapper, tensor_dma, …). Chỉ được một bản csr_pkg/desc_pkg/accel_pkg trong
// lệnh compile (ưu tiên PHASE_5/packages khi dùng top này).
// =============================================================================
module accel_top (
  input  logic        clk,
  input  logic        rst_n,

  input  logic [11:0] s_axil_awaddr,
  input  logic        s_axil_awvalid,
  output logic        s_axil_awready,
  input  logic [31:0] s_axil_wdata,
  input  logic        s_axil_wvalid,
  output logic        s_axil_wready,
  output logic [1:0]  s_axil_bresp,
  output logic        s_axil_bvalid,
  input  logic        s_axil_bready,
  input  logic [11:0] s_axil_araddr,
  input  logic        s_axil_arvalid,
  output logic        s_axil_arready,
  output logic [31:0] s_axil_rdata,
  output logic        s_axil_rvalid,
  input  logic        s_axil_rready,

  output logic [39:0]  m_axi_araddr,
  output logic [7:0]   m_axi_arlen,
  output logic         m_axi_arvalid,
  input  logic         m_axi_arready,
  input  logic [255:0] m_axi_rdata,
  input  logic         m_axi_rvalid,
  input  logic         m_axi_rlast,
  output logic         m_axi_rready,
  output logic [39:0]  m_axi_awaddr,
  output logic [7:0]   m_axi_awlen,
  output logic         m_axi_awvalid,
  input  logic         m_axi_awready,
  output logic [255:0] m_axi_wdata,
  output logic         m_axi_wvalid,
  output logic         m_axi_wlast,
  input  logic         m_axi_wready,
  input  logic [1:0]   m_axi_bresp,
  input  logic         m_axi_bvalid,
  output logic         m_axi_bready,

  output logic         irq
);

  import accel_pkg::*;
  import desc_pkg::*;

  //###########################################################################
  //  [PPU] Bank tham số nội bộ — sau nạp từ DDR qua AXI4 master (TODO loader + mux)
  //###########################################################################
  logic signed [31:0] ppu_bias    [32];
  logic signed [31:0] ppu_m_int   [32];
  logic        [5:0]  ppu_shift   [32];
  logic signed [7:0]  ppu_zp_out;
  logic signed [7:0]  ppu_silu_lut [256];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ppu_zp_out <= '0;
      for (int i = 0; i < 32; i++) begin
        ppu_bias[i]   <= '0;
        ppu_m_int[i]  <= '0;
        ppu_shift[i]  <= '0;
      end
      for (int j = 0; j < 256; j++)
        ppu_silu_lut[j] <= '0;
    end
    // TODO: cập nhật bank khi FSM/DMA đọc xong blob tại CSR_PPU_TABLE_* (hoặc từ net aux)
  end

  // TODO: instance ppu_param_loader / arbiter thêm vào mux m_axi_* bên dưới





  //###########################################################################
  //  [MMIO] AXI-Lite slave — giao tiếp CPU (decode AW/W/AR → mmio_we / mmio_re)
  //###########################################################################
  logic        mmio_we, mmio_re;
  logic [11:0] mmio_addr_wr, mmio_addr_rd;
  logic [31:0] mmio_wdata_i, mmio_rdata_o;

  logic aw_done, w_done;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_done      <= 1'b0;
      w_done       <= 1'b0;
      mmio_addr_wr <= '0;
      mmio_wdata_i <= '0;
    end else begin
      if (s_axil_awvalid && s_axil_awready) begin
        mmio_addr_wr <= s_axil_awaddr;
        aw_done      <= 1'b1;
      end
      if (s_axil_wvalid && s_axil_wready) begin
        mmio_wdata_i <= s_axil_wdata;
        w_done       <= 1'b1;
      end
      if (aw_done && w_done) begin
        aw_done <= 1'b0;
        w_done  <= 1'b0;
      end
    end
  end

  assign s_axil_awready = !aw_done;
  assign s_axil_wready  = !w_done;
  assign mmio_we        = aw_done && w_done;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      s_axil_bvalid <= 1'b0;
    else if (mmio_we)
      s_axil_bvalid <= 1'b1;
    else if (s_axil_bready)
      s_axil_bvalid <= 1'b0;
  end
  assign s_axil_bresp = 2'b00;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mmio_addr_rd   <= '0;
      s_axil_rvalid  <= 1'b0;
    end else begin
      if (s_axil_arvalid && s_axil_arready) begin
        mmio_addr_rd  <= s_axil_araddr;
        s_axil_rvalid <= 1'b1;
      end else if (s_axil_rready)
        s_axil_rvalid <= 1'b0;
    end
  end
  assign s_axil_arready = !s_axil_rvalid;
  assign mmio_re        = s_axil_arvalid && s_axil_arready;
  wire [11:0] mmio_addr_mux = mmio_we ? mmio_addr_wr
                           : ((s_axil_arvalid && s_axil_arready) ? s_axil_araddr : mmio_addr_rd);
  logic [31:0] s_axil_rdata_latched;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      s_axil_rdata_latched <= '0;
    else if (s_axil_arvalid && s_axil_arready)
      s_axil_rdata_latched <= mmio_rdata_o;
  end
  assign s_axil_rdata   = s_axil_rdata_latched;




  //###########################################################################
  //  [CTRL] Instance: controller_system — CSR + desc_fetch + scheduler + barrier
  //###########################################################################
  tile_desc_t  ctrl_sc_tile [4];
  layer_desc_t ctrl_sc_layer;
  logic        ctrl_sc_tile_valid [4];
  logic        ctrl_sc_tile_accept [4];
  logic        ctrl_barrier_sig [4];
  logic [4:0]  ctrl_barrier_id [4];
  logic [31:0] ctrl_barrier_scoreboard;

  logic [39:0] ctrl_axi_araddr;
  logic [7:0]  ctrl_axi_arlen;
  logic        ctrl_axi_arvalid, ctrl_axi_arready_i;
  logic        ctrl_axi_rready;
  logic        ctrl_rd_busy;

  controller_system u_ctrl (
    .clk                 (clk),
    .rst_n               (rst_n),
    .mmio_addr           (mmio_addr_mux),
    .mmio_wdata          (mmio_wdata_i),
    .mmio_we             (mmio_we),
    .mmio_re             (mmio_re),
    .mmio_rdata          (mmio_rdata_o),
    .irq                 (irq),
    .axi_araddr          (ctrl_axi_araddr),
    .axi_arlen           (ctrl_axi_arlen),
    .axi_arvalid         (ctrl_axi_arvalid),
    .axi_arready         (ctrl_axi_arready_i),
    .axi_rdata           (m_axi_rdata),
    .axi_rvalid          (m_axi_rvalid),
    .axi_rlast           (m_axi_rlast),
    .axi_rready          (ctrl_axi_rready),
    .sc_tile             (ctrl_sc_tile),
    .sc_layer_desc       (ctrl_sc_layer),
    .sc_tile_valid       (ctrl_sc_tile_valid),
    .sc_tile_accept      (ctrl_sc_tile_accept),
    .barrier_signal_in   (ctrl_barrier_sig),
    .barrier_signal_id_in(ctrl_barrier_id),
    .barrier_scoreboard  (ctrl_barrier_scoreboard)
  );

  wire ctrl_ar_hs  = ctrl_axi_arvalid && m_axi_arready;
  wire ctrl_r_done = ctrl_rd_busy && m_axi_rvalid && m_axi_rready && m_axi_rlast;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      ctrl_rd_busy <= 1'b0;
    else begin
      if (ctrl_r_done)
        ctrl_rd_busy <= 1'b0;
      else if (ctrl_ar_hs)
        ctrl_rd_busy <= 1'b1;
    end
  end




  //###########################################################################
  //  [SC×4] Instance: supercluster_wrapper — DMA tensor + compute; PPU ← bank nội bộ
  //###########################################################################
  logic [39:0] sc_araddr [4];
  logic [7:0]  sc_arlen [4];
  logic        sc_arvalid [4];
  logic [39:0] sc_awaddr [4];
  logic [7:0]  sc_awlen [4];
  logic        sc_awvalid [4];
  logic [255:0] sc_wdata [4];
  logic        sc_wvalid [4], sc_wlast [4];
  logic        sc_layer_done [4];
  logic [15:0] sc_tiles_completed [4];

  logic [1:0] axiw_grant;
  logic       axiw_busy;
  logic [1:0] axiw_pick;
  logic       aw_idle_handshake;
  logic [3:0] sc_aw_first;

  always_comb begin
    axiw_pick = '0;
    for (int j = 0; j < 4; j++) begin
      if (sc_awvalid[j]) begin
        axiw_pick = j[1:0];
        break;
      end
    end
    for (int g = 0; g < 4; g++) begin
      sc_aw_first[g] = sc_awvalid[g];
      for (int h = 0; h < g; h++)
        if (sc_awvalid[h])
          sc_aw_first[g] = 1'b0;
    end
    aw_idle_handshake = !axiw_busy && m_axi_awready && sc_awvalid[axiw_pick];
  end

  wire axiw_b_done = axiw_busy && m_axi_bvalid && m_axi_bready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      axiw_busy  <= 1'b0;
      axiw_grant <= '0;
    end else begin
      if (axiw_b_done)
        axiw_busy <= 1'b0;
      else if (aw_idle_handshake) begin
        axiw_busy  <= 1'b1;
        axiw_grant <= axiw_pick;
      end
    end
  end

  logic [1:0] axir_grant;
  logic       axir_busy;
  logic [1:0] axir_pick;
  logic       ar_idle_handshake;
  logic [3:0] sc_ar_first;
  logic       sc_m_axi_rready [4];
  wire        axir_r_done = axir_busy && m_axi_rvalid && m_axi_rready && m_axi_rlast;

  always_comb begin
    axir_pick = '0;
    for (int jr = 0; jr < 4; jr++) begin
      if (sc_arvalid[jr]) begin
        axir_pick = jr[1:0];
        break;
      end
    end
    for (int gr = 0; gr < 4; gr++) begin
      sc_ar_first[gr] = sc_arvalid[gr];
      for (int hr = 0; hr < gr; hr++)
        if (sc_arvalid[hr])
          sc_ar_first[gr] = 1'b0;
    end
    ar_idle_handshake = !axir_busy && m_axi_arready && sc_arvalid[axir_pick];
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      axir_busy  <= 1'b0;
      axir_grant <= '0;
    end else begin
      if (axir_r_done)
        axir_busy <= 1'b0;
      else if (ar_idle_handshake && !ctrl_axi_arvalid && !ctrl_rd_busy) begin
        axir_busy  <= 1'b1;
        axir_grant <= axir_pick;
      end
    end
  end

  genvar gi;
  generate
    for (gi = 0; gi < 4; gi++) begin : gen_sc
      supercluster_wrapper #(
        .NUM_SUBS(4),
        .LANES(32)
      ) u_sc (
        .clk               (clk),
        .rst_n             (rst_n),
        .tile_in           (ctrl_sc_tile[gi]),
        .layer_desc        (ctrl_sc_layer),
        .tile_valid        (ctrl_sc_tile_valid[gi]),
        .tile_accept       (ctrl_sc_tile_accept[gi]),
        .m_axi_araddr      (sc_araddr[gi]),
        .m_axi_arlen       (sc_arlen[gi]),
        .m_axi_arvalid     (sc_arvalid[gi]),
        .m_axi_arready     (m_axi_arready
                             && !ctrl_axi_arvalid && !ctrl_rd_busy
                             && ((!axir_busy && sc_ar_first[gi])
                                 || (axir_busy && (axir_grant == 2'(gi))))),
        .m_axi_rdata       (m_axi_rdata),
        .m_axi_rvalid      (m_axi_rvalid && ((axir_busy && (axir_grant == 2'(gi)))
                             || (ar_idle_handshake && (axir_pick == 2'(gi))))),
        .m_axi_rlast       (m_axi_rlast && ((axir_busy && (axir_grant == 2'(gi)))
                             || (ar_idle_handshake && (axir_pick == 2'(gi))))),
        .m_axi_rready      (sc_m_axi_rready[gi]),
        .m_axi_awaddr      (sc_awaddr[gi]),
        .m_axi_awlen       (sc_awlen[gi]),
        .m_axi_awvalid     (sc_awvalid[gi]),
        .m_axi_awready     (m_axi_awready
                             && ((!axiw_busy && sc_aw_first[gi])
                                 || (axiw_busy && (axiw_grant == 2'(gi))))),
        .m_axi_wdata       (sc_wdata[gi]),
        .m_axi_wvalid      (sc_wvalid[gi]),
        .m_axi_wlast       (sc_wlast[gi]),
        .m_axi_wready      (m_axi_wready && ((axiw_busy && (axiw_grant == 2'(gi)))
                             || (aw_idle_handshake && (axiw_pick == 2'(gi))))),
        .m_axi_bresp       (m_axi_bresp),
        .m_axi_bvalid      (m_axi_bvalid && axiw_busy && (axiw_grant == 2'(gi))),
        .m_axi_bready      (),
        .ppu_bias          (ppu_bias),
        .ppu_m_int         (ppu_m_int),
        .ppu_shift         (ppu_shift),
        .ppu_zp_out        (ppu_zp_out),
        .ppu_silu_lut      (ppu_silu_lut),
        .barrier_signal    (ctrl_barrier_sig[gi]),
        .barrier_signal_id (ctrl_barrier_id[gi]),
        .barrier_grant     (1'b1),
        .layer_done        (sc_layer_done[gi]),
        .tiles_completed   (sc_tiles_completed[gi])
      );
    end
  endgenerate




  //###########################################################################
  //  [AXI MUX] Ghép master: controller đọc desc  ||  SC×4 tensor_dma (TODO: nhánh PPU load)
  //###########################################################################
  always_comb begin
    m_axi_araddr  = ctrl_axi_araddr;
    m_axi_arlen   = ctrl_axi_arlen;
    m_axi_arvalid = ctrl_axi_arvalid;
    ctrl_axi_arready_i = m_axi_arready;
    m_axi_rready  = ctrl_axi_rready;

    m_axi_awaddr  = '0;
    m_axi_awlen   = '0;
    m_axi_awvalid = 1'b0;
    m_axi_wdata   = '0;
    m_axi_wvalid  = 1'b0;
    m_axi_wlast   = 1'b0;
    m_axi_bready  = 1'b1;

    if (ctrl_rd_busy) begin
      m_axi_araddr  = ctrl_axi_araddr;
      m_axi_arlen   = ctrl_axi_arlen;
      m_axi_arvalid = ctrl_axi_arvalid;
      ctrl_axi_arready_i = m_axi_arready;
      m_axi_rready  = ctrl_axi_rready;
    end else if (ctrl_ar_hs) begin
      m_axi_araddr  = ctrl_axi_araddr;
      m_axi_arlen   = ctrl_axi_arlen;
      m_axi_arvalid = ctrl_axi_arvalid;
      ctrl_axi_arready_i = m_axi_arready;
      m_axi_rready  = ctrl_axi_rready;
    end else if (m_axi_rvalid) begin
      m_axi_rready  = axir_busy ? sc_m_axi_rready[axir_grant] : 1'b1;
      if (axir_busy) begin
        m_axi_araddr  = sc_araddr[axir_grant];
        m_axi_arlen   = sc_arlen[axir_grant];
        m_axi_arvalid = sc_arvalid[axir_grant];
        ctrl_axi_arready_i = 1'b0;
      end else begin
        m_axi_araddr  = sc_araddr[axir_pick];
        m_axi_arlen   = sc_arlen[axir_pick];
        m_axi_arvalid = sc_arvalid[axir_pick];
        ctrl_axi_arready_i = 1'b0;
      end
    end else if (ctrl_axi_arvalid) begin
      m_axi_araddr  = ctrl_axi_araddr;
      m_axi_arlen   = ctrl_axi_arlen;
      m_axi_arvalid = ctrl_axi_arvalid;
      ctrl_axi_arready_i = m_axi_arready;
      m_axi_rready  = ctrl_axi_rready;
    end else begin
      if (axir_busy) begin
        m_axi_araddr  = sc_araddr[axir_grant];
        m_axi_arlen   = sc_arlen[axir_grant];
        m_axi_arvalid = sc_arvalid[axir_grant];
        ctrl_axi_arready_i = 1'b0;
        m_axi_rready  = sc_m_axi_rready[axir_grant];
      end else begin
        m_axi_araddr  = sc_araddr[axir_pick];
        m_axi_arlen   = sc_arlen[axir_pick];
        m_axi_arvalid = sc_arvalid[axir_pick];
        ctrl_axi_arready_i = 1'b0;
        m_axi_rready  = 1'b1;
      end
    end

    if (axiw_busy) begin
      m_axi_awaddr  = sc_awaddr[axiw_grant];
      m_axi_awvalid = sc_awvalid[axiw_grant];
      m_axi_awlen   = sc_awlen[axiw_grant];
      m_axi_wdata   = sc_wdata[axiw_grant];
      m_axi_wvalid  = sc_wvalid[axiw_grant];
      m_axi_wlast   = sc_wlast[axiw_grant];
    end else begin
      m_axi_awaddr  = sc_awaddr[axiw_pick];
      m_axi_awvalid = sc_awvalid[axiw_pick];
      m_axi_awlen   = sc_awlen[axiw_pick];
      m_axi_wdata   = sc_wdata[axiw_pick];
      m_axi_wvalid  = sc_wvalid[axiw_pick];
      m_axi_wlast   = sc_wlast[axiw_pick];
    end
  end

endmodule
