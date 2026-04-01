`timescale 1ns/1ps
package csr_pkg;
  parameter int CSR_CTRL          = 12'h000;
  parameter int CSR_STATUS        = 12'h004;
  parameter int CSR_VERSION       = 12'h008;
  parameter int CSR_CAP0          = 12'h00C;
  parameter int CSR_NET_DESC_LO   = 12'h010;
  parameter int CSR_NET_DESC_HI   = 12'h014;
  parameter int CSR_LAYER_START   = 12'h018;
  parameter int CSR_LAYER_END     = 12'h01C;
  parameter int CSR_IRQ_MASK      = 12'h020;
  parameter int CSR_PERF_CTRL     = 12'h030;
  parameter int CSR_PERF_CYCLE_LO = 12'h034;
  parameter int CSR_PERF_CYCLE_HI = 12'h038;
  parameter int CSR_PERF_TILE_DONE= 12'h03C;
  parameter int CSR_PERF_STALL    = 12'h040;
  parameter int CSR_BARRIER_STATUS= 12'h050;
  parameter int CSR_DBG_LAYER_ID  = 12'h060;
  parameter int CSR_DBG_TILE_ID   = 12'h064;
endpackage
