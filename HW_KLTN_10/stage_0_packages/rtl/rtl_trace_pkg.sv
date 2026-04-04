// ============================================================================
// Package: rtl_trace_pkg
// Cycle-accurate RTL trace to a single text file (simulation only).
// Enable compile:  xvlog ... -d RTL_TRACE
// Open file once from testbench:  rtl_trace_pkg::rtl_trace_open("path.log");
// Close before $finish:            rtl_trace_pkg::rtl_trace_close();
//
// Each line:  <time_ps>\t<tag>\t<message>
// ============================================================================
`timescale 1ns / 1ps

package rtl_trace_pkg;
  // Vivado may return a multichannel handle that is negative when stored in signed int;
  // use an explicit "open" flag instead of comparing the handle to > 0.
  integer      rtl_trace_fd = 0;
  bit          rtl_trace_active = 1'b0;

  task automatic rtl_trace_open(input string path);
    if (!rtl_trace_active) begin
      rtl_trace_fd = $fopen(path, "w");
      if (rtl_trace_fd == 0) begin
        $display("RTL_TRACE: FATAL could not open %s", path);
        rtl_trace_active = 1'b0;
      end else begin
        rtl_trace_active = 1'b1;
        $fdisplay(rtl_trace_fd, "# rtl_cycle_trace time_ps\ttag\tmessage");
      end
    end
  endtask

  function automatic void rtl_trace_line(input string tag, input string msg);
    if (rtl_trace_active)
      $fdisplay(rtl_trace_fd, "%0t\t%s\t%s", $time, tag, msg);
  endfunction

  task automatic rtl_trace_close;
    if (rtl_trace_active) begin
      $fclose(rtl_trace_fd);
      rtl_trace_active = 1'b0;
      rtl_trace_fd     = 0;
    end
  endtask
endpackage
