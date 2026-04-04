stage_100 — Cosim / golden-vector observation hub
================================================

Muc dich: mot cho duy nhat de xem .memh, log xsim/xvlog, va chay lai flow Stage 8 (+USE_GOLDEN_VECTORS).

Thu muc:
  work\
    run_stage100_golden.cmd   <- Chay tu day (generate Python + xvlog/xelab/xsim)
    vectors\<prim>\            <- File .memh do cosim_vector_gen.py ghi (input/weight/expected)
    compile_rtl.f             <- Tao lai moi lan chay script (khong can sua tay)
    xvlog.log, xsim_sim.log, xsim.dir\  <- Xuat hien sau khi chay sim

Cach chay:
  cd HW_KLTN_10\stage_100\work
  .\run_stage100_golden.cmd

Hoac tu stage_8_primitive_verify:
  .\run_stage8_golden.cmd     (wrapper goi stage_100\work\...)

Ghi chu:
  - RTL + testbench van nam o stage_8_primitive_verify (khong copy RTL vao day).
  - Stage 11 block vectors van vao stage_11_block_verify\generated\ (Python khong doi).
