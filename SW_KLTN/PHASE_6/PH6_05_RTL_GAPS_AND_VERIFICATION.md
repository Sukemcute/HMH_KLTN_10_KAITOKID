# PH6-05 — Khoảng trống RTL so với HW_MAPPING & kế hoạch verify

So sánh tên trong `HW_MAPPING_RESEARCH.md` §8 với **file thực tế PHASE_3**.

---

## 1. Tên khối trong mapping vs PHASE_3

| Mapping (tên lý tưởng) | PHASE_3 thực tế | Hành động |
|------------------------|-----------------|-----------|
| pe_lane_mac | `pe_unit.sv` + `dsp_pair_int8.sv` | Giữ; mở rộng LANES theo V2. |
| pool_engine | `comparator_tree.sv` (trong `pe_cluster`) | Đảm bảo window 5×5 pad=2 s=1 đủ cho SPPF. |
| ppu_lite | `ppu.sv` | Đủ chức năng “lite”; thêm nạp param từ DDR nếu cần. |
| tensor_post_engine | `swizzle_engine` + DMA `ext_*` | Không tách file riêng; chức năng MOVE/upsample gộp. |
| gemm_attn_engine | **Chưa có module riêng** | **P9 QPSA:** dùng `PE_GEMM` + điều khiển ngoài hoặc thêm `gemm_attn_ctrl.sv`; softmax = LUT/phân đoạn — **block rủi ro cao nhất**. |
| row_slot_manager | Logic trong `addr_gen_*` / metadata | Có thể tách sau để debug. |

---

## 2. subcluster_wrapper — trạng thái tích hợp

File ghi chú **behavioral cosim** vs spec đầy đủ GLB→PE→PPU. **Để inference đúng bit-exact với PTQ**, cần:

1. Nối `glb_*`, `addr_gen_*`, `router_cluster`, `window_gen`, `pe_cluster`, `ppu`, `swizzle_engine` theo `PH6_02`.  
2. Loại bỏ / thay buffer conv tạm bằng đường chuẩn.

---

## 3. global_scheduler & dual-RUNNING (V2)

`HW_ARCHITECTURE_V2_100FPS.md` yêu cầu **2 sub RUNNING / SC**. Kiểm tra `local_arbiter`, `accel_pkg.ACTIVE_PER_SC`, và occupancy thực tế — nếu RTL vẫn 1 active, **FPS không đạt** mục tiêu spreadsheet.

---

## 4. Thứ tự verify (khuyến nghị)

1. **Primitive đơn:** RS3 s1/s2, OS1, DW3, MP5, upsample, concat (cùng scale).  
2. **Quant:** requant, concat khác scale.  
3. **DW7 multipass** ≡ golden monolithic 7×7.  
4. **Block:** QC2f nhỏ, SCDown, SPPF.  
5. **Full L0–22** trace INT8 → so P3/P4/P5 với PyTorch quantized.

---

## 5. Checklist sign-off (trích ý HW_MAPPING §9)

- [ ] `test_primitives` / tương đương cosim RTL  
- [ ] `test_quant` / concat domain  
- [ ] Layout bank không overlap  
- [ ] `test_model_forward`: shapes P3/P4/P5 + metadata  

---

*PHASE_6 là tài liệu; hoàn thành RTL là công việc trên nhánh PHASE_3 / PHASE_5.*
