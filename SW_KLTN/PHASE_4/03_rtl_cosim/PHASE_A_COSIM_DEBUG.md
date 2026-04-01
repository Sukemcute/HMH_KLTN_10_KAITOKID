# Giai đoạn A — Quan sát deadlock (co-sim `tb_golden_check`)

Mục tiêu: **chốt nhánh nguyên nhân** khi hệ thống im sau `LAYER_DESC layer_id=1` (hoặc bất kỳ chỗ kẹt nào), **không đoán mù**.

## Bật monitor

Trong `run_cosim.tcl`:

```tcl
set PHASE_A_DBG_DEFINE 1
```

Sau đó `source run_cosim.tcl` (xóa `xsim.dir` nếu cần), `xelab tb_golden_check -s sim_golden`, `xsim sim_golden -runall`.

Chỉ file `tb_golden_check.sv` được compile kèm `-d ACCEL_PHASE_A_DBG` (log giai đoạn A không làm ảnh hưởng RTL PHASE_3).

Tắt monitor: `set PHASE_A_DBG_DEFINE ""`.

## Các tag log

| Tag | Ý nghĩa |
|-----|--------|
| `[PHASE-A] DF_STATE` | `desc_fetch_engine` đổi state (chỉ in khi state đổi). |
| `[PHASE-A] GS_STATE` | `global_scheduler` đổi state. |
| `[PHASE-A] LAYER_ID` | `layer_id` của fetch tăng (ranh giới layer). |
| `[PHASE-A] STALL` | Cùng `DF` state ≥ `STALL_THRESHOLD` cycle liên tiếp → snapshot. |
| `[PHASE-A] TIMEOUT_SNAPSHOT` | Ngay khi TB báo TIMEOUT 10M cycle → một ảnh chụp tức thì. |

## Cách đọc `STALL` / `TIMEOUT_SNAPSHOT`

Mỗi dòng snapshot gồm:

- **DF** = state fetch (chuỗi), **tile_v/rdy** = `tile_desc_valid` / `tile_desc_ready` (handshake GS ↔ Fetch).
- **GS** = state scheduler, **sc_mask** từ tile đang giữ (tile_reg), **sc_disp** = tiến độ accept từng SC.
- **L/tc/tt** = `layer_id`, `tile_cnt`, `tile_total`.
- **AXI** = `m_axi_arvalid/arready`, `rvalid/rready/rlast` (kênh đọc nối TB ↔ DUT).
- **SC_acc** = `tile_accept` vào từng SuperCluster (1 = SC đang ack tile từ GS).

### Phân nhánh chẩn đoán (sau khi có 1 snapshot tại chỗ kẹt)

1. **DF = `DF_DISPATCH_TILE`, tile_v=1, tile_rdy=0** lâu  
   → Fetch đang chờ GS chấp nhận tile. Xem **GS**:  
   - Nếu **GS ≠ IDLE** (thường `GS_WAIT_ACCEPT`) → có thể kẹt **chưa đủ `sc_tile_accept`** từ một hoặc nhiều SC.  
   - Nếu **GS = IDLE** mà vẫn rdy=0 → lệch pha handshake (kiểm tra `tile_valid`/`tile_accept` cùng cycle).

2. **DF = `DF_FETCH_TILE_AR` hoặc `DF_FETCH_TILE_R`** lâu  
   → Fetch chờ đọc DDR descriptor tile. Xem **AXI**:  
   - `arvalid=1`, `arready=0` lâu → slave model / mux không grant (hiếm nếu TB model `arready` thường 1).  
   - `arvalid=0` → controller không đưa AR lên (logic fetch hoặc điều kiện trước đó).

3. **DF = `DF_FETCH_LAYER_*` / `DF_PARSE_LAYER`**  
   → Đang đọc/parse layer; nếu kẹt ở đây sau khi đã thấy layer 1 trong log thường ít gặp — vẫn ghi nhận AXI.

4. **GS = `GS_WAIT_ACCEPT`, sc_disp ≠ sc_mask**  
   → Một SC không bật `tile_accept`. Đối chiếu **SC_acc** và ingress/arbitration trong SC.

## Waveform (tùy chọn)

Trong GUI xsim, thêm tín hiệu (điều chỉnh hierarchy nếu cần):

- `tb_golden_check/u_dut/u_ctrl/u_fetch/state`
- `tb_golden_check/u_dut/u_ctrl/u_sched/state`
- `tb_golden_check/u_dut/u_ctrl/fetched_tile_valid`
- `tb_golden_check/u_dut/u_ctrl/fetched_tile_ready`
- `tb_golden_check/m_axi_arvalid`, `m_axi_arready`, `m_axi_rvalid`, `m_axi_rready`, `m_axi_rlast`

File gợi ý: `wave_phase_a.tcl` (trong thư mục này).

## Tham số trong TB

Trong `tb_golden_check.sv` (khối `ACCEL_PHASE_A_DBG`):

- `PHASE_A_STALL_THRESHOLD` (mặc định 10_000 cycle): sau số cycle này **cùng** `DF` state sẽ in một dòng `[PHASE-A] STALL`.

Giảm xuống 2000 nếu muốn báo sớm hơn (log dài hơn).

---

_Tài liệu này bổ sung `RTL_SYSTEM_COSIM_STATUS.md`._
