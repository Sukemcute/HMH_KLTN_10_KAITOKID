"""
1) In kiến trúc chi tiết từng layer của model quantization (PTQ).
2) Mô tả flow inference: input -> preprocess -> model forward -> postprocess -> output.

Chạy (trong thư mục project, với .venv đã kích hoạt):
  python quant_model_architecture_and_flow.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Ultralytics-dev"))

import torch

BASE_WEIGHTS = ROOT / "Ultralytics-dev" / "ultralytics" / "qyolov10n.yaml"
QUANT_STATE_DICT = ROOT / "Ultralytics-dev" / "ultralytics" / "quant" / "quant_state_dict" / "qat_sttd.pt"


def _is_quantized_module(m: torch.nn.Module) -> bool:
    """Kiểm tra module có phải dạng quantized (PyTorch) không."""
    name = type(m).__name__
    return "Quantized" in name or "quantized" in name


def _param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _submodule_summary(m: torch.nn.Module, indent: str = "    ") -> str:
    """Tóm tắt submodules (conv, act, ...) nếu có."""
    lines = []
    for name, child in m.named_children():
        p = _param_count(child)
        q = " [quantized]" if _is_quantized_module(child) else ""
        lines.append(f"{indent}{name}: {type(child).__name__} (params={p}){q}")
    return "\n".join(lines) if lines else f"{indent}(no children)"


def print_quantized_model_architecture(model):
    """
    In ra kiến trúc model quantization từng layer một cách chi tiết.
    """
    net = model.model  # DetectionModel (nn.Module)
    layers_container = getattr(net, "model", None)
    # parse_model trả về nn.Sequential(*layers), không phải ModuleList
    if layers_container is None or not hasattr(layers_container, "__len__"):
        print("Model không có .model (Sequential/ModuleList).")
        print(f"  type(model.model) = {type(net).__name__}")
        if hasattr(net, "model"):
            print(f"  type(model.model.model) = {type(getattr(net, 'model', None)).__name__}")
        return

    layers = list(layers_container)
    print("=" * 70)
    print("KIẾN TRÚC MODEL QUANTIZATION (qYOLOv10n) - TỪNG LAYER")
    print("=" * 70)
    print(f"Tổng số layers: {len(layers)}")
    print(f"Stride: {getattr(net, 'stride', 'N/A')}")
    print(f"Names (classes): {getattr(net, 'names', 'N/A')}")
    print()

    for i, m in enumerate(layers):
        from_idx = getattr(m, "f", None)
        from_str = f"from layer(s) {from_idx}" if from_idx != -1 else "from previous"
        is_save = getattr(m, "i", i) in getattr(net, "save", [])
        layer_type = getattr(m, "type", type(m).__name__)

        print(f"[Layer {i:3d}] {type(m).__name__}")
        print(f"    type      : {layer_type}")
        print(f"    input     : {from_str}")
        print(f"    save_out  : {is_save}")
        print(f"    params    : {_param_count(m):,}")
        print(f"    quantized : {_is_quantized_module(m)}")

        # Chi tiết submodules (conv, bn, act, ...)
        sub = _submodule_summary(m)
        if sub and "no children" not in sub:
            print(f"    submodules:")
            print(sub)
        print()
    print("=" * 70)


def _dtype_to_int_or_float(dtype) -> str:
    """Trả về 'INT' nếu dtype là quantized (quint8/qint8), else 'Float'."""
    if dtype is None:
        return "?"
    if dtype in (torch.quint8, torch.qint8, torch.qint32):
        return "INT"
    return "Float"


def _tensor_dtype_str(t) -> str:
    """Lấy chuỗi mô tả dtype của tensor hoặc torch.dtype."""
    if t is None:
        return "?"
    if isinstance(t, torch.Tensor):
        return str(t.dtype).replace("torch.", "")
    if isinstance(t, torch.dtype):
        return str(t).replace("torch.", "")
    if isinstance(t, (list, tuple)) and t:
        return _tensor_dtype_str(t[0])
    return "?"


def _tensor_shape_str(t) -> str:
    """Lấy shape dạng chuỗi."""
    if t is None:
        return "?"
    if isinstance(t, torch.Tensor):
        return str(tuple(t.shape))
    if isinstance(t, (list, tuple)) and t:
        return _tensor_shape_str(t[0])
    return "?"


def trace_and_print_layer_dtypes(model, imgsz=640):
    """
    Chạy 1 lần forward với input giả, ghi lại dtype (INT/Float) và shape ở đầu vào/ra
    từng layer, rồi in bảng: dữ liệu biến đổi thế nào qua từng layer đến output.
    """
    net = model.model
    layers_container = getattr(net, "model", None)
    if layers_container is None or not hasattr(layers_container, "__len__"):
        print("Không có danh sách layers để trace.")
        return
    layers = list(layers_container)

    # Nơi lưu (layer_idx, input_dtype, output_dtype, output_shape) khi hook gọi
    trace_list = []

    def make_hook(idx):
        def hook(module, input, output):
            inp = input[0] if input else None
            if isinstance(inp, (list, tuple)):
                inp = inp[0] if inp else None
            in_dtype = inp.dtype if isinstance(inp, torch.Tensor) else None
            out = output
            if isinstance(output, (list, tuple)):
                out = output[0] if output else None
            out_dtype = out.dtype if isinstance(out, torch.Tensor) else None
            out_shape = out.shape if isinstance(out, torch.Tensor) else str(type(output))
            if isinstance(out_shape, torch.Size):
                out_shape = tuple(out_shape)
            trace_list.append((idx, in_dtype, out_dtype, out_shape))
        return hook

    handles = []
    for i, m in enumerate(layers):
        h = m.register_forward_hook(make_hook(i))
        handles.append(h)

    # Forward với input giả (float32)
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    try:
        with torch.no_grad():
            _ = net(dummy)
    except Exception as e:
        print(f"Forward trace bị lỗi: {e}")
        for h in handles:
            h.remove()
        return
    for h in handles:
        h.remove()

    # Output của QuantStub = input của layer 0
    quant_out_dtype = getattr(net.quant(dummy), "dtype", torch.quint8)

    print()
    print("=" * 100)
    print("LUỒNG DỮ LIỆU QUA TỪNG LAYER: INPUT → OUTPUT (dtype và INT/Float)")
    print("=" * 100)
    print(f"{'Layer':<6} {'Module':<22} {'Input dtype':<14} {'Output dtype':<14} {'Tính toán':<8} {'Output shape'}")
    print("-" * 100)

    for i, m in enumerate(layers):
        if i >= len(trace_list):
            break
        idx, in_dtype, out_dtype, out_shape = trace_list[i]
        if idx != i:
            continue
        # Input của layer 0 là output của QuantStub
        if i == 0:
            in_dtype = in_dtype or quant_out_dtype
        in_str = _tensor_dtype_str(in_dtype) if in_dtype is not None else "?"
        out_str = _tensor_dtype_str(out_dtype) if out_dtype is not None else "?"
        kind = _dtype_to_int_or_float(out_dtype) if out_dtype is not None else "?"
        shape_str = str(out_shape) if out_shape is not None else "?"
        if len(shape_str) > 32:
            shape_str = shape_str[:29] + "..."
        print(f"{i:<6} {type(m).__name__:<22} {in_str:<14} {out_str:<14} {kind:<8} {shape_str}")

    print("-" * 100)
    print("  Chú thích: INT = tính toán trên số nguyên (quint8/qint8); Float = float32/float16.")
    print("=" * 100)


def print_int_vs_float_inference():
    """
    Giải thích inference được tính trên INT hay Float.
    """
    text = r"""
================================================================================
INFERENCE TÍNH TRÊN INT HAY FLOAT? (Model Quantization PTQ)
================================================================================

  • Phần lớn inference trong model quantize được TÍNH TOÁN TRÊN SỐ NGUYÊN (INT8).
  • Một số bước vẫn dùng FLOAT (để chuẩn hóa, hoặc op chưa có kernel quantized).

  Chi tiết theo từng giai đoạn:
  ───────────────────────────────────────────────────────────────────────────

  1. PREPROCESS
     → FLOAT (float32): ảnh normalize 0–1, tensor (N,3,H,W) float32.

  2. ĐẦU VÀO MODEL: QuantStub
     → FLOAT → INT:  x_float → quantize (scale, zero_point) → tensor QUINT8 (INT8 không dấu).
     → Từ đây activations trong model chủ yếu là INT8.

  3. CÁC LỚP QUANTIZED (Conv, Linear, ...)
     → TÍNH TRÊN INT: trọng số đã là INT8, input/output activations cũng INT8.
     → PyTorch dùng kernel QuantizedCPU: nhân-cộng thực tế là số nguyên (int8),
       sau đó scale/zero_point chỉ dùng để “giải thích” giá trị (không tính lại từng bước).

  4. ACTIVATION SiLU (đã sửa trong conv.py)
     → FLOAT tạm thời: vì PyTorch không có SiLU cho QuantizedCPU, ta làm:
       dequant (INT→float) → SiLU (float) → requant (float→INT).
     → Phần lớn thời gian vẫn ở INT; chỉ đoạn SiLU chạy float rồi lại về INT.

  5. HEAD (Qv10Detect) VÀ RA MODEL
     → Có thể vẫn quantized (INT) đến cuối, rồi DeQuantStub (nếu có) đưa về float.
     → Postprocess (NMS, scale box) chạy trên FLOAT.

  TÓM TẮT:
  ───────────────────────────────────────────────────────────────────────────
  • Phần “nặng” (Conv, phần lớn backbone/neck): TÍNH TOÁN TRÊN INT (INT8).
  • Preprocess, SiLU (đoạn nhỏ đã patch), NMS và output: FLOAT.
  • Mục đích quantization: giảm bộ nhớ và tăng tốc nhờ tính toán INT thay vì float.

================================================================================
"""
    print(text)


def print_inference_flow():
    """
    In ra flow inference: từ khi nhận input đến output.
    """
    flow = r"""
================================================================================
FLOW INFERENCE - TỪ INPUT ĐẾN OUTPUT (Model Quantization PTQ)
================================================================================

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 1. INPUT                                                                  │
  │    • source: đường dẫn ảnh (str/Path) hoặc thư mục / list ảnh / tensor   │
  │    • Ví dụ: "img1.jpg" hoặc Path("D:/SW_KLTN/img1.jpg")                   │
  └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 2. SETUP SOURCE (Predictor.setup_source)                                 │
  │    • check_imgsz() → kích thước inference (mặc định 640, stride 32)      │
  │    • load_inference_source() → tạo dataset iterator (ảnh/video)          │
  │    • LetterBox: chuẩn bị letterbox cho từng ảnh                          │
  └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 3. PREPROCESS (mỗi batch: paths, im0s, s)                                │
  │    • pre_transform(im0s): LetterBox(imgsz, stride) → resize + pad        │
  │    • BGR → RGB (nếu cần)                                                 │
  │    • HWC → CHW (batch, 3, H, W)                                          │
  │    • numpy → torch.Tensor, .to(device), .float()                         │
  │    • im /= 255  (normalize 0–1)                                          │
  │    Output: im shape (N, 3, H, W) float32, device=cpu                      │
  └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 4. MODEL FORWARD (Quantized)                                             │
  │    BaseModel.forward → predict() → _predict_once_quantized(x)            │
  │                                                                          │
  │    a) QuantStub:  x = self.quant(x)   # float → quantized (quint8)       │
  │    b) Backbone/Neck (lặp từng layer trong model.model):                  │
  │       • Nếu m.f != -1: lấy input từ layer trước (y[m.f] hoặc concat)      │
  │       • x = m(x)  # Conv, C2f, SPPF, ... (Conv đã fuse: conv+act;       │
  │                   #  act với quantized: dequant→SiLU→requant)            │
  │       • Lưu output vào y[] nếu m.i in self.save                          │
  │    c) Head (Qv10Detect): x = m(x) → raw predictions (boxes, scores)      │
  │    Output: tensor raw predictions (batch, num_anchors, 4+nc)             │
  └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 5. POSTPROCESS (DetectionPredictor.postprocess)                           │
  │    • non_max_suppression(preds, conf, iou, classes, max_det, ...)        │
  │      → loại bỏ box trùng, giữ theo conf/iou                              │
  │    • Scale box về tọa độ ảnh gốc (từ letterbox)                           │
  │    • Tạo Results: boxes (xyxy, conf, cls), names, orig_shape              │
  └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ 6. OUTPUT                                                                │
  │    • results: list[Results] (mỗi ảnh 1 Results)                          │
  │    • result.boxes: xyxy, conf, cls (tọa độ, độ tin cậy, class id)         │
  │    • result.names: dict id → tên class                                    │
  │    • result.save(filename=...): vẽ box lên ảnh và lưu file                │
  └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""
    print(flow)


def main():
    print_int_vs_float_inference()
    print_inference_flow()

    if not BASE_WEIGHTS.exists() or not QUANT_STATE_DICT.exists():
        print("Thiếu file config/weights. Bỏ qua in kiến trúc.")
        return

    try:
        from ultralytics.quant import load_ptq_model_from_state_dict
    except Exception as e:
        print(f"Không load được ultralytics (cần .venv và pip install -e Ultralytics-dev): {e}")
        return

    print("\nĐang load model PTQ để in kiến trúc và trace dtype...")
    model = load_ptq_model_from_state_dict(
        base_weights=str(BASE_WEIGHTS),
        quant_state_dict=str(QUANT_STATE_DICT),
    )
    print_quantized_model_architecture(model)
    trace_and_print_layer_dtypes(model)


if __name__ == "__main__":
    main()
