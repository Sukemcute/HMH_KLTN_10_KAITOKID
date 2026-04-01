
import numpy as np
import torch
import torch.nn as nn
import sys, os

# Set backend globally before imports that might trigger ops
if 'x86' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'x86'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blocks.block_conv import block_conv
from blocks.block_qc2f import block_qc2f
from blocks.block_scdown import block_scdown
from blocks.block_sppf import block_sppf
from blocks.block_qpsa import block_qpsa
from blocks.block_upsample import block_upsample
from blocks.block_concat import block_concat
from blocks.block_qc2f_cib import block_qc2f_cib

def to_signed_int8(uint8_numpy):
    return (uint8_numpy.astype(np.int16) - 128).astype(np.int8)

def get_conv_params(qconv_module, input_scale, input_zp, is_dw=False, is_3x3=False):
    qconv = qconv_module.conv
    w_q = qconv.weight()
    w_int8 = w_q.int_repr().numpy().view(np.int8)
    if is_dw: w_int8 = np.squeeze(w_int8, axis=1)
    w_scale = w_q.q_per_channel_scales().numpy().astype(np.float64)
    bias_float = qconv.bias().detach().numpy()
    denom = input_scale * w_scale
    denom = np.where(denom == 0, 1e-12, denom)
    bias_int32 = np.nan_to_num(np.round(bias_float / denom)).astype(np.int32)
    zp_y = int(qconv.zero_point) - 128
    
    params = {
        "scale_x": input_scale, "zp_x": input_zp,
        "scale_y": float(qconv.scale), "zp_y": zp_y,
        "activation": "relu" if "ReLU" in str(type(qconv_module.act)) else "none"
    }
    if is_dw:
        params.update({"W_int8_per_ch": w_int8, "B_int32_per_ch": bias_int32, "scale_w_per_ch": w_scale, "stride": qconv.stride[0]})
    else:
        params.update({"W_int8": w_int8, "B_int32": bias_int32, "scale_w": w_scale, "zp_w": 0})
        if is_3x3: params["stride"] = qconv.stride[0]
    return params

class QYOLOv10nMapped:
    """
    Complete mapped model using Golden Python blocks for backbone/neck (0-22)
    and reference head for Layer 23.
    """
    def __init__(self, fused_model):
        if 'x86' in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = 'x86'
        
        self.fused_model = fused_model
        self.layers = fused_model.model
        self.detect_head = self.layers[23]
        
        # Pre-extract all block parameters to simulate hardware descriptors
        self.param_map = {}
        self._extract_all_params()

    def _extract_all_params(self):
        from ultralytics.myimplm import collect_results
        dummy_input = torch.rand(1, 3, 640, 640)
        res = collect_results(self.fused_model, dummy_input)
        
        for i in range(23):
            m = self.layers[i]
            node = getattr(res, f"Layer{i}")
            ref_in = node.input()
            
            if isinstance(ref_in, (list, tuple)):
                in_scale = [float(t.q_scale()) for t in ref_in]
                in_zp = [int(t.q_zero_point()) - 128 for t in ref_in]
            else:
                in_scale = float(ref_in.q_scale())
                in_zp = int(ref_in.q_zero_point()) - 128
            
            type_name = type(m).__name__
            if type_name == "Conv":
                is_3x3 = (m.conv.kernel_size[0] == 3)
                self.param_map[i] = get_conv_params(m, in_scale, in_zp, is_3x3=is_3x3)
            elif type_name == "QC2f":
                from exhaustive_verify_model_flow import get_qc2f_params
                self.param_map[i] = get_qc2f_params(m, in_scale, in_zp)
            elif type_name == "SCDown":
                cv1_p = get_conv_params(m.cv1, in_scale, in_zp, is_dw=False)
                cv2_p = get_conv_params(m.cv2, cv1_p["scale_y"], cv1_p["zp_y"], is_dw=True)
                self.param_map[i] = {"cv1_params": cv1_p, "cv2_params": cv2_p}
            elif type_name == "SPPF":
                cv1_p = get_conv_params(m.cv1, in_scale, in_zp)
                c_p = {"scales": [cv1_p["scale_y"]]*4, "zps": [cv1_p["zp_y"]]*4, "scale_out": cv1_p["scale_y"], "zp_out": cv1_p["zp_y"], "strategy": "offline"}
                cv2_p = get_conv_params(m.cv2, cv1_p["scale_y"], cv1_p["zp_y"])
                self.param_map[i] = {"cv1_p": cv1_p, "cv2_p": cv2_p, "concat_p": c_p}
            elif type_name == "QPSA":
                from exhaustive_verify_model_flow import get_qpsa_params
                self.param_map[i] = get_qpsa_params(m, in_scale, in_zp)
            elif type_name == "Upsample":
                self.param_map[i] = {"scale_x": in_scale, "zp_x": in_zp, "scale_factor": 2}
            elif type_name == "QConcat":
                self.param_map[i] = {"scales": in_scale, "zps": in_zp, "concat_params": {"scale_out": float(m.fl.scale), "zp_out": int(m.fl.zero_point)-128, "strategy": "offline"}}
            elif type_name == "QC2fCIB":
                from exhaustive_verify_model_flow import get_qc2fcib_params
                self.param_map[i] = get_qc2fcib_params(m, in_scale, in_zp)

    def forward(self, x_float):
        # 0. Quantize Input
        ref_in = self.fused_model.quant(x_float)
        current_x = to_signed_int8(ref_in.int_repr().numpy())
        
        saved_features = {} # Store for skip connections (Concat)
        
        for i in range(23):
            m = self.layers[i]
            type_name = type(m).__name__
            p = self.param_map[i]
            
            if type_name == "Conv":
                current_x, _, _ = block_conv(current_x, padding="same", **p)
            elif type_name == "QC2f":
                current_x, _, _ = block_qc2f(current_x, **p)
            elif type_name == "SCDown":
                current_x, _, _ = block_scdown(current_x, p["cv1_params"], p["cv2_params"])
            elif type_name == "SPPF":
                current_x, _, _ = block_sppf(current_x, p["cv1_p"], p["cv2_p"], concat_params=p["concat_p"])
            elif type_name == "QPSA":
                current_x, _, _ = block_qpsa(current_x, **p)
            elif type_name == "Upsample":
                current_x, _, _ = block_upsample(current_x, **p)
            elif type_name == "QConcat":
                if i == 12: inputs = [saved_features[11], saved_features[6]]
                elif i == 15: inputs = [saved_features[14], saved_features[4]]
                elif i == 18: inputs = [saved_features[17], saved_features[13]]
                elif i == 21: inputs = [saved_features[20], saved_features[10]]
                current_x, _, _ = block_concat(inputs, p["scales"], p["zps"], p["concat_params"])
            elif type_name == "QC2fCIB":
                current_x, _, _ = block_qc2f_cib(current_x, **p)
            
            saved_features[i] = current_x
            
        # Final Head (Layer 23)
        # Inputs: L16, L19, L22
        def re_quant(signed_int8, scale, zp_uint8):
            # signed_int8 is in [-128, 127]
            # PyTorch quint8 is in [0, 255]
            # Shift domain: uint8 = signed_int8 + 128
            uint8_data = (signed_int8.astype(np.int16) + 128).astype(np.uint8)
            # Create a quantized tensor directly from the bytes
            # zp_uint8 is the zero_point in [0, 255] range
            return torch._make_per_tensor_quantized_tensor(torch.from_numpy(uint8_data), scale, zp_uint8)

        q_head_in = [
            re_quant(saved_features[16], float(self.layers[16].cv2.conv.scale), int(self.layers[16].cv2.conv.zero_point)),
            re_quant(saved_features[19], float(self.layers[19].cv2.conv.scale), int(self.layers[19].cv2.conv.zero_point)),
            re_quant(saved_features[22], float(self.layers[22].cv2.conv.scale), int(self.layers[22].cv2.conv.zero_point))
        ]
        
        return self.detect_head(q_head_in)
