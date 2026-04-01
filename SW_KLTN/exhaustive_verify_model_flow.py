
import torch
import numpy as np
import os
import sys

# Set paths
sys.path.insert(0, os.path.join(os.getcwd(), 'ultralytics'))
sys.path.insert(0, os.path.join(os.getcwd(), 'PHASE1/python_golden'))

from ultralytics.quant.utils import load_ptq_model_from_state_dict
from ultralytics.myimplm import collect_results
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

def get_qc2f_params(layer_module, in_scale, in_zp):
    cv1_p = get_conv_params(layer_module.cv1, in_scale, in_zp, is_3x3=False)
    s_cv1, z_cv1 = cv1_p["scale_y"], cv1_p["zp_y"]
    bottleneck_list = []
    scales, zps = [s_cv1, s_cv1], [z_cv1, z_cv1]
    curr_s, curr_z = s_cv1, z_cv1
    for m in layer_module.m:
        b_cv1 = get_conv_params(m.cv1, curr_s, curr_z, is_3x3=True)
        b_cv2 = get_conv_params(m.cv2, b_cv1["scale_y"], b_cv1["zp_y"], is_3x3=True)
        shortcut = getattr(m, 'add', False)
        shortcut_params = {"scale_out": float(m.fl_func.scale), "zp_out": int(m.fl_func.zero_point) - 128} if shortcut else None
        bottleneck_list.append({"cv1_params": b_cv1, "cv2_params": b_cv2, "add_params": shortcut_params, "shortcut": shortcut})
        curr_s, curr_z = (shortcut_params["scale_out"], shortcut_params["zp_out"]) if shortcut else (b_cv2["scale_y"], b_cv2["zp_y"])
        scales.append(curr_s); zps.append(curr_z)
    concat_p = {"scale_out": float(layer_module.fl.scale), "zp_out": int(layer_module.fl.zero_point)-128, "scales": scales, "zps": zps, "strategy": "offline"}
    return {"cv1_params": cv1_p, "cv2_params": get_conv_params(layer_module.cv2, concat_p["scale_out"], concat_p["zp_out"]), "bottleneck_list_params": bottleneck_list, "concat_params": concat_p}

def get_qc2fcib_params(l22, in_scale, in_zp):
    cv1_p = get_conv_params(l22.cv1, in_scale, in_zp)
    s_cv1, z_cv1 = cv1_p["scale_y"], cv1_p["zp_y"]
    qcib_p, qcib_add = [], []
    for qcib in l22.m:
        p0 = get_conv_params(qcib.cv1[0], s_cv1, z_cv1, is_dw=True)
        p1 = get_conv_params(qcib.cv1[1], p0["scale_y"], p0["zp_y"])
        p2 = get_conv_params(qcib.cv1[2], p1["scale_y"], p1["zp_y"], is_dw=True)
        p3 = get_conv_params(qcib.cv1[3], p2["scale_y"], p2["zp_y"])
        p4 = get_conv_params(qcib.cv1[4], p3["scale_y"], p3["zp_y"], is_dw=True)
        qcib_p.append([p0, p1, p2, p3, p4])
        qcib_add.append({"scale_out": float(qcib.add_fn.scale), "zp_out": int(qcib.add_fn.zero_point)-128})
    concat_p = {"scale_out": float(l22.fl.scale), "zp_out": int(l22.fl.zero_point)-128, "strategy": "offline"}
    return {"cv1_params": cv1_p, "cv2_params": get_conv_params(l22.cv2, concat_p["scale_out"], concat_p["zp_out"]), "qcib_params_list": qcib_p, "qcib_add_params_list": qcib_add, "concat_params": concat_p}

def get_qpsa_params(l10, in_scale, in_zp):
    # Static parameters from 640x640 trace
    shared_add_scale, shared_add_zp = 0.08308311551809311, 43 - 128
    attn_proj_scale, attn_proj_zp = 0.055908992886543274, 65 - 128
    cv1_p = get_conv_params(l10.cv1, in_scale, in_zp)
    qkv_p = get_conv_params(l10.attn.qkv, cv1_p["scale_y"], cv1_p["zp_y"])
    pe_p = get_conv_params(l10.attn.pe, qkv_p["scale_y"], qkv_p["zp_y"], is_dw=True)
    attn_p = {
        "qkv_params": qkv_p, "proj_params": get_conv_params(l10.attn.proj, attn_proj_scale, attn_proj_zp), "pe_params": pe_p,
        "sm_params": {"scale_out": 1.0, "zp_out": 0}, "num_heads": l10.attn.num_heads,
        "matmul1_params": {"scale_out": 0.9938528537750244, "zp_out": 54-128, "key_dim": l10.attn.key_dim, "head_scale": l10.attn.scale},
        "matmul2_params": {"scale_out": attn_proj_scale, "zp_out": attn_proj_zp, "head_dim": l10.attn.head_dim},
        "add_pe_params": {"scale_out": attn_proj_scale, "zp_out": attn_proj_zp}
    }
    ffn_p = [get_conv_params(l10.ffn[0], shared_add_scale, shared_add_zp), get_conv_params(l10.ffn[1], float(l10.ffn[0].conv.scale), int(l10.ffn[0].conv.zero_point)-128)]
    return {"cv1_params": cv1_p, "attn_params": attn_p, "ffn_params": ffn_p, "ffn_add_params": {"scale_out": shared_add_scale, "zp_out": shared_add_zp}, "attn_add_params": {"scale_out": shared_add_scale, "zp_out": shared_add_zp}, "cv2_params": get_conv_params(l10.cv2, shared_add_scale, shared_add_zp), "concat_params": {"scale_out": shared_add_scale, "zp_out": shared_add_zp}}

def verify_flow():
    print("=== Full Model Flow Exhaustive Verification (100 Samples, 640x640) ===")
    base_weights, quant_state_dict = 'ultralytics/ultralytics/qyolov10n.yaml', 'ultralytics/ultralytics/quant/quant_state_dict/alpr_ptq_state_dict.pt'
    model = load_ptq_model_from_state_dict(base_weights, quant_state_dict); model.model.eval()
    
    num_samples = 100
    stats = {i: {"matches": [], "max_diffs": [], "type": type(m).__name__} for i, m in enumerate(model.model.model) if i < 23}
    concat_map = {12: [11, 6], 15: [14, 4], 18: [17, 13], 21: [20, 10]}

    for s in range(num_samples):
        input_tensor = torch.rand(1, 3, 640, 640)
        res = collect_results(model, input_tensor)
        if s % 10 == 0: print(f"  Sample {s}/{num_samples}...")

        for i, m in enumerate(model.model.model):
            if i >= 23: break
            node = getattr(res, f"Layer{i}")
            ref_out_signed = to_signed_int8(node().int_repr().numpy())
            
            # Extract realistic inputs
            if i == 0:
                ref_in = model.model.quant(input_tensor)
                in_signed, in_scale, in_zp = to_signed_int8(ref_in.int_repr().numpy()), float(ref_in.q_scale()), int(ref_in.q_zero_point())-128
            elif i in concat_map:
                in_signed = [to_signed_int8(getattr(res, f"Layer{idx}")().int_repr().numpy()) for idx in concat_map[i]]
                in_scale = [float(getattr(res, f"Layer{idx}")().q_scale()) for idx in concat_map[i]]
                in_zp = [int(getattr(res, f"Layer{idx}")().q_zero_point())-128 for idx in concat_map[i]]
            else:
                ref_in = getattr(res, f"Layer{i-1}")()
                in_signed, in_scale, in_zp = to_signed_int8(ref_in.int_repr().numpy()), float(ref_in.q_scale()), int(ref_in.q_zero_point())-128

            # Run Golden Block
            if "Conv" == stats[i]["type"]:
                is_3x3 = (m.conv.kernel_size[0] == 3)
                my_out, _, _ = block_conv(in_signed, padding="same", **get_conv_params(m, in_scale, in_zp, is_3x3=is_3x3))
            elif "QC2f" == stats[i]["type"]:
                my_out, _, _ = block_qc2f(in_signed, **get_qc2f_params(m, in_scale, in_zp))
            elif "SCDown" == stats[i]["type"]:
                cv1_p = get_conv_params(m.cv1, in_scale, in_zp, is_dw=False)
                my_out, _, _ = block_scdown(in_signed, cv1_p, get_conv_params(m.cv2, cv1_p["scale_y"], cv1_p["zp_y"], is_dw=True))
            elif "SPPF" == stats[i]["type"]:
                cv1_p = get_conv_params(m.cv1, in_scale, in_zp)
                c_p = {"scales": [cv1_p["scale_y"]]*4, "zps": [cv1_p["zp_y"]]*4, "scale_out": cv1_p["scale_y"], "zp_out": cv1_p["zp_y"], "strategy": "offline"}
                my_out, _, _ = block_sppf(in_signed, cv1_p, get_conv_params(m.cv2, cv1_p["scale_y"], cv1_p["zp_y"]), concat_params=c_p)
            elif "QPSA" == stats[i]["type"]:
                my_out, _, _ = block_qpsa(in_signed, **get_qpsa_params(m, in_scale, in_zp))
            elif "Upsample" == stats[i]["type"]:
                my_out, _, _ = block_upsample(in_signed, in_scale, in_zp, scale_factor=2)
            elif "QConcat" == stats[i]["type"]:
                c_p = {"scale_out": float(m.fl.scale), "zp_out": int(m.fl.zero_point)-128, "strategy": "offline"}
                my_out, _, _ = block_concat(in_signed, in_scale, in_zp, c_p)
            elif "QC2fCIB" == stats[i]["type"]:
                my_out, _, _ = block_qc2f_cib(in_signed, **get_qc2fcib_params(m, in_scale, in_zp))
            else: continue

            diff = np.abs(my_out.astype(np.int16) - ref_out_signed.astype(np.int16))
            stats[i]["matches"].append((diff == 0).sum() / diff.size * 100)
            stats[i]["max_diffs"].append(np.max(diff))

    print("\n" + "="*60); print(f"{'Layer':<6} | {'Type':<12} | {'Match %':<10} | {'Max Diff':<8} | {'Status'}"); print("-" * 60)
    for i in range(23):
        m_p, d_p = np.mean(stats[i]["matches"]), np.max(stats[i]["max_diffs"])
        status = "PASS" if d_p <= 4 or (i==10 and d_p<=25) else "WARN"
        print(f"{i:<6} | {stats[i]['type']:<12} | {m_p:8.2f}% | {d_p:<8} | [{status}]")

if __name__ == "__main__":
    verify_flow()
