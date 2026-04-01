"""
test_quant.py – Test suite cho quant_affine.py và quant_domain_align.py

Tất cả test phải PASS trước khi code primitive.

Bao gồm:
  Step A: quant_affine – quantize/dequantize/requant/decompose
  Step B: quant_domain_align – concat/add domain alignment
"""

import pytest
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INT8_MIN, INT8_MAX, INT32_MAX
from quant.quant_affine import (
    quantize_affine,
    dequantize_affine,
    make_requant_params,
    post_process_int32_to_int8,
    build_silu_lut,
    apply_silu_float,
    apply_silu_lut,
    _fixed_point_decompose_scalar,
)
from quant.quant_domain_align import (
    compute_common_scale,
    requant_to_common,
    align_and_concat,
    align_and_add,
)


# ═══════════════════════════════════════════════════════════════════
# STEP A – quant_affine
# ═══════════════════════════════════════════════════════════════════

class TestQuantizeAffine:
    """Test quantize_affine / dequantize_affine."""

    def test_round_trip_error_less_than_1_lsb(self):
        """quantize(dequantize(x)) ≈ x với sai số < 1 LSB cho giá trị trong range."""
        scale = 0.05
        zp = 0
        # Dùng giá trị nằm trong representable range: [INT8_MIN*scale, INT8_MAX*scale]
        # = [-6.4, 6.35]
        x_orig = np.array([-5.0, -2.5, 0.0, 1.25, 3.5, 5.5], dtype=np.float64)
        x_int8 = quantize_affine(x_orig, scale, zp)
        x_back = dequantize_affine(x_int8, scale, zp)
        err = np.abs(x_orig - x_back)
        # Sai số phải ≤ 0.5 * scale (nửa LSB) do rounding
        assert np.all(err <= scale), \
            f"Round-trip error quá lớn: max={err.max():.4f}, 1LSB={scale}"

    def test_clamp_boundaries_int8(self):
        """Giá trị vượt biên phải bị clamp về [-128, 127]."""
        scale = 0.01
        zp = 0
        x = np.array([-200.0, 200.0], dtype=np.float32)
        x_int8 = quantize_affine(x, scale, zp)
        assert x_int8[0] == INT8_MIN, f"Âm phải clamp về {INT8_MIN}"
        assert x_int8[1] == INT8_MAX, f"Dương phải clamp về {INT8_MAX}"

    def test_clamp_boundaries_uint8(self):
        """uint8: clamp về [0, 255]."""
        scale = 0.01
        zp = 0
        x = np.array([-10.0, 300.0], dtype=np.float32)
        x_uint8 = quantize_affine(x, scale, zp, dtype="uint8")
        assert x_uint8[0] == 0
        assert x_uint8[1] == 255

    def test_zero_point_effect(self):
        """zp ≠ 0: điểm 0 float ánh xạ tới zp trong INT8."""
        scale = 0.1
        zp = 20
        x_zero = np.array([0.0])
        q = quantize_affine(x_zero, scale, zp)
        assert q[0] == zp, f"x=0 phải map tới zp={zp}, nhận {q[0]}"

    def test_dequantize_formula(self):
        """dequantize(v, scale, zp) = (v - zp) * scale."""
        scale = 0.04
        zp = 5
        x_int8 = np.array([-50, 0, 50, 100], dtype=np.int8)
        expected = (x_int8.astype(np.float64) - zp) * scale
        result = dequantize_affine(x_int8, scale, zp)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_symmetric_quantize(self):
        """Symmetric quant (zp=0): giá trị dương/âm đối xứng."""
        scale = 0.1
        zp = 0
        x = np.array([-1.0, 0.0, 1.0])
        q = quantize_affine(x, scale, zp)
        assert q[0] == -10
        assert q[1] == 0
        assert q[2] == 10

    def test_batch_shape(self):
        """Quantize hoạt động đúng trên tensor nhiều chiều."""
        scale = 0.05
        zp = 0
        x = np.random.randn(2, 3, 8, 8).astype(np.float32) * 2.0
        q = quantize_affine(x, scale, zp)
        assert q.shape == x.shape
        assert q.dtype == np.int8


class TestFixedPointDecompose:
    """Test _fixed_point_decompose_scalar và make_requant_params."""

    def test_decompose_accuracy(self):
        """M_int / 2^shift ≈ M với relative error < 1e-5."""
        for M in [0.003, 0.05, 0.1, 0.5, 0.9, 1.0, 1.5, 2.5, 3.14]:
            M_int, shift = _fixed_point_decompose_scalar(M)
            M_approx = M_int / (2 ** shift)
            rel_err = abs(M_approx - M) / M
            assert rel_err < 1e-5, \
                f"M={M}: M_int={M_int}, shift={shift}, rel_err={rel_err:.2e}"

    def test_decompose_zero(self):
        """M = 0 → (0, 0)."""
        M_int, shift = _fixed_point_decompose_scalar(0.0)
        assert M_int == 0 and shift == 0

    def test_decompose_M_int_fits_int32(self):
        """M_int phải fit trong INT32."""
        for M in [0.001, 0.5, 0.9999]:
            M_int, shift = _fixed_point_decompose_scalar(M)
            assert 0 <= M_int <= INT32_MAX, f"M_int={M_int} không fit INT32"

    def test_decompose_shift_in_range(self):
        """Shift phải ∈ [0, 31]."""
        for M in [0.001, 0.1, 0.5, 0.99]:
            _, shift = _fixed_point_decompose_scalar(M)
            assert 0 <= shift <= 31, f"shift={shift} ngoài range [0,31]"

    def test_make_requant_params_shape(self):
        """make_requant_params trả về arrays shape [Cout]."""
        scale_in = 0.004
        scale_w = np.array([0.01, 0.02, 0.015, 0.008])
        scale_out = 0.05
        M_int_arr, shift_arr = make_requant_params(scale_in, scale_w, scale_out)
        assert M_int_arr.shape == (4,)
        assert shift_arr.shape == (4,)

    def test_make_requant_params_accuracy(self):
        """(M_int[c] / 2^shift[c]) ≈ scale_in * scale_w[c] / scale_out."""
        scale_in = 0.00392
        scale_w = np.array([0.01, 0.02, 0.005])
        scale_out = 0.04
        M_true = scale_in * scale_w / scale_out
        M_int_arr, shift_arr = make_requant_params(scale_in, scale_w, scale_out)
        for i in range(len(scale_w)):
            M_approx = M_int_arr[i] / (2 ** int(shift_arr[i]))
            rel_err = abs(M_approx - M_true[i]) / M_true[i]
            assert rel_err < 1e-4, f"Channel {i}: rel_err={rel_err:.2e}"


class TestPostProcessInt32:
    """Test post_process_int32_to_int8."""

    def test_basic_requant(self):
        """Kiểm tra basic requant cho shape [N, Cout, H, W]."""
        N, Cout, H, W = 1, 4, 2, 2
        scale_in = 0.004
        scale_w = np.array([0.01, 0.02, 0.015, 0.008])
        scale_out = 0.05

        M_int, shift = make_requant_params(scale_in, scale_w, scale_out)
        acc = np.array([[[[100, 200], [300, 400]],
                         [[50, 100], [150, 200]],
                         [[80, 160], [240, 320]],
                         [[30, 60], [90, 120]]]], dtype=np.int64)
        y = post_process_int32_to_int8(acc, M_int, shift, zp_out=0)
        assert y.shape == (N, Cout, H, W)
        assert y.dtype == np.int8
        assert np.all(y >= INT8_MIN) and np.all(y <= INT8_MAX)

    def test_clamp_to_int8(self):
        """Giá trị lớn phải bị clamp về [-128, 127]."""
        M_int = np.array([INT32_MAX // 2], dtype=np.int64)
        shift = np.array([0], dtype=np.int32)
        acc = np.array([[[[1000000]]]], dtype=np.int64)
        y = post_process_int32_to_int8(acc, M_int, shift, zp_out=0)
        assert y[0, 0, 0, 0] == INT8_MAX

    def test_overflow_safety(self):
        """acc_int32 = INT32_MAX không gây crash."""
        M_int = np.array([1], dtype=np.int64)
        shift = np.array([0], dtype=np.int32)
        acc = np.array([[[[np.iinfo(np.int32).max]]]], dtype=np.int64)
        y = post_process_int32_to_int8(acc, M_int, shift, zp_out=0)
        assert y.dtype == np.int8

    def test_zp_out_applied(self):
        """zp_out được cộng vào kết quả cuối."""
        M_int = np.array([1 << 20], dtype=np.int64)
        shift = np.array([20], dtype=np.int32)
        acc = np.zeros((1, 1, 1, 1), dtype=np.int64)
        zp_out = 5
        y = post_process_int32_to_int8(acc, M_int, shift, zp_out=zp_out)
        assert int(y[0, 0, 0, 0]) == zp_out


class TestSiLUActivation:
    """Test SiLU LUT và float path."""

    def test_silu_lut_vs_float_path(self):
        """LUT và float path phải cho kết quả ±1 LSB."""
        scale_y = 0.03
        zp_y = 0
        lut = build_silu_lut(scale_y, zp_y)

        x_int8 = np.arange(-128, 128, dtype=np.int8)
        y_float = apply_silu_float(x_int8, scale_y, zp_y)
        y_lut = apply_silu_lut(x_int8, lut)

        diff = np.abs(y_float.astype(np.int16) - y_lut.astype(np.int16))
        assert np.all(diff <= 1), \
            f"LUT vs float path sai lệch quá 1 LSB: max diff={diff.max()}"

    def test_silu_lut_size(self):
        """LUT phải có đúng 256 entries, dtype int8."""
        lut = build_silu_lut(0.05, 0)
        assert lut.shape == (256,)
        assert lut.dtype == np.int8

    def test_silu_positive_inputs(self):
        """Với input dương, SiLU ≈ input (SiLU(x) ≈ x với x >> 0)."""
        scale_y = 0.1
        zp_y = 0
        # Giá trị lớn dương: SiLU(x) ≈ x
        x_large_pos = np.array([100], dtype=np.int8)  # = 10.0 in float
        y = apply_silu_float(x_large_pos, scale_y, zp_y)
        # SiLU(10) ≈ 10 → output ≈ 100
        assert abs(int(y[0]) - 100) <= 2, f"SiLU(large pos) sai: {y[0]}"

    def test_silu_zero(self):
        """SiLU(0) = 0."""
        scale_y = 0.1
        zp_y = 0
        x_zero = np.array([0], dtype=np.int8)
        y = apply_silu_float(x_zero, scale_y, zp_y)
        assert y[0] == 0, f"SiLU(0) phải = 0, nhận {y[0]}"


# ═══════════════════════════════════════════════════════════════════
# STEP B – quant_domain_align
# ═══════════════════════════════════════════════════════════════════

class TestComputeCommonScale:
    """Test compute_common_scale."""

    def test_strategy_max(self):
        """strategy='max' → chọn scale lớn nhất."""
        scales = [0.05, 0.1, 0.03]
        zps = [0, 1, 2]
        common_scale, common_zp = compute_common_scale(scales, zps, strategy="max")
        assert abs(common_scale - 0.1) < 1e-10
        assert common_zp == 1

    def test_strategy_offline(self):
        """strategy='offline' → chọn scales[0]."""
        scales = [0.05, 0.1, 0.03]
        zps = [3, 1, 2]
        common_scale, common_zp = compute_common_scale(scales, zps, strategy="offline")
        assert abs(common_scale - 0.05) < 1e-10
        assert common_zp == 3

    def test_requires_at_least_2(self):
        """Cần ít nhất 2 scales."""
        with pytest.raises(AssertionError):
            compute_common_scale([0.1])


class TestRequantToCommon:
    """Test requant_to_common."""

    def test_identity_path_same_params(self):
        """scale/zp giống nhau → không requant, kết quả y hệt."""
        x = np.array([-10, 0, 50, 127], dtype=np.int8)
        y = requant_to_common(x, 0.1, 0, 0.1, 0)
        np.testing.assert_array_equal(x, y)

    def test_requant_changes_values(self):
        """scale khác nhau → giá trị thay đổi đúng."""
        scale_src, zp_src = 0.05, 0
        scale_dst, zp_dst = 0.1, 0
        x = np.array([20], dtype=np.int8)  # = 1.0 in float (20 * 0.05)
        y = requant_to_common(x, scale_src, zp_src, scale_dst, zp_dst)
        # 1.0 / 0.1 = 10 → y = 10
        assert abs(int(y[0]) - 10) <= 1, f"Expected 10, got {y[0]}"

    def test_output_dtype_is_int8(self):
        """Output phải là int8."""
        x = np.array([50], dtype=np.int8)
        y = requant_to_common(x, 0.05, 0, 0.1, 0)
        assert y.dtype == np.int8


class TestAlignAndConcat:
    """Test align_and_concat – rủi ro số 1 của dự án."""

    def test_same_domain_no_requant(self):
        """Cùng scale/zp → concat trực tiếp, không requant."""
        A = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)   # [1,1,2,2]
        B = np.array([[[[5, 6], [7, 8]]]], dtype=np.int8)   # [1,1,2,2]
        Y, s, z = align_and_concat([A, B], [0.1, 0.1], [0, 0], axis=1)
        assert Y.shape == (1, 2, 2, 2)
        assert abs(s - 0.1) < 1e-10
        np.testing.assert_array_equal(Y[:, 0, :, :], A[:, 0, :, :])
        np.testing.assert_array_equal(Y[:, 1, :, :], B[:, 0, :, :])

    def test_domain_mismatch_scale_A_ne_scale_B(self):
        """scale_A ≠ scale_B → requant nhánh nhỏ về common scale."""
        scale_A, scale_B = 0.1, 0.05
        # A: [1, 4, 2, 2], float value = A_int8 * 0.1
        A = np.random.randint(-50, 50, (1, 4, 2, 2), dtype=np.int8)
        B = np.random.randint(-50, 50, (1, 8, 2, 2), dtype=np.int8)

        Y, common_scale, common_zp = align_and_concat(
            [A, B], [scale_A, scale_B], [0, 0], axis=1, strategy="max"
        )
        assert Y.shape == (1, 12, 2, 2)
        assert Y.dtype == np.int8
        # Common scale phải là max(0.1, 0.05) = 0.1
        assert abs(common_scale - 0.1) < 1e-10

    def test_concat_4_branches_sppf(self):
        """Concat 4 nhánh (SPPF case): X1, P1, P2, P3."""
        C = 32  # 128 channels giả lập nhỏ
        tensors = [np.random.randint(-50, 50, (1, C, 5, 5), dtype=np.int8)
                   for _ in range(4)]
        scales = [0.05] * 4
        zps = [0] * 4
        Y, s, z = align_and_concat(tensors, scales, zps, axis=1)
        assert Y.shape == (1, C * 4, 5, 5)
        assert Y.dtype == np.int8

    def test_output_values_float_consistent(self):
        """Float reconstruction của output ≈ cat(float(A), float(B))."""
        scale_A, scale_B = 0.1, 0.1
        zp_A, zp_B = 0, 0
        A = np.array([[[[10, 20], [30, 40]]]], dtype=np.int8)
        B = np.array([[[[50, 60], [70, 80]]]], dtype=np.int8)

        Y, s, z = align_and_concat([A, B], [scale_A, scale_B], [zp_A, zp_B], axis=1)
        # Same domain → Y[:, 0] = A, Y[:, 1] = B
        A_float = A.astype(np.float32) * scale_A
        B_float = B.astype(np.float32) * scale_B
        Y_float = Y.astype(np.float32) * s

        expected = np.concatenate(
            [A_float, B_float], axis=1
        )
        err = np.abs(Y_float - expected)
        assert np.all(err <= s), f"Max float error: {err.max()}"


class TestAlignAndAdd:
    """Test align_and_add."""

    def test_add_saturation(self):
        """Cộng vượt INT8 range → clamp đúng."""
        A = np.array([[[[100]]]], dtype=np.int8)
        B = np.array([[[[100]]]], dtype=np.int8)
        # scale = 0.1: float A = 10.0, B = 10.0, sum = 20.0
        # scale = 0.1: 20.0 / 0.1 = 200 → clamp về 127
        Y, s, z = align_and_add(
            A, 0.1, 0, B, 0.1, 0, scale_out=0.1, zp_out=0
        )
        assert Y.dtype == np.int8
        assert int(Y[0, 0, 0, 0]) == INT8_MAX, \
            f"Expected {INT8_MAX}, got {Y[0, 0, 0, 0]}"

    def test_add_domain_equal_identity(self):
        """scale_A == scale_B, zp_A == zp_B → không requant thêm."""
        A = np.array([[[[10, 20], [30, 40]]]], dtype=np.int8)
        B = np.array([[[[5, 10], [15, 20]]]], dtype=np.int8)
        # scale=0.1, zp=0: float A = [1,2,3,4], B = [0.5,1,1.5,2]
        Y, s, z = align_and_add(
            A, 0.1, 0, B, 0.1, 0, scale_out=0.1, zp_out=0
        )
        assert Y.dtype == np.int8
        # Expected: clamp(A + B - zp + zp_out) clamped to INT8
        expected_sum = np.clip(
            A.astype(np.int16) + B.astype(np.int16), -128, 127
        ).astype(np.int8)
        np.testing.assert_array_equal(Y, expected_sum)

    def test_add_domain_mismatch(self):
        """scale_A ≠ scale_B → align trước khi add."""
        # Float: A = 5.0, B = 3.0 → sum = 8.0
        scale_A, scale_B = 0.1, 0.05
        A = np.array([[[[50]]]], dtype=np.int8)   # 50 * 0.1 = 5.0
        B = np.array([[[[60]]]], dtype=np.int8)   # 60 * 0.05 = 3.0
        # common scale = max = 0.1
        # B requanted: 3.0 / 0.1 = 30
        # sum = 50 + 30 = 80 (before zp adjustment)
        Y, s, z = align_and_add(
            A, scale_A, 0, B, scale_B, 0, scale_out=0.1, zp_out=0
        )
        Y_float = dequantize_affine(Y, s, z)
        assert abs(float(Y_float[0, 0, 0, 0]) - 8.0) <= s, \
            f"Expected ~8.0, got {float(Y_float[0,0,0,0]):.3f}"

    def test_output_shape_preserved(self):
        """Shape output == shape input."""
        A = np.random.randint(-50, 50, (2, 8, 6, 6), dtype=np.int8)
        B = np.random.randint(-50, 50, (2, 8, 6, 6), dtype=np.int8)
        Y, s, z = align_and_add(A, 0.05, 0, B, 0.05, 0)
        assert Y.shape == A.shape


# ═══════════════════════════════════════════════════════════════════
# Run independently
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
