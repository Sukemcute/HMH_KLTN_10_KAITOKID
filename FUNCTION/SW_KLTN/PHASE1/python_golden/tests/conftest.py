"""
conftest.py – pytest configuration, thêm python_golden vào sys.path.
"""
import sys, os

# Thêm thư mục gốc python_golden vào path để import hoạt động đúng
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
