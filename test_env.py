# test_environment.py
import torch
import spconv
import numpy as np

print("=== Environment Check ===")
print(f"Python: {__import__('sys').version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"spconv version: {spconv.__version__}")

# GPU 메모리 확인
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# spconv 테스트
try:
    x = spconv.SparseConvTensor(
        torch.randn(100, 32), 
        torch.randint(0, 10, (100, 4)), 
        [10, 10, 10], 
        1
    )
    print("✅ spconv working correctly")
except Exception as e:
    print(f"❌ spconv error: {e}")

print("=== All checks completed ===")