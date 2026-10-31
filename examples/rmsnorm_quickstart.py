"""Residual Add + RMSNorm 最小示例。

先在项目根目录运行：
    bash scripts/build.sh
再运行：
    python examples/rmsnorm_quickstart.py
"""
import sys
from pathlib import Path

import torch

# 允许从 examples/ 子目录导入项目根目录中的 lab.py。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 NVIDIA CUDA GPU")

    torch.manual_seed(20261014)
    device = "cuda"
    rows, hidden = 128, 4096

    # X/R 的每一行代表一个 token；weight 在所有 token 之间共享。
    x = torch.randn(rows, hidden, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.ones(hidden, device=device, dtype=torch.float16)

    # variant=3：优先使用 half2 版本；不满足偶数 H 或对齐条件时安全回退。
    (residual_out, y), launch = lab.prepare_rms(
        x, residual, weight, variant=3, eps=1e-6
    )

    # prepare_rms 只检查参数并分配输出；调用 launch 才真正提交 CUDA kernel。
    launch()

    # CUDA 默认异步执行；读取数值或正式计时前需要等待 GPU 完成。
    torch.cuda.synchronize()

    reference_residual, reference_y = lab.rms_reference(
        x, residual, weight, eps=1e-6
    )
    print("GPU:", torch.cuda.get_device_name())
    print("output shape:", tuple(y.shape))
    print("residual exact match:", torch.equal(residual_out, reference_residual))
    print("RMSNorm max abs error:", (y.float() - reference_y.float()).abs().max().item())
    print("RMSNorm allclose:", torch.allclose(y, reference_y, atol=2e-3, rtol=2e-3))


if __name__ == "__main__":
    main()
