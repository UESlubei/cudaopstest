# LLM-CUDA-KernelLab：三天核心实验

当前环境已完成三类算子的实现、正确性测试、性能测量和实验报告。

- 学习顺序：[三天实验计划](三天实验计划.md)
- RMSNorm 小白教程：[代码、性能测试与使用入门](RMSNorm_代码与性能测试入门.md)
- 实测结论：[完整实验报告](reports/实验报告.md)
- 失败与修复：[问题记录](reports/问题记录.md)
- 原始数据：[80 组性能数据](reports/raw/benchmark.csv)、[正确性检查](reports/raw/correctness.json)、[打乱顺序复测](reports/raw/confirm.csv)
- 长期扩展：[原项目设计](项目设计文档.md)

## 快速复现

在本目录执行。已验证环境：Linux、RTX 3090/sm_86、CUDA Toolkit 12.8、GCC 9.4、Python 3.11、PyTorch 2.8.0+cu128。Python 运行依赖 torch，图表依赖 numpy/matplotlib；无 pytest 依赖。

```bash
bash scripts/build.sh
python tests/test_correctness.py
./build/native_check
python benchmarks/run.py
python benchmarks/confirm.py
python scripts/make_report.py
```

检查 GPU 内存与同步：

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/native_check
compute-sanitizer --tool racecheck --error-exitcode 99 ./build/native_check
compute-sanitizer --tool synccheck --error-exitcode 99 ./build/native_check
compute-sanitizer --tool initcheck --error-exitcode 99 ./build/native_check
```

这些检查覆盖原生测试程序的代表形状；Python 完整数值测试覆盖更大的集合，两者范围不同。

采集时间线：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
  -o reports/raw/nsys_representative python scripts/profile.py
nsys stats --report cuda_gpu_kern_sum,nvtx_sum --format csv \
  reports/raw/nsys_representative.nsys-rep
```

Nsight Compute 复现命令如下。本机当前账号执行会返回 `ERR_NVGPUCTRPERM`；报告保留失败记录，没有修改驱动安全配置。

```bash
ncu --set basic --section Occupancy --section MemoryWorkloadAnalysis \
  --section WarpStateStats ./build/native_check --profile gemm 2
```

## 如何读代码

| 顺序 | 文件 | 关注点 |
|---|---|---|
| 1 | `csrc/common.cuh` | warp/block 归约、mask、同步 |
| 2 | `csrc/rmsnorm.cuh` | 融合、寄存器保留、half2、FP16 舍入 |
| 3 | `csrc/gemm.cuh` | naive、16×16 shared tile、64×64 WMMA tile |
| 4 | `csrc/decode.cuh` | 三阶段基线、在线状态更新、分片合并 |
| 5 | `csrc/library.cu` / `lab.py` | kernel 分派、参数校验、当前 stream |
| 6 | `tests/test_correctness.py` / `benchmarks/run.py` | 精度、边界与公平计时 |

## 接口示例

```python
import torch
import lab

x = torch.randn(128, 4096, device='cuda', dtype=torch.float16)
r = torch.randn_like(x)
w = torch.ones(4096, device='cuda', dtype=torch.float16)
(residual_out, y), launch = lab.prepare_rms(x, r, w, variant=3)
launch()
```

`prepare_*` 检查参数并预分配输出/workspace，返回可重复调用的启动函数。保留启动函数和输入，直到异步计算完成；示例用于推理学习，不提供 autograd、线程安全重入或任意 stride 支持。

`prepare_decode` 在准备阶段同步校验 lengths；启动函数使用期间 lengths 必须保持不变。有效长度可为零，此时输出零，但 KV 的存储长度 Lmax 至少为 1。输入变换/长度校验/分配不计入本报告的稳态 GPU 时间。

## 性能口径

每条主记录是同一输入上、单次 CUDA Graph 含 16 次算子执行、30 组事件计时的组均值中位数。所有实现使用相同模式；结果是重复输入缓存条件下的 GPU 时间，不能当作 Python 调用延迟或模型 tokens/s。主 CSV 保存全部样本。

版本编号：RMSNorm `0=separate,1=fused_shared,2=fused_shuffle,3=fused_half2`；GEMM `0=naive,1=shared_tile16,2=wmma_tile64`；Decode `0=three_stage,1=online,2=split`。RMSNorm half2 在 H 为奇数或指针未四字节对齐时回退到自写 shuffle 版本。

三天版本使用显式 variant/splits 做实验，尚未实现生产级自动调优和形状分派。GQA 支持正确的 head 映射，但没有实现 query head 组内的 KV 共享优化。
