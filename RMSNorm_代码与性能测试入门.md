# Residual Add + RMSNorm：代码、测试与使用入门

这份文档只讲项目中的 `Residual Add + RMSNorm`。阅读顺序建议是：先运行最小示例，再理解数据和线程，随后阅读四个 CUDA 版本，最后学习正确性与性能测试。

## 1. 先把程序跑起来

### 1.1 环境要求

已验证环境为 RTX 3090、CUDA Toolkit 12.8、PyTorch 2.8.0+cu128。最低需要：

- NVIDIA GPU 与可用驱动；
- CUDA Toolkit，终端能找到 `nvcc`；
- Python 与支持 CUDA 的 PyTorch；
- C++ 编译器。

先检查环境：

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一项应输出 `True`。

### 1.2 编译

进入项目目录并编译：

```bash
cd /home/sdn/wjh/LLM-CUDA-KernelLab
bash scripts/build.sh
```

编译后产生：

```text
build/libllm_cuda_lab.so   Python 调用的 CUDA 动态库
build/native_check         独立 C++ 数值和 sanitizer 检查程序
reports/raw/build.log      ptxas 输出的寄存器、shared memory、spill 信息
```

`scripts/build.sh` 中的关键编译选项：

- `-O3`：开启编译优化；
- `-std=c++17`：使用 C++17；
- `-arch=sm_86`：为 RTX 3090 的计算能力 8.6 编译；换 GPU 时要调整；
- `-lineinfo`：保留源码行映射，方便 profiler 定位；
- `--ptxas-options=-v`：输出寄存器、共享内存和 spill 信息；
- `--shared -Xcompiler -fPIC`：生成 Python 能加载的动态库。

### 1.3 最小示例

```bash
python examples/rmsnorm_quickstart.py
```

预期看到：输出形状为 `(128, 4096)`，残差结果精确相同，RMSNorm 的 `allclose` 为 `True`。浮点误差不要求严格等于零。

## 2. 算子到底算什么

输入：

```text
X        [T, H]  当前子层输出
R        [T, H]  残差
weight   [H]     RMSNorm 可学习缩放参数
eps              防止除零的小常数
```

`T` 是行数，可以理解为本次处理的 token 数；`H` 是 hidden size。项目将前面的 batch 和 sequence 维展平为 T。

第一步计算残差：

```text
U[t,h] = FP16(float(X[t,h]) + float(R[t,h]))
```

第二步按行计算均方根并归一化：

```text
mean_square[t] = Σ U[t,h]² / H
inv_rms[t] = 1 / sqrt(mean_square[t] + eps)
Y[t,h] = FP16(float(U[t,h]) × inv_rms[t] × float(weight[h]))
```

项目返回两个结果：`U` 给后续残差路径使用，`Y` 给下一层使用。

这里没有减均值，这是 RMSNorm 和 LayerNorm 的关键区别。平方和与 `inv_rms` 使用 FP32，输入输出使用 FP16。

Python 参考实现位于 `lab.py`：

```python
def rms_reference(x, r, w, eps=1e-6):
    u = (x.float() + r.float()).half()
    uf = u.float()
    y = uf * torch.rsqrt(uf.square().mean(-1, keepdim=True) + eps) * w.float()
    return u, y.half()
```

它首先显式产生 FP16 的 U，再从 U 计算 Y。这一点用于保证参考与 CUDA 版本具有相同的中间舍入语义。

## 3. 先理解 CUDA 的线程层次

这个算子使用以下映射：

```text
grid：T 个 block
每个 block：256 个 thread
一个 block：负责一个 token，也就是一整行 H
一个 thread：负责这一行中间隔 256 的多个元素
```

例如 H=1024：

```text
thread 0   → 0, 256, 512, 768
thread 1   → 1, 257, 513, 769
...
thread 255 → 255, 511, 767, 1023
```

CUDA 的一个 warp 固定包含 32 个线程，所以 256-thread block 有 8 个 warp。

代码中的几个内置变量：

- `blockIdx.x`：当前是第几行；
- `threadIdx.x`：当前线程在 block 中的编号，0～255；
- `blockDim.x`：每个 block 的线程数，本项目固定为 256；
- `__shared__`：同一 block 中所有线程共享的片上存储；
- `__syncthreads()`：等待同一 block 的所有线程到达；
- `__shfl_down_sync`：同一 warp 内在线程寄存器之间交换数据。

## 4. 公共归约代码

RMSNorm 的难点是求一整行的平方和。每个线程只能先得到自己负责元素的局部和，随后必须把 256 个局部和合并成一个总和。

### 4.1 warp_sum

`common.cuh` 中的代码：

```cpp
__device__ __forceinline__ float warp_sum(float x) {
    for (int d = 16; d; d /= 2) {
        x += __shfl_down_sync(0xffffffff, x, d);
    }
    return x;
}
```

`__shfl_down_sync(mask, x, d)` 让当前 lane 读取编号加 d 的 lane 中的 x。归约过程为：

```text
d=16：0 加 16，1 加 17，...
d=8 ：0 再加 8，...
d=4、2、1
```

最后 lane 0 持有整个 warp 的和。因为本项目的 warp 都是完整的 32 个线程，所以使用 `0xffffffff` 作为参与 mask。

### 4.2 shared memory 树形归约

`block_sum256<false>` 的工作方式：

```cpp
scratch[t] = x;
__syncthreads();
for (int d = 128; d; d /= 2) {
    if (t < d) scratch[t] += scratch[t + d];
    __syncthreads();
}
```

第一轮让前 128 个线程分别加后 128 个值，随后依次变成 128、64、32……1 个有效部分和。每轮之后必须同步，否则某些线程可能在其他线程写完前读取数据。

### 4.3 shuffle 两级归约

`block_sum256<true>` 先让每个 warp 在寄存器内归约：

```cpp
x = warp_sum(x);
if ((t & 31) == 0) scratch[t / 32] = x;
```

`t & 31` 等价于 `t % 32`。每个 warp 的 lane 0 把结果写入 `scratch`，因此只产生 8 个部分和。同步之后，由第一个 warp 归约这 8 个数：

```cpp
if (t < 32) {
    x = warp_sum(t < 8 ? scratch[t] : 0.0f);
    if (t == 0) scratch[0] = x;
}
```

最后再次同步，使所有线程都能安全读取 `scratch[0]`。与纯 shared 版本相比，它减少了共享内存访问和 block 级同步。

## 5. 四个 CUDA 版本逐段解释

### 5.1 variant 0：两个独立 kernel

Residual Add：

```cpp
__global__ void residual_add(const half* x, const half* r, half* u, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        u[i] = __float2half(__half2float(x[i]) + __half2float(r[i]));
    }
}
```

这里把二维数组看成长度为 `T×H` 的一维连续数组。每个线程处理一个元素。`if (i<n)` 保护最后一个不完整 block。

RMSNorm：

```cpp
__global__ void rms_separate(
    const half* u, const half* w, half* y, int h, float eps) {
    __shared__ float scratch[256];
    int t = threadIdx.x;
    int row = blockIdx.x * h;
    float sum = 0;

    for (int j = t; j < h; j += 256) {
        float v = __half2float(u[row + j]);
        sum += v * v;
    }

    float inv = rsqrtf(block_sum256<false>(sum, scratch) / h + eps);

    for (int j = t; j < h; j += 256) {
        y[row + j] = __float2half(
            __half2float(u[row + j]) * inv * __half2float(w[j])
        );
    }
}
```

理解这段代码需要抓住三个阶段：每线程求局部平方和、block 归约、所有线程并行写 Y。它会在写 Y 时再次从显存读取 U。

启动代码是：

```cpp
residual_add<<<ceil(T*H/256), 256, 0, stream>>>(...);
rms_separate<<<T, 256, 0, stream>>>(...);
```

前三个启动参数分别是 grid、block、动态 shared memory 字节数；第四个是 CUDA stream。

### 5.2 variant 1：融合 + shared 归约

`rms_fused<false>` 将 Add 和 Norm 合进一个 kernel。每个线程最多保存 32 个 FP32 值：

```cpp
float vals[32];
```

原因是项目限制 `H<=8192`，而 `8192/256=32`。索引为：

```cpp
int j = t + i * 256;
```

对有效元素，代码完成：

```text
读取 X/R → 相加并舍入到 FP16 → 写 U
                         ↓
                 转回 FP32 存入 vals
                         ↓
                   累加局部平方和
```

归约得到 `inv` 后，从寄存器数组 `vals` 计算 Y，无需再次读取 U。

这里的 `#pragma unroll` 要求编译器展开固定循环，可以减少循环控制，但也会扩大代码并增加寄存器压力。`vals[32]` 是优化的收益来源，也是代价来源。

### 5.3 variant 2：融合 + shuffle 归约

variant 2 使用同一个 `rms_fused` 模板，但模板参数为 true：

```cpp
rms_fused<true><<<T, 256, 0, stream>>>(...);
```

区别只在：

```cpp
block_sum256<SHUFFLE>(sum, scratch)
```

因此它非常适合做消融实验：数据读取、融合方式、线程布局完全相同，主要改变归约方法。

编译结果中，variant 1 使用 64 registers/thread，variant 2 使用 54 registers/thread；两者均没有编译器报告的 spill。寄存器减少不等于已经证明 occupancy 提高，因为当前机器没有开放 Nsight Compute 性能计数器。

### 5.4 variant 3：融合 + shuffle + half2

`half` 是一个 FP16，占 2 字节；`half2` 将两个相邻 FP16 打包成 4 字节。H=4096 时，标量版每线程最多处理 16 个值，half2 版只需处理 8 对。

关键代码：

```cpp
half2 v16 = __hadd2(x[row + j], r[row + j]);
u[row + j] = v16;
float2 v = __half22float2(v16);
sum += v.x * v.x + v.y * v.y;
```

归一化写回也一次处理两个值：

```cpp
float2 g = __half22float2(w[j]);
y[row + j] = __floats2half2_rn(
    v.x * inv * g.x,
    v.y * inv * g.y
);
```

这里的 `x[row+j]` 已经是 half2 指针索引，所以 j 表示“第几对”，不是第几个标量。

half2 要求 H 为偶数且相关指针至少 4 字节对齐。分派代码会检查：

```cpp
variant == 3 && h % 2 == 0 && all_pointers_aligned_to_4_bytes
```

不满足条件时自动执行 `rms_fused<true>`，而不是错误地强制 half2。variant 3 因而表示“优先 half2，必要时回退”。

## 6. C++ 分派与 Python 接口

`library.cu` 的 `lab_rms` 是统一 C 接口。它先检查：

```text
rows >= 0
1 <= H <= 8192
eps > 0
variant 属于 0、1、2、3
```

随后按 variant 启动对应 kernel，并返回 `cudaGetLastError()`。`extern "C"` 避免 C++ 名字改编，使 Python ctypes 能找到符号。

`lab.py` 用 `ctypes.CDLL` 加载 `build/libllm_cuda_lab.so`。`prepare_rms` 做四件事：

1. 检查输入位于同一 CUDA 设备、dtype 为 FP16、内存连续且不需要梯度；
2. 检查 X/R 为 `[T,H]`，weight 为 `[H]`；
3. 预先分配 U 和 Y；
4. 返回 `(U,Y)` 和一个 `launch()` 函数。

使用方式：

```python
(u, y), launch = lab.prepare_rms(x, residual, weight, variant=3)
launch()
```

为什么分为 prepare 和 launch？因为性能测试不能把参数检查和显存分配混入 kernel 时间，而且 CUDA Graph 要求捕获阶段不进行普通动态分配。

`launch()` 将 PyTorch 当前 stream 传给 C++。这使自写 kernel 能正确跟随 PyTorch 已提交的前序工作，也允许在非默认 stream 上运行。

CUDA kernel 是异步提交的：`launch()` 返回通常只表示任务已交给 GPU。若 CPU 立即需要结果，调用：

```python
torch.cuda.synchronize()
```

在正常神经网络流水线中通常不应每个算子都同步；这里只在测试、打印结果或计时时使用。

## 7. 正确性测试怎么读、怎么跑

运行全部测试：

```bash
python tests/test_correctness.py
```

测试中的 RMSNorm 部分遍历：

- T/H：空输入、H=1、奇数 H、常见 H、最大支持 H；
- 数值尺度：0、1、20；
- 四个 variant；
- 未满足 4 字节对齐时 variant 3 的安全回退；
- 非连续输入、错误 dtype、H 超界、requires_grad 等错误参数；
- 非默认 CUDA stream。

比较 U 时要求完全相同：

```python
compare(..., u, reference_u, atol=0, rtol=0)
```

比较 Y 时允许 FP16 浮点误差：

```python
compare(..., y, reference_y, atol=2e-3, rtol=2e-3)
```

`torch.allclose` 的判据近似为：

```text
|实际值 - 参考值| <= atol + rtol × |参考值|
```

项目当前完整测试结果是 437 项检查通过，其中 RMSNorm Y 有 96 项数值比较，最大绝对误差为 0.001953125，最大相对 L2 误差约为 2.19e-5。

还可以运行原生检查和内存检查：

```bash
./build/native_check
compute-sanitizer --tool memcheck --error-exitcode 99 ./build/native_check
compute-sanitizer --tool racecheck --error-exitcode 99 ./build/native_check
compute-sanitizer --tool synccheck --error-exitcode 99 ./build/native_check
compute-sanitizer --tool initcheck --error-exitcode 99 ./build/native_check
```

它们分别帮助发现越界/错误访问、shared memory 数据竞争、同步错误和未初始化读取。sanitizer 很慢，适合正确性检查，不用于性能计时。

## 8. 性能测试脚本逐段解释

只运行 RMSNorm：

```bash
python benchmarks/run.py --op rms --output rms_benchmark.csv
```

结果保存到：

```text
reports/raw/rms_benchmark.csv
reports/raw/rms_benchmark_environment.json
```

### 8.1 固定随机数和数学模式

脚本先设置固定随机种子，使输入可复现：

```python
torch.manual_seed(20261014)
```

GEMM 相关的 TF32 选项也被关闭。虽然它对这个 RMSNorm kernel 没有直接作用，但保证统一 benchmark 中的其他比较语义清楚。

### 8.2 graph_time 为什么这样写

第一步创建侧 stream，并等待当前 stream：

```python
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
```

第二步预热 10 次：

```python
with torch.cuda.stream(stream):
    for _ in range(10):
        fn()
stream.synchronize()
```

预热用于排除首次 CUDA context、缓存和库初始化等一次性成本。

第三步捕获 CUDA Graph：

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    for _ in range(16):
        fn()
```

一次 Graph replay 包含 16 次完整算子。这样可以减少 Python 提交间隔对微秒级 kernel 的影响。所有被比较版本都使用相同方式。

第四步用 CUDA Event 计时：

```python
begin.record()
graph.replay()
end.record()
end.synchronize()
latency_us = begin.elapsed_time(end) * 1000 / 16
```

`elapsed_time` 返回毫秒，所以乘 1000 转换为微秒；再除以 16，得到一次算子的平均 GPU 时间。Event 被记录在 GPU stream 上，比 Python 的 `time.time()` 更适合测异步 CUDA 工作。

上述过程重复 30 组，保存每组平均值，并报告中位数。CSV 中的 P10/P90 是“30 个组均值”的分位数，不是线上请求的尾延迟。

### 8.3 add 做了什么

`add()` 在正式计时前先运行一次并同步，然后与参考输出比较：

```python
fn()
torch.cuda.synchronize()
torch.allclose(output, reference, atol=1e-2, rtol=1e-2)
```

通过后才调用 `graph_time`，最终记录：

- 算子和 shape；
- 实现版本；
- 中位延迟；
- P10/P90 组均值；
- workspace 字节数；
- 最大绝对误差；
- 全部 30 个原始样本。

性能脚本使用统一的 `1e-2` 快速门槛，严格 RMSNorm 测试仍以正确性脚本的 `2e-3` 为准。

### 8.4 rms 测了哪些内容

脚本使用五组 shape：

```text
T=1,    H=4096    小任务/低并行度
T=128,  H=768     较小 hidden size
T=128,  H=4096    常见大 hidden size
T=1024, H=4096    大工作量
T=1024, H=8192    更大 hidden size
```

每组依次测试：

```text
separate       variant 0
fused_shared   variant 1
fused_shuffle  variant 2
fused_half2    variant 3
torch_eager    PyTorch 参考表达式
```

`torch_eager` 会预分配 U/Y，但仍包含多次 PyTorch kernel 和 FP32 中间运算。因此对它的加速只表示相对这段表达式，不能宣称超过所有成熟 RMSNorm 库。

### 8.5 如何看 CSV

最重要的列：

```text
shape          测试形状
variant        实现版本
latency_us     每次算子的组均值中位数
p10_group_mean_us / p90_group_mean_us
               运行波动范围
max_abs        本次输出最大绝对误差
samples_us     30 个原始组均值
```

加速比计算方式：

```text
speedup = 基线延迟 / 优化版延迟
```

例如 T=1024,H=4096：

```text
separate：    59.904 μs
fused_half2： 44.288 μs
speedup：     59.904 / 44.288 = 1.35×
```

这表示相对自写两-kernel 基线的 GPU 稳态时间，不是完整模型速度提升。

## 9. 常见错误

### `nvcc: command not found`

CUDA Toolkit 没安装或没有加入 PATH。先确认 `/usr/local/cuda/bin/nvcc` 是否存在。

### `libllm_cuda_lab.so: cannot open shared object file`

通常是还没有运行 `bash scripts/build.sh`，或者不在正确项目中运行脚本。

### `no kernel image is available for execution`

编译的 `-arch=sm_86` 与当前 GPU 不匹配。使用 `nvidia-smi` 和设备规格确认 compute capability，然后修改 `scripts/build.sh`。

### 输入报 `expected contiguous CUDA float16 tensors`

确保输入满足：

```python
x = x.cuda().half().contiguous()
```

本项目是推理演示，不支持 `requires_grad=True`，也没有 backward。

### 不同步时打印的时间接近零

CUDA 默认异步。CPU 计时若没有同步，测到的往往只是提交时间。使用 CUDA Event；简单调试时可在前后调用 `torch.cuda.synchronize()`。

### half2 是否始终运行

不是。variant 3 只有在 H 为偶数且所有指针 4 字节对齐时才运行 half2，否则自动回退到 shuffle 标量版本。

## 10. 推荐的小白学习步骤

1. 先运行 `examples/rmsnorm_quickstart.py`，确认输入、输出和参考结果。
2. 将示例中的 `variant` 改为 0、1、2、3，确认都正确。
3. 在纸上画出 H=1024 时 thread 0、1、255 处理哪些元素。
4. 用 8 个数手工模拟树形归约，再用 8 个 lane 模拟 shuffle 归约。
5. 阅读 `rms_separate`，确认每次读取和同步的目的。
6. 阅读 `rms_fused`，理解为什么 `vals` 同时带来访存收益和寄存器成本。
7. 阅读 `rms_half2`，理解标量下标和 half2 下标的区别。
8. 运行正确性测试和 sanitizer。
9. 运行 `python benchmarks/run.py --op rms --output rms_benchmark.csv`，自己计算加速比。
10. 尝试把 block size 或支持的最大 H 改掉，先写预期，再重新测试。修改 block size 时不能只改启动参数，因为归约函数当前明确按 256 个线程实现。

## 11. 当前实现的边界

- 只支持连续 CUDA FP16 tensor；
- H 范围为 1～8192；
- 固定一个 token 对应一个 256-thread block；
- 没有 backward、BF16、自动调优或按 H 编译多个专用 kernel；
- variant 3 可能回退，因此若要确认实际 kernel，应使用 Nsight Systems 或在分派层添加日志；
- 性能结果来自 RTX 3090，不能直接复制为其他 GPU 的结论；
- Nsight Compute 硬件计数器在当前环境无权限，因此没有实测 occupancy、DRAM 吞吐或 warp stall。

继续优化时，最值得尝试的是按 H 生成专用 kernel，例如为 H=768、4096、8192 分别确定每线程保留数量，并比较 128/256 个线程的布局。每次只改变一个因素，同时保留数值测试与完整基线。
