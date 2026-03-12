"""
Fused MoE FP8 Benchmark & Correctness Framework
=================================================
用法:
  python moe_bench.py --mode correctness   # 正确性验证
  python moe_bench.py --mode perf          # 性能对比
  python moe_bench.py --mode both          # 两者都跑
  python moe_bench.py --mode quick         # 框架自检
  python moe_bench.py --mode chart         # 生成图表
"""

import torch
import triton
import triton.testing
import argparse
from dataclasses import dataclass
from typing import Callable, Dict


# ============================================================
# 1. 参数配置（DeepSeek V3/R1 绑定参数 + 可调参数）
# ============================================================

@dataclass
class MoEConfig:
    """MoE 层的所有维度参数"""
    H: int = 7168          # hidden size
    I: int = 2048          # intermediate size (per gate/up)
    E_global: int = 256    # 全局专家数
    E_local: int = 32      # 本地专家数 (如 8卡各32个)
    TOP_K: int = 8         # 每 token 选几个专家
    N_GROUP: int = 8       # 路由分组数
    TOPK_GROUP: int = 4    # 选几个组
    BLOCK: int = 128       # FP8 block-scale 粒度
    local_expert_offset: int = 0
    routed_scaling_factor: float = 2.5  # DeepSeek V3 用的值

    def __post_init__(self):
        assert self.H % self.BLOCK == 0
        assert self.I % self.BLOCK == 0
        assert (2 * self.I) % self.BLOCK == 0
        assert self.E_global % self.N_GROUP == 0


# ============================================================
# 2. 数据构造工厂（核心！把复杂的输入构造封装起来）
# ============================================================

class MoEDataFactory:
    """构造 MoE kernel 的全部输入张量"""

    def __init__(self, config: MoEConfig, device: str = "cuda"):
        self.cfg = config
        self.device = device

    def create(self, T: int, seed: int = 42) -> Dict[str, torch.Tensor]:
        """
        一键构造所有输入，返回字典。
        T: token 数量
        """
        cfg = self.cfg
        torch.manual_seed(seed)
        device = self.device

        # --- 路由相关 ---
        routing_logits = torch.randn(T, cfg.E_global, device=device, dtype=torch.float32)
        routing_bias = torch.randn(cfg.E_global, device=device, dtype=torch.float32) * 0.1

        # --- 激活值 FP8 模拟 ---
        hidden_real = torch.randn(T, cfg.H, device=device, dtype=torch.float32)
        hidden_fp8, hidden_scale = self._quantize_activation(hidden_real, T, cfg.H)

        # --- 权重 W13 (Gate+Up) FP8 ---
        # shape: [E_local, 2*I, H]
        w13_real = torch.randn(cfg.E_local, 2 * cfg.I, cfg.H, device=device, dtype=torch.float32) * 0.02
        w13_fp8, w13_scale = self._quantize_weight(w13_real)

        # --- 权重 W2 (Down) FP8 ---
        # shape: [E_local, H, I]
        w2_real = torch.randn(cfg.E_local, cfg.H, cfg.I, device=device, dtype=torch.float32) * 0.02
        w2_fp8, w2_scale = self._quantize_weight(w2_real)

        # --- 输出 buffer ---
        output = torch.zeros(T, cfg.H, device=device, dtype=torch.bfloat16)

        return {
            "routing_logits": routing_logits,
            "routing_bias": routing_bias,
            "hidden_states": hidden_fp8,
            "hidden_states_scale": hidden_scale,
            "gemm1_weights": w13_fp8,
            "gemm1_weights_scale": w13_scale,
            "gemm2_weights": w2_fp8,
            "gemm2_weights_scale": w2_scale,
            "local_expert_offset": cfg.local_expert_offset,
            "routed_scaling_factor": cfg.routed_scaling_factor,
            "output": output,
            # 额外保存真实值用于精度验证
            "_hidden_real": hidden_real,
            "_w13_real": w13_real,
            "_w2_real": w2_real,
        }

    def _quantize_activation(self, x: torch.Tensor, T: int, H: int):
        """
        模拟 per-block 激活量化
        x: [T, H] fp32
        返回: (fp8_values, scales)
          fp8_values: [T, H] (用 float8_e4m3fn 或 fp32 模拟)
          scales: [H//BLOCK, T] — 注意 DeepSeek 的 scale 布局是转置的！
        """
        B = self.cfg.BLOCK
        n_blocks = H // B

        # 按 block 计算 absmax
        x_blocked = x.view(T, n_blocks, B)
        absmax = x_blocked.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)

        # FP8 E4M3 的 max 是 448
        fp8_max = 448.0
        scales = absmax.squeeze(2) / fp8_max  # [T, n_blocks]

        # 量化
        x_quant = (x_blocked / absmax * fp8_max).clamp(-fp8_max, fp8_max)
        x_quant = x_quant.view(T, H)

        # 模拟 FP8 精度损失 (round to nearest)
        x_quant = x_quant.round()

        # DeepSeek 的 scale 布局: [H//BLOCK, T] (转置)
        scales_transposed = scales.permute(1, 0).contiguous()

        # 尝试用真 FP8 dtype，不支持就用 fp32 模拟
        try:
            x_fp8 = x_quant.to(torch.float8_e4m3fn)
        except (RuntimeError, AttributeError):
            x_fp8 = x_quant  # fallback to fp32 模拟

        return x_fp8, scales_transposed

    def _quantize_weight(self, w: torch.Tensor):
        """
        模拟 2D block-scale 权重量化
        w: [E, out_dim, in_dim] fp32
        返回: (fp8_values, scales)
          scales: [E, out_dim//BLOCK, in_dim//BLOCK]
        """
        B = self.cfg.BLOCK
        E, out_dim, in_dim = w.shape
        nb_out = out_dim // B
        nb_in = in_dim // B
        fp8_max = 448.0

        w_blocked = w.view(E, nb_out, B, nb_in, B)
        absmax = w_blocked.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-12)
        scales = absmax.squeeze(4).squeeze(2) / fp8_max  # [E, nb_out, nb_in]

        w_quant = (w_blocked / absmax * fp8_max).clamp(-fp8_max, fp8_max).round()
        w_quant = w_quant.view(E, out_dim, in_dim)

        try:
            w_fp8 = w_quant.to(torch.float8_e4m3fn)
        except (RuntimeError, AttributeError):
            w_fp8 = w_quant

        return w_fp8, scales


# ============================================================
# 3. PyTorch Reference Baseline
# ============================================================

@torch.no_grad()
def pytorch_reference(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor, output,
    config: MoEConfig,
    **kwargs,
):
    """PyTorch baseline — 数值正确但性能差"""
    H = config.H
    I = config.I
    E_local = gemm1_weights.shape[0]
    BLOCK = config.BLOCK
    E_global = config.E_global
    T = routing_logits.shape[0]
    TOP_K = config.TOP_K
    N_GROUP = config.N_GROUP
    TOPK_GROUP = config.TOPK_GROUP
    device = hidden_states.device

    # 1) FP8 反量化
    A_fp32 = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.to(torch.float32)
    A_scale_TH = A_scale.permute(1, 0).contiguous()
    A_scale_expanded = (
        A_scale_TH.unsqueeze(-1)
        .repeat(1, 1, BLOCK)
        .reshape(T, H)
        .contiguous()
    )
    A = A_fp32 * A_scale_expanded

    W13_fp32 = gemm1_weights.to(torch.float32)
    S13 = gemm1_weights_scale.to(torch.float32)
    S13_expanded = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_expanded = torch.repeat_interleave(S13_expanded, BLOCK, dim=2)
    W13 = W13_fp32 * S13_expanded

    W2_fp32 = gemm2_weights.to(torch.float32)
    S2 = gemm2_weights_scale.to(torch.float32)
    S2_expanded = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_expanded = torch.repeat_interleave(S2_expanded, BLOCK, dim=2)
    W2 = W2_fp32 * S2_expanded

    # 2) Grouped routing
    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)
    s = 1.0 / (1.0 + torch.exp(-logits))
    s_with_bias = s + bias

    group_size = E_global // N_GROUP
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # 3) Expert compute
    temp_output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel_mask_per_token = (topk_idx == ge).any(dim=1)
        if not sel_mask_per_token.any():
            continue

        token_idx = torch.nonzero(sel_mask_per_token, as_tuple=False).squeeze(1)
        A_e = A.index_select(0, token_idx)
        W13_e = W13[le]
        W2_e = W2[le]

        G1 = A_e.matmul(W13_e.t())
        X1 = G1[:, :I]
        X2 = G1[:, I:]
        silu_X2 = X2 / (1.0 + torch.exp(-X2))
        C = silu_X2 * X1
        O = C.matmul(W2_e.t())

        w_tok = weights.index_select(0, token_idx)[:, ge]
        temp_output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    output.copy_(temp_output.to(output.dtype))
    return output


# ============================================================
# 4. Triton 实现占位（你后续填入）
# ============================================================

def triton_fused_moe(
    routing_logits, routing_bias, hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor, output,
    config: MoEConfig,
    **kwargs,
):
    """
    TODO: 你的 Triton fused MoE 实现
    保持和 pytorch_reference 完全相同的函数签名。
    可以分阶段替换：
      Phase 1: 路由用 PyTorch，GEMM 用 Triton kernel
      Phase 2: fuse GEMM + SwiGLU
      Phase 3: grouped GEMM (多专家并行)
      Phase 4: fuse routing dispatch + compute + scatter
    """
    raise NotImplementedError(
        "在这里填入你的 Triton 实现！"
    )


# ============================================================
# 5. 正确性验证
# ============================================================

def check_correctness(
    config: MoEConfig,
    impl_fn: Callable,
    ref_fn: Callable = pytorch_reference,
    token_counts: list = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """验证 impl_fn 对比 ref_fn 的数值正确性"""
    if token_counts is None:
        token_counts = [1, 4, 16, 64]

    factory = MoEDataFactory(config)
    print("=" * 60)
    print("正确性验证")
    print("=" * 60)
    print(f"Config: H={config.H}, I={config.I}, E_local={config.E_local}, "
          f"E_global={config.E_global}, TOP_K={config.TOP_K}")
    print(f"Tolerances: atol={atol}, rtol={rtol}")
    print("-" * 60)

    all_passed = True
    for T in token_counts:
        data = factory.create(T, seed=42)

        # Reference
        data_ref = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}
        ref_fn(**data_ref, config=config)
        ref_output = data_ref["output"].clone()

        # Implementation
        data_impl = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in data.items()}
        try:
            impl_fn(**data_impl, config=config)
            impl_output = data_impl["output"].clone()

            # 比较
            max_diff = (ref_output.float() - impl_output.float()).abs().max().item()
            mean_diff = (ref_output.float() - impl_output.float()).abs().mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                ref_output.float().flatten().unsqueeze(0),
                impl_output.float().flatten().unsqueeze(0),
            ).item()

            passed = torch.allclose(
                ref_output.float(), impl_output.float(), atol=atol, rtol=rtol
            )
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False

            print(f"T={T:>4d}  {status}  "
                  f"max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  "
                  f"cos_sim={cos_sim:.6f}")

        except NotImplementedError:
            print(f"T={T:>4d}  ⏭️  SKIP (not implemented)")
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("🎉 全部通过！")
    print()
    return all_passed


# ============================================================
# 6. 性能对比（使用 triton.testing.do_bench）
# ============================================================

def run_perf_benchmark(
    config: MoEConfig,
    impl_fn: Callable,
    ref_fn: Callable = pytorch_reference,
    token_counts: list = None,
):
    """性能对比：PyTorch baseline vs Triton 实现"""
    if token_counts is None:
        token_counts = [1, 4, 16, 64, 128, 256, 512]

    factory = MoEDataFactory(config)
    print("=" * 60)
    print("性能对比 (ms)")
    print("=" * 60)
    print(f"{'T':>6s} | {'PyTorch':>12s} | {'Triton':>12s} | {'Speedup':>8s}")
    print("-" * 50)

    for T in token_counts:
        data = factory.create(T, seed=42)

        # 测 PyTorch baseline
        def run_ref():
            d = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in data.items()}
            ref_fn(**d, config=config)

        ref_ms = triton.testing.do_bench(run_ref, warmup=5, rep=20)

        # 测 Triton 实现
        try:
            def run_impl():
                d = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in data.items()}
                impl_fn(**d, config=config)

            impl_ms = triton.testing.do_bench(run_impl, warmup=5, rep=20)
            speedup = ref_ms / impl_ms
            print(f"{T:>6d} | {ref_ms:>10.3f}ms | {impl_ms:>10.3f}ms | {speedup:>7.2f}x")
        except NotImplementedError:
            print(f"{T:>6d} | {ref_ms:>10.3f}ms | {'N/A':>12s} | {'N/A':>8s}")

    print()


# ============================================================
# 7. Triton 官方风格 Benchmark（生成图表）
# ============================================================

def create_triton_benchmark(config: MoEConfig, impl_fn: Callable):
    """用 triton.testing.Benchmark 生成性能对比图"""
    factory = MoEDataFactory(config)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["T"],
            x_vals=[2**i for i in range(0, 10)],  # 1, 2, 4, ..., 512
            line_arg="provider",
            line_vals=["pytorch", "triton"],
            line_names=["PyTorch Baseline", "Triton Fused"],
            styles=[("blue", "-"), ("red", "-")],
            ylabel="ms",
            plot_name=f"fused-moe-fp8-H{config.H}-I{config.I}-E{config.E_local}",
            args={"config": config},
        )
    )
    def bench_fn(T, provider, config):
        data = factory.create(T, seed=42)

        if provider == "pytorch":
            def fn():
                d = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in data.items()}
                pytorch_reference(**d, config=config)
            return triton.testing.do_bench(fn, warmup=5, rep=20)
        else:
            def fn():
                d = {k: v.clone() if isinstance(v, torch.Tensor) else v
                     for k, v in data.items()}
                impl_fn(**d, config=config)
            try:
                return triton.testing.do_bench(fn, warmup=5, rep=20)
            except NotImplementedError:
                return float("nan")

    bench_fn.run(save_path="./bench_results/", print_data=True)


# ============================================================
# 8. 配置预设
# ============================================================

def get_small_config():
    """缩小版配置，快速迭代调试用"""
    return MoEConfig(
        H=512,         # 7168 → 512
        I=256,         # 2048 → 256
       E_global=32,   # 256 → 32
        E_local=4,     # 32 → 4
        TOP_K=4,       # 8 → 4
        N_GROUP=4,     # 8 → 4
        TOPK_GROUP=2,  # 4 → 2
        BLOCK=128,
        local_expert_offset=0,
        routed_scaling_factor=2.5,
    )
def get_medium_config():
    """中等配置，接近真实但跑得动"""
    return MoEConfig(
        H=2048,
        I=1024,
        E_global=64,
        E_local=8,
        TOP_K=8,
        N_GROUP=8,
        TOPK_GROUP=4,
        BLOCK=128,
        local_expert_offset=0,
        routed_scaling_factor=2.5,
    )
def get_full_config():
    """DeepSeek V3/R1 完整配置"""
    return MoEConfig()  # 默认值即完整参数
# ============================================================
# 9. 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fused MoE FP8 Benchmark")
    parser.add_argument(
        "--mode",
        choices=["correctness", "perf", "chart", "both", "quick"],
        default="quick",
        help="运行模式",
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "full"],
        default="small",
        help="small=缩小版快速调试, medium=中等, full=DeepSeek V3 完整尺寸",
    )
    args = parser.parse_args()
    config_map = {
        "small": get_small_config,
        "medium": get_medium_config,
        "full": get_full_config,
    }
    config = config_map[args.size]()
    size_label = {"small": "Small (调试)", "medium": "Medium (中等)", "full": "Full (DeepSeek V3)"}
    print(f"使用配置: {size_label[args.size]}")
    print(f"   H={config.H}, I={config.I}, E_local={config.E_local}, "
          f"E_global={config.E_global}, BLOCK={config.BLOCK}")
    # ↓↓↓ 你的 Triton 实现（替换这里）↓↓↓
    impl = triton_fused_moe
    if args.mode in ("correctness", "both"):
        token_counts = [1, 4, 16] if args.size == "small" else [1, 4, 16, 64]
        check_correctness(config, impl, token_counts=token_counts)
    if args.mode in ("perf", "both"):
        token_counts = (
            [1, 4, 16, 64] if args.size == "small"
            else [1, 4, 16, 64, 128, 256]
        )
        run_perf_benchmark(config, impl, token_counts=token_counts)
    if args.mode == "chart":
        create_triton_benchmark(config, impl)
    if args.mode == "quick":
        # 快速自检：ref vs ref，验证框架本身没 bug
        print("🧪 快速自检：reference vs reference（验证框架正确性）")
        check_correctness(config, pytorch_reference, token_counts=[1, 4, 16])
        print("📊 Reference 性能基线：")
        run_perf_benchmark(config, pytorch_reference, token_counts=[1, 4, 16])