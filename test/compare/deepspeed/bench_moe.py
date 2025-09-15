import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Assuming scattermoe is installed.
try:
    from scattermoe.mlp import GLUMLP
except ImportError:
    print("⚠️ Warning: 'scattermoe' library not found. Using a placeholder for GLUMLP.")
    print("   The script will run, but the benchmark results will NOT be meaningful.")
    class GLUMLP(nn.Module):
        def __init__(self, input_size, hidden_size, activation, num_experts, top_k):
            super().__init__()
            self.num_experts = num_experts
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Linear(input_size, hidden_size),
                    activation,
                    nn.Linear(hidden_size, input_size)
                ) for _ in range(num_experts)
            ])
        def forward(self, x, routing_weights, selected_experts):
            B, D = x.shape
            output = torch.zeros_like(x)
            for i in range(B):
                for k in range(routing_weights.shape[1]):
                    expert_idx = selected_experts[i, k]
                    weight = routing_weights[i, k]
                    gate_val = self.experts[expert_idx][0](x[i])
                    up_val = self.experts[expert_idx][1](x[i])
                    activated_val = self.experts[expert_idx][2](gate_val * up_val)
                    output[i] += weight * self.experts[expert_idx][3](activated_val)
            return output

class MoEFeedForward(nn.Module):
    def __init__(self, num_experts, model_dim, expert_dim, top_k):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.model_dim, self.expert_dim = model_dim, expert_dim
        self.router = nn.Linear(model_dim, num_experts, bias=False)
        self.moe_layer = GLUMLP(
            input_size=model_dim, hidden_size=expert_dim,
            activation=nn.SiLU(), num_experts=num_experts, top_k=top_k
        )

    def forward(self, x: torch.Tensor):
        router_logits = self.router(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        return self.moe_layer(x, routing_weights, selected_experts)

def benchmark(args):
    """ Main function to run the benchmark with detailed timing. """
    if not torch.cuda.is_available():
        print("🛑 CUDA is not available. Benchmarking on CPU is not meaningful for TFLOPS.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("\n--- Arguments ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print("-----------------------\n")

    # 1. Initialize Model and Data
    model = MoEFeedForward(
        num_experts=args.num_experts, model_dim=args.model_dim,
        expert_dim=args.expert_dim, top_k=args.top_k
    ).to(device, dtype=dtype)
    
    input_tensor = torch.randn(args.chunk_tokens, args.model_dim, device=device, dtype=dtype)
    target_tensor = torch.randn(args.chunk_tokens, args.model_dim, device=device, dtype=dtype)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 2. Warmup Phase
    print(f"🔥 Running {args.warmup_iters} warmup iterations...")
    for _ in range(args.warmup_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # 3. Timed Benchmark Phase with Detailed Timing
    print(f"⏱️  Running {args.num_iters} timed iterations...")
    
    # Create CUDA events for precise timing of each section
    start_event = torch.cuda.Event(enable_timing=True)
    fwd_end_event = torch.cuda.Event(enable_timing=True)
    bwd_end_event = torch.cuda.Event(enable_timing=True)
    iter_end_event = torch.cuda.Event(enable_timing=True)

    total_fwd_time, total_bwd_time, total_iter_time = 0.0, 0.0, 0.0

    for _ in range(args.num_iters):
        optimizer.zero_grad(set_to_none=True)

        start_event.record()

        # Forward Pass
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        fwd_end_event.record()

        # Backward Pass
        loss.backward()
        bwd_end_event.record()

        # Optimizer Step
        optimizer.step()
        iter_end_event.record()
        
        # Synchronize and accumulate timings
        torch.cuda.synchronize()
        total_fwd_time += start_event.elapsed_time(fwd_end_event)
        total_bwd_time += fwd_end_event.elapsed_time(bwd_end_event)
        total_iter_time += start_event.elapsed_time(iter_end_event)

    # Calculate average times in milliseconds
    avg_fwd_ms = total_fwd_time / args.num_iters
    avg_bwd_ms = total_bwd_time / args.num_iters
    avg_iter_ms = total_iter_time / args.num_iters

    # 4. TFLOPS Calculation for each phase
    total_tokens = args.chunk_tokens
    D, E, N, K = (args.model_dim, args.expert_dim, args.num_experts, args.top_k)

    # Forward FLOPs
    flops_router_fwd = 2 * total_tokens * D * N
    flops_glu_fwd = (2 * D * E) + (2 * D * E) + (2 * E * D)
    flops_experts_fwd = total_tokens * K * flops_glu_fwd
    total_flops_fwd = flops_router_fwd + flops_experts_fwd

    # Backward FLOPs are ~2x Forward FLOPs
    total_flops_bwd = 2 * total_flops_fwd
    # Overall FLOPs are ~3x Forward FLOPs
    total_flops_iter = 3 * total_flops_fwd

    # Calculate TFLOPS for each component
    fwd_tflops = total_flops_fwd / ((avg_fwd_ms / 1000) * 1e12)
    bwd_tflops = total_flops_bwd / ((avg_bwd_ms / 1000) * 1e12)
    iter_tflops = total_flops_iter / ((avg_iter_ms / 1000) * 1e12)

    # 5. Print Results
    print("\n--- ⚡ Performance Results ---")
    print(f"{'Phase':<15} | {'Time (ms)':>12} | {'TFLOPS':>10}")
    print(f"-----------------|----------------|-----------")
    print(f"{'Forward Pass':<15} | {avg_fwd_ms:>12.2f} | {fwd_tflops:>10.2f}")
    print(f"{'Backward Pass':<15} | {avg_bwd_ms:>12.2f} | {bwd_tflops:>10.2f}")
    print(f"{'Overall Iter':<15} | {avg_iter_ms:>12.2f} | {iter_tflops:>10.2f}")
    print("------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE Layer FWD+BWD Performance")
    # Use hyphens for CLI arguments for consistency
    parser.add_argument('--chunk_tokens', type=int, default=131072, help='Total number of tokens to process per iteration')
    parser.add_argument('--model_dim', type=int, default=1536, help='Model hidden dimension (D_model)')
    parser.add_argument('--expert_dim', type=int, default=768, help='Expert intermediate dimension (D_expert)')
    parser.add_argument('--num_experts', type=int, default=64, help='Total number of experts')
    parser.add_argument('--top_k', type=int, default=8, help='Number of experts to route each token to')
    parser.add_argument('--num_iters', type=int, default=200, help='Number of timed iterations for the benchmark')
    parser.add_argument('--warmup_iters', type=int, default=20, help='Number of warmup iterations before timing')
    
    args = parser.parse_args()
    benchmark(args)

if __name__ == "__main__":
    main()