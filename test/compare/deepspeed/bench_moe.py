import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

# Assuming scattermoe is installed. If not, this script will still run 
# with a placeholder class, but performance will not be representative.
try:
    from scattermoe.mlp import GLUMLP
except ImportError:
    print("⚠️ Warning: 'scattermoe' library not found. Using a placeholder for GLUMLP.")
    print("   The script will run, but the benchmark results will NOT be meaningful.")
    # This is a placeholder to allow the script to be syntactically correct
    # without the actual scattermoe library. It is not an efficient MoE implementation.
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
            # A real MoE layer uses efficient scatter/gather operations.
            # This placeholder simulates the computation for a single token to ensure
            # the forward pass completes with the correct tensor shapes.
            B, D = x.shape
            output = torch.zeros_like(x)
            for i in range(B): # Iterate over each token
                for k in range(routing_weights.shape[1]): # Iterate over top_k experts
                    expert_idx = selected_experts[i, k]
                    weight = routing_weights[i, k]
                    # Simulate GLU: (gate(x) * up(x)) @ down
                    gate_val = self.experts[expert_idx][0](x[i])
                    up_val = self.experts[expert_idx][1](x[i])
                    activated_val = self.experts[expert_idx][2](gate_val * up_val)
                    output[i] += weight * self.experts[expert_idx][3](activated_val)
            return output

class MoEFeedForward(nn.Module):
    """
    The Mixture of Experts module based on the provided skeleton.
    """
    def __init__(self, num_experts, model_dim, expert_dim, top_k):    
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.model_dim = model_dim
        self.expert_dim = expert_dim

        # The router sends tokens to experts.
        self.router = nn.Linear(model_dim, num_experts, bias=False)

        # The efficient MoE layer from scattermoe.
        self.moe_layer = GLUMLP(
            input_size=model_dim, 
            hidden_size=expert_dim, 
            activation=nn.SiLU(), 
            num_experts=num_experts, 
            top_k=top_k
        )

    def forward(self, x: torch.Tensor):
        # x shape: (total_tokens, hidden_dim)
        batch_size, sequence_length, hidden_dim = x.shape

        # Flatten input for routing and expert processing.
        x_flat = x.view(-1, hidden_dim) # Shape: (batch*seq_len, hidden_dim)
        
        # Get routing logits from the router.
        router_logits = self.router(x_flat)

        # Apply softmax to get routing weights and select the top-k experts.
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize the weights of the selected experts.
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # Cast weights back to the input's dtype for the MoE layer.
        routing_weights = routing_weights.to(x.dtype)
        
        # Pass tokens, weights, and expert indices to the efficient MoE layer.
        final_hidden_states_flat = self.moe_layer(x_flat, routing_weights, selected_experts)

        # Reshape the output back to the original input shape.
        final_hidden_states = final_hidden_states_flat.view(batch_size, sequence_length, hidden_dim)

        return final_hidden_states

def benchmark(args):
    """
    Main function to run the benchmark.
    """
    if not torch.cuda.is_available():
        print("🛑 CUDA is not available. Benchmarking on CPU is not meaningful for TFLOPS.")
        return
        
    device = torch.device("cuda")
    # Use bfloat16 for better performance on modern GPUs.
    dtype = torch.bfloat16
    
    print("\n--- Arguments ---")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    print("-----------------------\n")

    # 1. Initialize Model and Data
    model = MoEFeedForward(
        num_experts=args.num_experts,
        model_dim=args.model_dim,
        expert_dim=args.expert_dim,
        top_k=args.top_k
    ).to(device, dtype=dtype)
    
    input_tensor = torch.randn(
        args.batch_size, args.sequence_length, args.model_dim, 
        device=device, dtype=dtype
    )
    target_tensor = torch.randn(
        args.batch_size, args.sequence_length, args.model_dim, 
        device=device, dtype=dtype
    )
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 2. Warmup Phase (to handle CUDA initialization overhead)
    print(f"🔥 Running {args.warmup_iters} warmup iterations...")
    for _ in range(args.warmup_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # 3. Timed Benchmark Phase
    print(f"⏱️  Running {args.num_iters} timed iterations...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(args.num_iters):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_s = (elapsed_time_ms / 1000) / args.num_iters
    print(f"   Average iteration time: {avg_time_s * 1000:.3f} ms")

    # 4. TFLOPS Calculation
    # A standard rule of thumb is that the backward pass is ~2x the FLOPs of the forward pass.
    # Total FLOPs (fwd + bwd) ≈ 3 * FLOPs_fwd.
    
    B, S, D, E, N, K = (
        args.batch_size, args.sequence_length, args.model_dim,
        args.expert_dim, args.num_experts, args.top_k
    )
    
    total_tokens = B * S

    # FLOPs for the router (a single Linear layer)
    flops_router_fwd = 2 * total_tokens * D * N
    
    # FLOPs for the experts (GLU consists of 3 Linear layers)
    # Each of the K experts processes the token.
    flops_glu_fwd = (
        (2 * D * E) +  # Gate projection (D -> E)
        (2 * D * E) +  # Up projection (D -> E)
        (2 * E * D)    # Down projection (E -> D)
    )
    flops_experts_fwd = total_tokens * K * flops_glu_fwd
    
    total_flops_fwd = flops_router_fwd + flops_experts_fwd
    total_flops_fwd_bwd = 3 * total_flops_fwd
    
    tflops = total_flops_fwd_bwd / (avg_time_s * 1e12)
    
    print("\n--- ⚡ Performance Results ---")
    print(f"   Achieved TFLOPS: {tflops:.2f}")
    print("-----------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE Layer FWD+BWD Performance")
    # Common settings inspired by models like Mixtral 8x7B
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for the benchmark')
    parser.add_argument('--sequence-length', type=int, default=65536, help='Sequence length of the input')
    parser.add_argument('--model-dim', type=int, default=1536, help='Model hidden dimension (D_model)')
    parser.add_argument('--expert-dim', type=int, default=768, help='Expert intermediate dimension (D_expert)')
    parser.add_argument('--num-experts', type=int, default=64, help='Total number of experts')
    parser.add_argument('--top-k', type=int, default=8, help='Number of experts to route each token to')
    parser.add_argument('--num-iters', type=int, default=100, help='Number of timed iterations for the benchmark')
    parser.add_argument('--warmup-iters', type=int, default=10, help='Number of warmup iterations before timing')
    
    args = parser.parse_args()
    benchmark(args)

if __name__ == "__main__":
    main()