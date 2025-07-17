# Example Profiling (High I/O Pressure -- Fast Compute + Small/Medium Memory + Short/Medium Seq)

## Training of Llama3 8B with 8k sequence length on H100. 

To reproduce:

```shell
cd bench
./do_transformer_profile.sh 75 40 8192 8
```

This will create an nsys report: `bench/profiling/host_75_dev_40_seq_8192_model_8.nsys-rep` that can be loading in the Nsight Systems GUI.

### Auto-configuration details

- Chunk size: 16384
- Chunks per round: 1
- Seqs per round: 2
- Rounds per step: 34
- Seqs per step: 68
- Global batch size: 68 seqs/544k tokens

- Device Memory Partitioning (model has 32 blocks)
    - Param Blocks: 10
    - Grad Blocks: 9
    - Total (chunk, layer) full activation slots: 9

- Host Activations ($32 * 4 - 9 = 23$ total):
    - Fully Saved: 9
    - Only Inp + Context + Attn: 14
    - Only Inp + Context: 0