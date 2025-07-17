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
- Chunks per round:
- Seqs per round:
- Rounds per step:
- Seqs per step:
- Global batch size:

- Device Memory Partitioning (model has 32 blocks)
    - Param Blocks: 
    - Grad Blocks: 
    - Total (chunk, layer) full activation slots: 

- Host Activations ($32 * 4 -  = $ total):
    - Fully Saved: 
    - Only Inp + Context + Attn:
    - Only Inp + Context: 0