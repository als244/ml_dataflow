# *Portable* & *Efficient* Machine Learning

<sup><em>Currently a work in-progress... </em></sup>

You can learn more about the project's background/details [here](docs/background.md)

-----

## Training Performance Demo

You can demo training performance of canonical causal transformer under different memory environments, sequence lengths, or model sizes. The demo program expects users to specify:
- Host Memory Capacity
- Device Memory Capacity
- Sequence Length
- Model Size (llama3 arch for now, either 1B or 8B)


#### Installation & Usage

##### Current hardware support 
- Nvidia GPUs
    - sm80+ (Ampere/Ada, Hopper, Blackwell)

1. *Download this repo*: 

```shell
git clone git@github.com:als244/ml_dataflow.git
```

2. *Build from source*:

```shell
make -j <NUM_PROCS>
```

###### Note that building the flash2 and flash3 wrapper libraries may take some time (a few hours)...using more processors will help. 

3. *Download model checkpoints (llama3 1B and 8B instruct models in raw binary format)*:

```shell
data/download_llama3_model_binaries.sh
```

4. *Test out training*:

```shell
bench/transformerDemo <host_mem_gb> <dev_mem_gb> <seqlen: [num tokens]> <model size billions: [1 | 8]>
```

For example:

`bench/transformerDemo 80 20 16384 1` will train the 1B model architecture (full bf16, causal attention, next token prediction, AdamW). The sequence length is set to 16384 tokens. The memory capacities are set to enforce <= 80 GB of host memory and <= 20 GB of device memory (where XXX GB is defined as XXX * 2^30 bytes).

4b. *Profile the training*

```shell
bench/do_transformer_proifle.sh <host_mem_gb> <dev_mem_gb> <seqlen: [num tokens]> <model size billions: [1 | 8]>
```

This will create a `.nsys-rep` file within `bench/profile` prthat be can loaded into the `Nvidia Sight Systems` GUI. There are nvtx markings that should have intuitive meanings when inspecting the report. 

-----

## Benchmarked Results

TODO: Heatmaps showing throughput vs. system_build + combinations of host_mem/dev_mem/seq_len/model_size

#### 1B



#### 8B



-----

**Practical note**: Critical upstream functionality (*data ingestion*, *model/loss/optimizer customization*, *model saving/loading*, *multi-worker training*, & *a wider set of common kernels such as convolutions and MoE selecting/routing/combining*) is underway. You can try out a [simulator](https://dataflowsim.sunshein.net) for what this repo aims to accomplish in its final multi-worker form.

The plan is to build a robust core of C libraries and create user-friendly Python bindings (at the various layers of stack) for convenient interfacing. Typical usage will have a similar API to most other training frameworks and only need to use the top-level bindings. 

A true interface will be released when the basic dataloading functionality is ready. 

The intial emphasis is for training; after this is working properly, focus will shift to inference. 

-----

## API


***Not ready yet...***

