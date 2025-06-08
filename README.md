# *Portable* & *Efficient* Machine Learning

<sup><em>Currently a work in-progress... </em></sup>

-----

## Purpose

This repo is structured as a collection of libraries to help foster a robust, transparent, & performant ecosystem for machine learning and other accelertor-centric workloads. The transition to heterogeneous computing environments (CPUs + GPUs/TPUs/FPGAs/etc.) has posed challenges for portability and efficency. Dataflow processing, encompassing concurrent streams and asynchronous data-movement, is fundamental to all AI workloads. Yet, we lack a quality way of expressing these types of programs. 

The current ecosystem lies at the extremes:
- ***Optimizing for performance***: Custom-built solutions targeting a specific computing environment (accounting for known: architectures, # of acccelerators, bandwidths, memory capacities, etc.)
    - Lacks portability
    - High development cost
- ***Optimizing for convenience***: High-level autograd frameworks such as PyTorch or JAX
    - Lack mechanisms for precise control over memory, data-movement
        - Painful to manage multiple threads and sync primitives within Python frameworks
    - Large codebases that are diffcult to trace how/when/why system resources are utilized
        - Deeply embedded depedencies on third-party libraries can cause frustration to remove, swap, or update
    - Auto-differentiation is a complex stack: specifying when computations occur, where parameters/activations/gradients/optimizer state are housed, & how they are transferred is not easily controllable. 
        - Packages built on top (such as DDP, ZeRO, FSDP) manage this complexity (making edits to 'contexts' within computation graph), but they too are complex and limited in expressivity
            - Meanwhile, more haziness is added to the system regarding underlying resource usage.
    

The objective is to offer the best of both worlds. 

-----

## Installation & Usage

#### Supported hardware

Currently has support for: 
- Nvidia GPUs
    - >= sm80 (Ampere, Ada, Hopper, Blackwell)

1. *Download this repo*: 

```console
git clone git@github.com:als244/ml_dataflow.git
```

2. *Build from source*:

```console
make -j <NUM_PROCS>
```

###### Note that building the flash2 and flash3 wrapper libraries may take some time (a few hours)...using more processors will help. 

3. *Download model checkpoints (llama3 1B and 8B instruct models in raw binary format)*:

```console
cd test
./download_llama3_model_binaries.sh
```

4. *Test out training*:

```console
cd test/dataflow_models
./transformer
```

**The example training program does finetuning of llama3 8B model with data as 64k sequence length** (the first 65536 tokens of harry potter, where this one sequence is repeated); ***the program expects 1 device with 80 GB of device memory and 200 GB of host memory --- it should achieve ~45-47% MFU on a Nvidia H100***. More general support for dataset ingestion, sequence lengths, hardware/memory flexibility, and other model checkpoints/sizes is coming soon.

----

You can try out a [simulator](https://dataflowsim.sunshein.net) for what this repo aims to accomplish.

Plan is to build a robust core of C libraries and create user-friendly Python bindings (at the various layers of stack) for convenient interfacing. Typical usage will have a similar API to most other training frameworks and only need to use the top-level bindings. 

The intial emphasis is for training; after this is working properly, focus will shift to inference. 



## API


***Not ready yet, hopefully soon!***


## Under the Hood

| Library Structure Flow   |
| :----------------------: |
| :----------------------: |
| Orchestration Manager: **libdataflowring** |
| &#11015;                 |
| Default Models (compositions of default operations): **libdataflowmodels** |
| &#11015;                 |
| Set of Default Operations (interfaces): **libdataflowops** |
| &#11015;                 |
| Hardware Abstraction: **libdataflow** |

----

- ***The goal is for streams/command queues to be a first-class citizen***. This is the major distinction between this 'scaffolding' and others. 
    - While streams can be employed through PyTorch, the actual mechanism for doing so is via a non-intuitive Python context (`with stream(): ...`); expressing desired behavior is challenging, and it is even unclear if the program is actually behaving in inteded manner (especially when employing addtional frameworks on top of base framework). 
    - Instead of treating streams as an add-on feature, the bottom level library (`libdataflow`) explicity requires stream identifiers for every operation and data-transfer.
        - Particularly relevant for managing compute & communication overlap along with fine-grained synchronization

- There will be an initial collection of default functionality (backend implementions, operation interfaces, backend operation implementations, model definitions, & orchestration management). 
    - However, the design allows for plug-and-play customizations to the base set of functionality (additions or replacement). Any new research innovations (whether it be new operations or compositions, improved backend kernels, novel hardware backends, different communication schemes, etc.) can be integrated at its corresponding level of the stack while keeping the other components fixed. 



You can see more details about each of the components [here](docs/library_details.md)

###### Food for Thought

<sup><em> As the world becomes increasingly dependent upon AI, it is essential to maximize hardware throughput; our most precious resources --- <b> time </b> & <b>natural energy</b> --- are on the line...</em></b></sup>
