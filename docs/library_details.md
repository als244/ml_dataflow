## libdataflow & libdataflowops

Some of the basic components:

### Dataflow Handle

The core data-structure is a `Dataflow_Handle` (subject to change as this repo reaches maturity). Hardware backends are responsible for supplying an implementation for the API functions contained within this struct. This object is resposible for hardware abstraction, giving a foundation for higher level development.

It is the heart this whole repo. The API functions expose 4 fundamental functionalities:
- Computation Loading/Dispatching
- Handling Depdendencies & Synchronization
- Memory Management
- Data Transfers

***For reference see***:
- Dataflow Handle API: `libdataflow/include/dataflow_handle.h`
- Example Implementation (for Nvidia Devices): `backends/nvidia/src/handle/cuda_dataflow_handle.c`

### Ops 

To be of any use, users will need to register operations with this backend. Operations can be registered as either:
- *Native*
    - Where backend can register and load this representation directly (e.g binary or source code if backend supports JIT)
        - Each native op should be tied to a host function which is responsible for setting the launch configuration at runtime (i.e. can based upon argument values). 
- *External*
    - Where the operation is defined as a wrapper for a thirdparty function responsible for computation dispatching.

Each operation can supply an intialization function that can handle setting an operation's attributes or creating/saving any data that might be required during dispatching.

Upon `submit_op()` the backend should either directly call its own mechanism for launching ops or call the external function.

***For reference see***:
- Op Structure: `libdataflow/include/dataflow_op_structs.h`

### Op Registration & Submission

<sup><em> Note: The registration aspect is currently clunkly will be re-factored in the future. The the extra config library for native ops is annoying handling external ops is awkward, and generally the whole setup requires too many specific path names / symbol names. The plans are to simplify this component to make it simpler for higher level bindings.</em></sup>

Every `Op` must have a corresponding `Op_Skeleton`. This defines:
- Nickname for the operation
- Number of arguments
- Datatypes of arguments

This `Op_Skeleton` should be fingerprinted to create a unique identfier. Now this identifer will be used a key into the handle's Op Table. The value within the Op Table is defined by the backend and will be created during one of the register API calls. The combined Op_Skeleton => 'Op Reference' pair is inserted into the Op Table such that the backend can look up this reference and correctly dispatch upon a `submit_op()` call. 

As a parameter to `submit_op()`, the user passes in a Op struct that has a fully constructed skeleton (this should match the one passed in during registration). Additionally the Op contains an args array that is populated with references (i.e. host memory address where the argument value lives) for each argument.  

***For reference see***: 
- Example of backend-defined Op Refernce: `backends/nvidia/include/cuda_dataflow_handle.h`. The `Cuda_Function` struct is created for every Op and is the value within the Op Table for Nvidia backend. 
- A helper library defines default op interfaces (backend-agnostic) for commonly used machine learning operations: `libdataflowops`
    - This library provides functions for setting the Op Skeletons, which need to be done both at registration time and submission time
    - It also provides functions for interfacing with the `submit_op()` API allowing for a typical looking mechanism for calling functions
- Example of Registering Ops: `backends/nvidia/src/ops/src/register_ops/register_ops.c`
    - Relies upon `libdataflowops` for doing the skeleton setting at registration time. 
    - Relies upon a default implmentation of both native and external ops
        - ***External Ops***: The compute-bound operations. Performance critical.
            - **Matmul**: Lives within `backends/nvidia/src/ops/src/external/src/matmul_helper/matmul_helper.c` and is integrated into the register APIs via the `libmatmulhelper` shared library. The matmul helper relies upon `libcublasLt`. This example demonstrates creating and saving data (the cublas handle) at init time, which is later accessed during runtime dispatching. It also demonstrates that an external function can call an external library. 
            - **Attention**: Lives within `backends/nvidia/src/ops/src/external/src/attention_helper/attention_helper.c` and is integrated into the register APIs via the `libattentionhelper` shared library. The attention helper relies upon source code cloned from the [Flash Attention](https://github.com/Dao-AILab/flash-attention) repository. This C++ repo is then extended with an "extern" function and built into a shared library, `libflash3` (that then attention helper can call). This example demonstrates how to wrap C++ functions that are tightly coupled with backend kernels into an External Op. 
        - ***Native Ops***: The memory-bound operations. Easier to implement well.
            - I've provided a default set of kernels for the Nvidia backend that are sufficient for Transformers. The many different kernels live within `backends/nvidia/src/ops/src/native/src/*.cu`. Each kernel family also has a corresponding config file in order to set the correct launch spec (and a couple have init functions for setting shared memory attributes). All of these kernels get built into single module `cuda_kernels.cubin` and all of the config functions get built itno a single shared library `cuda_kernels_config.so` that can be passed together the register APIs call. 
    - Now the `register_ops.c` file utilizes these implementations to actually perform registration. This is then built into an shared library, contained in the top level `backend/nvidia/lib` directory that is meant for consumers to link with, and exposes one main functino for each `Dataflow_Handle` to obtain access to the default set of operations, `dataflow_register_default_ops()`
    
### TLDR; Registration & Submission

If you are only working with default transformer ops and don't want to specify your own kernel implementations then you do not need to worry about Op Registration and Submission if you link with the helper library `libdataflowops` along with top-level `backend/<type>/lib<backend_register_default_ops`. Once you call `dataflow_register_default_ops()` all of the default ops will be registered. Now you can use the functions defined in defined in `libdataflowops/include/dataflow_ops.h` to actually submit these operations in a normal function-call like manner.


-----


### libdataflowmodels


-----


### libdataflowring



-----