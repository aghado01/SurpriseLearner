torch
Tensors
Creation Ops
Indexing, Slicing, Joining, Mutating Ops


Accelerators
Generators
Random sampling
torch.default_generator
In-place random sampling
Quasi-random sampling


Serialization
Parallelism
Locally disabling gradient computation
Math operations
Constants
Pointwise Ops
Reduction Ops
Comparison Ops
Spectral Ops
Other Operations
BLAS and LAPACK Operations
Foreach Operations


Utilities
Symbolic Numbers
SymInt
SymInt.as_integer_ratio()


SymFloat
SymFloat.as_integer_ratio()
SymFloat.conjugate()
SymFloat.hex()
SymFloat.is_integer()


SymBool


Export Path
Control Flow
Optimizations
Operator Tags
Tag
Tag.name
PyTorch Governance | Build + CI
How to Add a New Maintainer
PyTorch Contribution Guide
Contribution Process
Getting Started
Proposing New Features
Reporting Issues
Implementing Features or Fixing Bugs
Adding Tutorials
Improving Documentation & Tutorials
Participating in Online Discussions
Submitting Pull Requests to Fix Open Issues
Reviewing Open Pull Requests
Improving Code Readability
Adding Test Cases to Make the Codebase More Robust
Promoting PyTorch
Triaging Issues


About Open Source Development
Common Mistakes To Avoid
Frequently Asked Questions
On Documentation
Python Docs
C++ Docs


Tutorials
Tutorials Build Overview
Contributing a New Tutorial
PyTorch Design Philosophy
Design Principles
Principle 1: Usability over Performance
Principle 2: Simple Over Easy
Principle 3: Python First with Best In Class Language Interoperability
PyTorch Governance | Mechanics
Summary
Module Maintainers
Core Maintainers
Lead Core Maintainer (BDFL)
Nominating, Confirming and Removing Maintainers
The Principles
The Process for Nomination
The Process for Removal
Nominating Core Maintainers
Removing the Lead Core Maintainer and Nominating a New Lead Core Maintainer


Add, Remove, and Re-Scope Modules and Projects
Decision Making
Uncontroversial Changes
Controversial Decision Process
General Project Policies


FAQ
PyTorch Governance | Maintainers
Responsibilities
Lead Core Maintainer (BDFL)
Core Maintainers
Module-level maintainers
NN APIs (torch.nn)
Optimizers (torch.optim)
Autograd (torch.autograd)
TorchDynamo
TorchInductor
Cudagraph Tree
PT2 Dispatcher
PT2 Export (torch.export)
AOT Inductor (AOTI) & AOTI Runtime
Compilers (JIT / TorchScript / Package / Deploy)
Distributions & RNG
Distributed
Multiprocessing
Linear Algebra (torch.linalg)
Sparse (torch.sparse)
NestedTensor (torch.nested)
MaskedTensor (torch.masked)
Fast Fourier Transform (torch.fft)
MKLDNN
CUDA
AMD/ROCm/HIP
Build + CI
Performance Tools
C++ API
C10 utils and operator dispatch
ONNX exporter
LiteInterpreter
Quantization (torch/ao)
Windows
Apple M1/MPS/Metal
PowerPC
x86 CPU
AArch64 CPU
Docs / Tutorials


Library-level maintainers
XLA
TorchServe
TorchVision
TorchText
TorchAudio
TorchRec
TorchX
TorchData
TorchArrow
ExecuTorch (Edge, Mobile)
TorchTune
TorchChat
TorchCodec
Automatic Mixed Precision examples
Typical Mixed Precision Training
Working with Unscaled Gradients
Gradient clipping


Working with Scaled Gradients
Gradient accumulation
Gradient penalty


Working with Multiple Models, Losses, and Optimizers
Working with Multiple GPUs
DataParallel in a single process
DistributedDataParallel, one GPU per process
DistributedDataParallel, multiple GPUs per process


Autocast and Custom Autograd Functions
Functions with multiple inputs or autocastable ops
Functions that need a particular dtype
Autograd mechanics
How autograd encodes the history
Saved tensors


Gradients for non-differentiable functions
Locally disabling gradient computation
Setting requires_grad
Grad Modes
Default Mode (Grad Mode)
No-grad Mode
Inference Mode
Evaluation Mode (nn.Module.eval())


In-place operations with autograd
In-place correctness checks


Multithreaded Autograd
Concurrency on CPU
Non-determinism
Graph retaining
Thread Safety on Autograd Node
No thread safety on C++ hooks


Autograd for Complex Numbers
What are complex derivatives?
Wirtinger Calculus comes into the picture …
How is Wirtinger Calculus useful in optimization?
How does PyTorch compute the conjugate Wirtinger derivative?
How can I write my own derivative formula for a complex function?
What about cross-domain functions?


Hooks for saved tensors
Registering hooks for a saved tensor
Registering default hooks for saved tensors


Backward Hooks execution
Whether a particular hook will be fired
The order in which the different hooks are fired
Special hooks
Behavior of Tensor hooks when Tensor is modified in-place
Broadcasting semantics
General semantics
In-place semantics
Backwards compatibility
CPU threading and TorchScript inference
Build options
Runtime API
Tuning the number of threads
CUDA semantics
TensorFloat-32 (TF32) on Ampere (and later) devices
Reduced Precision Reduction in FP16 GEMMs
Reduced Precision Reduction in BF16 GEMMs
Full FP16 Accmumulation in FP16 GEMMs
Asynchronous execution
CUDA streams
Stream semantics of backward passes
BC note: Using grads on the default stream




Memory management
Optimizing memory usage  with PYTORCH_CUDA_ALLOC_CONF


Using custom memory allocators for CUDA
Mixing different CUDA system allocators in the same program
cuBLAS workspaces
cuFFT plan cache
Just-in-Time Compilation
Best practices
Device-agnostic code
Use pinned memory buffers
Use nn.parallel.DistributedDataParallel instead of multiprocessing or nn.DataParallel


CUDA Graphs
Why CUDA Graphs?
PyTorch API
Constraints
Non-constraints


Whole-network capture
Partial-network capture
Usage with torch.cuda.amp
Usage with multiple streams
Usage with DistributedDataParallel
NCCL < 2.9.6
NCCL >= 2.9.6


Graph memory management
Sharing memory across captures
PyTorch Custom Operators Landing Page
Distributed Data Parallel
Example
Internal Design
Implementation
ProcessGroup
DistributedDataParallel
TorchDynamo DDPOptimizer
Extending PyTorch
Adding new operators
Extending torch.autograd
When to use
When not to use
How to use
Example
Combined or separate forward() and setup_context()
Forward mode AD
torch.func transforms and/or torch.vmap()


Extending torch.nn
Adding a Module


Extending torch Python API
Extending torch with a Tensor-like type
Subclassing torch.Tensor
Extending torch with a Tensor wrapper type
Operations on multiple types that define __torch_function__
Testing Coverage of Overrides for the PyTorch API


Extending torch native API
__torch_dispatch__ calling convention


Extending all torch API with Modes
Extending torch.func with autograd.Function
Basic Usage
Example 1: autograd.Function calls into another system
Example 2: autograd.Function specifies custom gradient rules
Limitations and gotchas


torch.vmap() Support
Automatically generate a vmap rule
Defining the vmap staticmethod


torch.func.jvp() Support
Frequently Asked Questions
My model reports “cuda runtime error(2): out of memory”
My GPU memory isn’t freed properly
My out of memory exception handler can’t allocate memory
My data loader workers return identical random numbers
My recurrent network doesn’t work with data parallelism
FSDP Notes
FSDP Prefetch Nuances
Communication payload size
FSDP buffers sizes
Getting Started on Intel GPU
Hardware Prerequisite
Software Prerequisite
Installation
Binaries
From Source


Check availability for Intel GPU
Minimum Code Change
Examples
Inference Examples
Inference with FP32
Inference with AMP
Inference with torch.compile


Training Examples
Train with FP32
Train with AMP
Train with torch.compile
Gradcheck mechanics
Notations and background information
Default backward mode gradcheck behavior
Real-to-real functions
Default real input numerical evaluation
Default real input analytical evaluation


Complex-to-real functions
Default complex input numerical evaluation
Default complex input analytical evaluation


Functions with complex outputs


Fast backward mode gradcheck
Fast gradcheck for real-to-real functions
Fast gradcheck for complex-to-real functions
Fast complex input numerical evaluation
Fast complex input analytical evaluation
Why not use a complex uuu


Fast gradcheck for functions with complex outputs


Gradgradcheck implementation
HIP (ROCm) semantics
HIP Interfaces Reuse the CUDA Interfaces
Checking for HIP
TensorFloat-32(TF32) on ROCm
Memory management
hipBLAS workspaces
hipFFT/rocFFT plan cache
torch.distributed backends
CUDA API to HIP API mappings in C++
Refer to CUDA Semantics doc
Enabling kernel asserts
Features for large-scale deployments
Fleet-wide operator profiling
API usage logging
Attaching metadata to saved TorchScript models
Build environment considerations
Common extension points
LibTorch Stable ABI
Modules
A Simple Custom Module
Modules as Building Blocks
Neural Network Training with Modules
Module State
Module Initialization
Module Hooks
Advanced Features
Distributed Training
Profiling Performance
Improving Performance with Quantization
Improving Memory Usage with Pruning
Parametrizations
Transforming Modules with FX
MPS backend
Multiprocessing best practices
CUDA in multiprocessing
Best practices and tips
Avoiding and fighting deadlocks
Reuse buffers passed through a Queue
Asynchronous multiprocess training (e.g. Hogwild)
Hogwild




CPU in multiprocessing
CPU oversubscription
Avoid CPU oversubscription
Numerical accuracy
Batched computations or slice computations
Extremal values
Linear algebra (torch.linalg)
Non-finite values
Extremal values in linalg


TensorFloat-32(TF32) on Nvidia Ampere (and later) devices
Reduced Precision Reduction for FP16 and BF16 GEMMs
Reduced Precision Reduction for FP16 and BF16 in Scaled Dot Product Attention (SDPA)
Reduced Precision FP16 and BF16 GEMMs and Convolutions on AMD Instinct MI200 devices
Reproducibility
Controlling sources of randomness
PyTorch random number generator
Python
Random number generators in other libraries
CUDA convolution benchmarking


Avoiding nondeterministic algorithms
CUDA convolution determinism
CUDA RNN and LSTM
Filling uninitialized memory


DataLoader
Serialization semantics
Saving and loading tensors
Saving and loading tensors preserves views
Saving and loading torch.nn.Modules
Serialized file format for torch.save
torch.load with weights_only=True
Troubleshooting weights_only
Getting unsafe globals
Environment Variables




Serializing torch.nn.Modules and loading them in C++
Saving and loading ScriptModules across PyTorch versions
torch.div performing integer division
torch.full always inferring a float dtype


Utility functions
register_package()
get_crc32_options()
set_crc32_options()
get_default_load_endianness()
set_default_load_endianness()
get_default_mmap_options()
set_default_mmap_options()
add_safe_globals()
clear_safe_globals()
get_safe_globals()
get_unsafe_globals_in_checkpoint()
safe_globals
skip_data


Config
Windows FAQ
Building from source
Include optional components
Speeding CUDA build for Windows
One key install script


Extension
CFFI Extension
Cpp Extension


Installation
Package not found in win-32 channel.
Import error


Usage (multiprocessing)
Multiprocessing error without if-clause protection
Multiprocessing error “Broken pipe”
Multiprocessing error “driver shut down”
CUDA IPC operations
C++
TorchScript C++ API
Extending PyTorch and TorchScript with C++ Extensions
Tensor and Autograd in C++
Authoring Models in C++
Packaging for C++
torch::deploy has been moved to pytorch/multipy
torch.nn
Containers


Convolution Layers
Pooling layers
Padding Layers
Non-linear Activations (weighted sum, nonlinearity)
Non-linear Activations (other)
Normalization Layers
Recurrent Layers
Transformer Layers
Linear Layers
Dropout Layers
Sparse Layers
Distance Functions
Loss Functions
Vision Layers
Shuffle Layers
DataParallel Layers (multi-GPU, distributed)
Utilities


Quantized Functions
Lazy Modules Initialization
Aliases
torch.nn.functional
Convolution functions
Pooling functions
Attention Mechanisms
Non-linear activation functions
Linear functions
Dropout functions
Sparse functions
Distance functions
Loss functions
Vision functions
DataParallel functions (multi-GPU, distributed)
data_parallel
torch.Tensor
Data types
Initializing and basic operations
Tensor class reference
Tensor
Tensor.__init__()
Tensor.T
Tensor.H
Tensor.mT
Tensor.mH
Tensor Attributes
torch.dtype
dtype


torch.device
device


torch.layout
layout


torch.memory_format
memory_format
Tensor Views
Automatic Mixed Precision package - torch.amp
Autocasting
is_autocast_available()
autocast
custom_fwd()
custom_bwd()
autocast
custom_fwd()
custom_bwd()
autocast


Gradient Scaling
GradScaler
GradScaler


Autocast Op Reference
Op Eligibility
CUDA Op-Specific Behavior
CUDA Ops that can autocast to float16
CUDA Ops that can autocast to float32
CUDA Ops that promote to the widest input type
Prefer binary_cross_entropy_with_logits over binary_cross_entropy


XPU Op-Specific Behavior (Experimental)
XPU Ops that can autocast to float16
XPU Ops that can autocast to float32
XPU Ops that promote to the widest input type


CPU Op-Specific Behavior
CPU Ops that can autocast to bfloat16
CPU Ops that can autocast to float32
CPU Ops that promote to the widest input type
Automatic differentiation package - torch.autograd
Forward-mode Automatic Differentiation
Functional higher level API
Locally disabling gradient computation
Default gradient layouts
Manual gradient layouts


In-place operations on Tensors
In-place correctness checks


Variable (deprecated)
Tensor autograd functions
Function
Function


Context method mixins
Custom Function utilities


Numerical gradient checking
Profiler
profile
emit_nvtx
emit_itt


Debugging and anomaly detection
detect_anomaly
set_detect_anomaly


Autograd graph
saved_tensors_hooks
save_on_cpu
disable_saved_tensors_hooks
register_multi_grad_hook
allow_mutation_on_saved_tensors
GradientEdge
get_gradient_edge()
torch.library
Testing custom ops
opcheck()


Creating new custom ops in Python
custom_op()
triton_op()
wrap_triton()


Extending custom ops (created from Python or C++)
register_kernel()
register_autocast()
register_autograd()
register_fake()
register_vmap()
impl_abstract()
get_ctx()
register_torch_dispatch()
infer_schema()
CustomOpDef
CustomOpDef.set_kernel_enabled()




Low-level APIs
Library
Library.define()
Library.fallback()
Library.impl()


fallthrough_kernel()
define()
impl()
torch.accelerator
torch.cpu
Streams and events
torch.cuda
Random Number Generator
Communication collectives
Streams and events
Graphs (beta)
Memory management
use_mem_pool


NVIDIA Tools Extension (NVTX)
Jiterator (beta)
TunableOp
Stream Sanitizer (prototype)
GPUDirect Storage (prototype)
Understanding CUDA Memory Usage
Generating a Snapshot
Using the visualizer
Active Memory Timeline
Allocator State History


Snapshot API Reference
_record_memory_history()
_snapshot()
_dump_snapshot()
torch.mps
MPS Profiler
MPS Event
torch.xpu
Random Number Generator
Streams and events
Memory management
torch.mtia
Streams and events
torch.mtia.memory
Meta device
Idioms for working with meta tensors
torch.backends
torch.backends.cpu
get_cpu_capability()


torch.backends.cuda
is_built()
allow_tf32
allow_fp16_reduced_precision_reduction
allow_bf16_reduced_precision_reduction
cufft_plan_cache
size
max_size
clear()
preferred_blas_library()
preferred_rocm_fa_library()
preferred_linalg_library()
SDPAParams
flash_sdp_enabled()
enable_mem_efficient_sdp()
mem_efficient_sdp_enabled()
enable_flash_sdp()
math_sdp_enabled()
enable_math_sdp()
fp16_bf16_reduction_math_sdp_allowed()
allow_fp16_bf16_reduction_math_sdp()
cudnn_sdp_enabled()
enable_cudnn_sdp()
is_flash_attention_available()
can_use_flash_attention()
can_use_efficient_attention()
can_use_cudnn_attention()
sdp_kernel()


torch.backends.cudnn
version()
is_available()
enabled
allow_tf32
deterministic
benchmark
benchmark_limit


torch.backends.cusparselt
version()
is_available()


torch.backends.mha
get_fastpath_enabled()
set_fastpath_enabled()


torch.backends.mps
is_available()
is_built()


torch.backends.mkl
is_available()
verbose


torch.backends.mkldnn
is_available()
verbose


torch.backends.nnpack
is_available()
flags()
set_flags()


torch.backends.openmp
is_available()


torch.backends.opt_einsum
is_available()
get_opt_einsum()
enabled
strategy


torch.backends.xeon
torch.export
Overview
Existing frameworks


Exporting a PyTorch Model
An Example
Non-Strict Export
Export for Training and Inference
Expressing Dynamism
Serialization
Specializations
Input Tensor Shapes
Python Primitives
Python Containers




Limitations of torch.export
Graph Breaks
Data/Shape-Dependent Control Flow
Missing Fake/Meta/Abstract Kernels for Operators


Read More


API Reference
export()
save()
load()
register_dataclass()
Dim()
default_decompositions()
dims()
ShapesCollection
ShapesCollection.dynamic_shapes()


refine_dynamic_shapes_from_suggested_fixes()
Constraint
ExportedProgram
ExportedProgram.module()
ExportedProgram.buffers()
ExportedProgram.named_buffers()
ExportedProgram.parameters()
ExportedProgram.named_parameters()
ExportedProgram.run_decompositions()


ExportBackwardSignature
ExportGraphSignature
ModuleCallSignature
ModuleCallEntry
CustomDecompTable
CustomDecompTable.copy()
CustomDecompTable.items()
CustomDecompTable.keys()
CustomDecompTable.materialize()
CustomDecompTable.pop()
CustomDecompTable.update()


InputKind
InputSpec
OutputKind
OutputSpec
SymIntArgument
SymBoolArgument
SymFloatArgument
ExportGraphSignature
ExportGraphSignature.replace_all_uses()
ExportGraphSignature.get_replace_hook()


CustomObjArgument
FlatArgsAdapter
FlatArgsAdapter.adapt()


InterpreterModule
InterpreterModuleDispatcher
unflatten()
move_to_device_pass()
Distributed communication package - torch.distributed
Backends
Backends that come with PyTorch
Which backend to use?
Common environment variables
Choosing the network interface to use
Other NCCL environment variables




Basics
Initialization
is_available()
init_process_group()
init_device_mesh()
is_initialized()
is_mpi_available()
is_nccl_available()
is_gloo_available()
is_xccl_available()
is_torchelastic_launched()
TCP initialization
Shared file-system initialization
Environment variable initialization


Post-Initialization
Backend
Backend.register_backend()


get_backend()
get_rank()
get_world_size()


Shutdown
Reinitialization


Groups
new_group()
get_group_rank()
get_global_rank()
get_process_group_ranks()


DeviceMesh
DeviceMesh
DeviceMesh.from_group()
DeviceMesh.get_all_groups()
DeviceMesh.get_coordinate()
DeviceMesh.get_group()
DeviceMesh.get_local_rank()
DeviceMesh.get_rank()




Point-to-point communication
send()
recv()
isend()
irecv()
send_object_list()
recv_object_list()
batch_isend_irecv()
P2POp


Synchronous and asynchronous collective operations
Collective functions
broadcast()
broadcast_object_list()
all_reduce()
reduce()
all_gather()
all_gather_into_tensor()
all_gather_object()
gather()
gather_object()
scatter()
scatter_object_list()
reduce_scatter()
reduce_scatter_tensor()
all_to_all_single()
all_to_all()
barrier()
monitored_barrier()
Work
Work.boxed()
Work.exception()
Work.get_future()
Work.get_future_result()
Work.is_completed()
Work.is_success()
Work.result()
Work.source_rank()
Work.synchronize()
Work.unbox()
Work.wait()


ReduceOp
reduce_op


Distributed Key-Value Store
Store
Store.__init__()
Store.add()
Store.append()
Store.check()
Store.compare_set()
Store.delete_key()
Store.get()
Store.has_extended_api()
Store.multi_get()
Store.multi_set()
Store.num_keys()
Store.set()
Store.set_timeout()
Store.timeout
Store.wait()


TCPStore
TCPStore.__init__()
TCPStore.host
TCPStore.libuvBackend
TCPStore.port


HashStore
HashStore.__init__()


FileStore
FileStore.__init__()
FileStore.path


PrefixStore
PrefixStore.__init__()
PrefixStore.underlying_store




Profiling Collective Communication
Multi-GPU collective functions
Third-party backends
Launch utility
Spawn utility
Debugging torch.distributed applications
Python Breakpoint
Monitored Barrier
TORCH_DISTRIBUTED_DEBUG


Logging
DistError
DistBackendError
DistNetworkError
DistStoreError
breakpoint()
torch.distributed.tensor
PyTorch DTensor (Distributed Tensor)
DTensor Class APIs
DTensor
DTensor.__create_chunk_list__()
DTensor.from_local()
DTensor.full_tensor()
DTensor.redistribute()
DTensor.to_local()
DTensor.device_mesh
DTensor.placements




DeviceMesh as the distributed communicator
DTensor Placement Types
Shard
Shard.dim


Replicate
Partial
Partial.reduce_op


Placement
Placement.is_partial()
Placement.is_replicate()
Placement.is_shard()






Different ways to create a DTensor
Create DTensor from a logical torch.Tensor
distribute_tensor()
distribute_module()


DTensor Factory Functions
zeros()
ones()
empty()
full()
rand()
randn()




Debugging
Logging
Debugging Tools
CommDebugMode
CommDebugMode.generate_comm_debug_tracing_table()
CommDebugMode.generate_json_dump()
CommDebugMode.get_comm_counts()
CommDebugMode.get_parameter_info()
CommDebugMode.get_sharding_info()
CommDebugMode.get_total_counts()
CommDebugMode.log_comm_debug_tracing_table_to_file()


visualize_sharding()




Experimental Features
context_parallel()
local_map()
register_sharding()
Generic Join Context Manager
Join
Join.notify_join_context()


Joinable
Joinable.join_device
Joinable.join_hook()
Joinable.join_process_group


JoinHook
JoinHook.main_hook()
JoinHook.post_hook()
Torch Distributed Elastic
Get Started
Documentation
FullyShardedDataParallel
FullyShardedDataParallel
FullyShardedDataParallel.apply()
FullyShardedDataParallel.check_is_root()
FullyShardedDataParallel.clip_grad_norm_()
FullyShardedDataParallel.flatten_sharded_optim_state_dict()
FullyShardedDataParallel.forward()
FullyShardedDataParallel.fsdp_modules()
FullyShardedDataParallel.full_optim_state_dict()
FullyShardedDataParallel.get_state_dict_type()
FullyShardedDataParallel.module
FullyShardedDataParallel.named_buffers()
FullyShardedDataParallel.named_parameters()
FullyShardedDataParallel.no_sync()
FullyShardedDataParallel.optim_state_dict()
FullyShardedDataParallel.optim_state_dict_to_load()
FullyShardedDataParallel.register_comm_hook()
FullyShardedDataParallel.rekey_optim_state_dict()
FullyShardedDataParallel.scatter_full_optim_state_dict()
FullyShardedDataParallel.set_state_dict_type()
FullyShardedDataParallel.shard_full_optim_state_dict()
FullyShardedDataParallel.sharded_optim_state_dict()
FullyShardedDataParallel.state_dict_type()
FullyShardedDataParallel.summon_full_params()


BackwardPrefetch
ShardingStrategy
MixedPrecision
CPUOffload
StateDictConfig
FullStateDictConfig
ShardedStateDictConfig
LocalStateDictConfig
OptimStateDictConfig
FullOptimStateDictConfig
ShardedOptimStateDictConfig
LocalOptimStateDictConfig
StateDictSettings
torch.distributed.fsdp.fully_shard
PyTorch FSDP2 (fully_shard)
fully_shard()
FSDPModule
FSDPModule.reshard()
FSDPModule.set_all_reduce_hook()
FSDPModule.set_is_last_backward()
FSDPModule.set_modules_to_backward_prefetch()
FSDPModule.set_modules_to_forward_prefetch()
FSDPModule.set_post_optim_event()
FSDPModule.set_reduce_scatter_divide_factor()
FSDPModule.set_requires_all_reduce()
FSDPModule.set_requires_gradient_sync()
FSDPModule.set_reshard_after_backward()
FSDPModule.set_unshard_in_backward()
FSDPModule.unshard()


UnshardHandle
UnshardHandle.wait()


register_fsdp_forward_method()
MixedPrecisionPolicy
OffloadPolicy
CPUOffloadPolicy
Tensor Parallelism - torch.distributed.tensor.parallel
parallelize_module()
ColwiseParallel
RowwiseParallel
SequenceParallel
PrepareModuleInput
PrepareModuleOutput
loss_parallel()
Distributed Optimizers
DistributedOptimizer
DistributedOptimizer.step()


PostLocalSGDOptimizer
PostLocalSGDOptimizer.load_state_dict()
PostLocalSGDOptimizer.state_dict()
PostLocalSGDOptimizer.step()


ZeroRedundancyOptimizer
ZeroRedundancyOptimizer.add_param_group()
ZeroRedundancyOptimizer.consolidate_state_dict()
ZeroRedundancyOptimizer.join_device
ZeroRedundancyOptimizer.join_hook()
ZeroRedundancyOptimizer.join_process_group
ZeroRedundancyOptimizer.load_state_dict()
ZeroRedundancyOptimizer.state_dict()
ZeroRedundancyOptimizer.step()
Pipeline Parallelism
Why Pipeline Parallel?
What is torch.distributed.pipelining?
Step 1: build PipelineStage
Step 2: use PipelineSchedule for execution
Options for Splitting a Model
Option 1: splitting a model manually
Option 2: splitting a model automatically


Hugging Face Examples
Technical Deep Dive
How does the pipeline API split a model?


Implementing Your Own Schedule
Logging
API Reference
Model Split APIs
SplitPoint
pipeline()
Pipe
pipe_split()


Microbatch Utilities
TensorChunkSpec
split_args_kwargs_into_chunks()
merge_chunks()


Pipeline Stages
PipelineStage
build_stage()


Pipeline Schedules
ScheduleGPipe
Schedule1F1B
ScheduleInterleaved1F1B
ScheduleLoopedBFS
ScheduleInterleavedZeroBubble
ScheduleZBVZeroBubble
PipelineScheduleSingle
PipelineScheduleSingle.step()


PipelineScheduleMulti
PipelineScheduleMulti.step()
Distributed Checkpoint - torch.distributed.checkpoint
Additional resources:
AsyncCheckpointerType
save()
async_save()
save_state_dict()
load()
load_state_dict()
AsyncStager
AsyncStager.should_synchronize_after_execute
AsyncStager.stage()
AsyncStager.synchronize_staging()


BlockingAsyncStager
BlockingAsyncStager.stage()
BlockingAsyncStager.synchronize_staging()


Stateful
Stateful.load_state_dict()
Stateful.state_dict()


StorageReader
StorageReader.prepare_global_plan()
StorageReader.prepare_local_plan()
StorageReader.read_data()
StorageReader.read_metadata()
StorageReader.reset()
StorageReader.set_up_storage_reader()
StorageReader.validate_checkpoint_id()


StorageWriter
StorageWriter.finish()
StorageWriter.prepare_global_plan()
StorageWriter.prepare_local_plan()
StorageWriter.reset()
StorageWriter.set_up_storage_writer()
StorageWriter.storage_meta()
StorageWriter.validate_checkpoint_id()
StorageWriter.write_data()


LoadPlanner
LoadPlanner.commit_tensor()
LoadPlanner.create_global_plan()
LoadPlanner.create_local_plan()
LoadPlanner.finish_plan()
LoadPlanner.load_bytes()
LoadPlanner.resolve_bytes()
LoadPlanner.resolve_tensor()
LoadPlanner.set_up_planner()


LoadPlan
ReadItem
SavePlanner
SavePlanner.create_global_plan()
SavePlanner.create_local_plan()
SavePlanner.finish_plan()
SavePlanner.resolve_data()
SavePlanner.set_up_planner()


SavePlan
WriteItem
WriteItem.tensor_storage_size()


FileSystemReader
FileSystemReader.checkpoint_id


FileSystemWriter
FileSystemWriter.stage()


DefaultSavePlanner
DefaultSavePlanner.lookup_object()
DefaultSavePlanner.transform_object()


DefaultLoadPlanner
DefaultLoadPlanner.lookup_tensor()
DefaultLoadPlanner.transform_tensor()


get_state_dict()
get_model_state_dict()
get_optimizer_state_dict()
set_state_dict()
set_model_state_dict()
set_optimizer_state_dict()
StateDictOptions
dcp_to_torch_save()
torch_save_to_dcp()
BroadcastingTorchSaveReader
BroadcastingTorchSaveReader.prepare_global_plan()
BroadcastingTorchSaveReader.prepare_local_plan()
BroadcastingTorchSaveReader.read_data()
BroadcastingTorchSaveReader.read_metadata()
BroadcastingTorchSaveReader.reset()
BroadcastingTorchSaveReader.set_up_storage_reader()
BroadcastingTorchSaveReader.validate_checkpoint_id()


DynamicMetaLoadPlanner
DynamicMetaLoadPlanner.set_up_planner()
Probability distributions - torch.distributions
Score function
Pathwise derivative
Distribution
Distribution
Distribution.arg_constraints
Distribution.batch_shape
Distribution.cdf()
Distribution.entropy()
Distribution.enumerate_support()
Distribution.event_shape
Distribution.expand()
Distribution.icdf()
Distribution.log_prob()
Distribution.mean
Distribution.mode
Distribution.perplexity()
Distribution.rsample()
Distribution.sample()
Distribution.sample_n()
Distribution.set_default_validate_args()
Distribution.stddev
Distribution.support
Distribution.variance




ExponentialFamily
ExponentialFamily
ExponentialFamily.entropy()




Bernoulli
Bernoulli
Bernoulli.arg_constraints
Bernoulli.entropy()
Bernoulli.enumerate_support()
Bernoulli.expand()
Bernoulli.has_enumerate_support
Bernoulli.log_prob()
Bernoulli.logits
Bernoulli.mean
Bernoulli.mode
Bernoulli.param_shape
Bernoulli.probs
Bernoulli.sample()
Bernoulli.support
Bernoulli.variance




Beta
Beta
Beta.arg_constraints
Beta.concentration0
Beta.concentration1
Beta.entropy()
Beta.expand()
Beta.has_rsample
Beta.log_prob()
Beta.mean
Beta.mode
Beta.rsample()
Beta.support
Beta.variance




Binomial
Binomial
Binomial.arg_constraints
Binomial.entropy()
Binomial.enumerate_support()
Binomial.expand()
Binomial.has_enumerate_support
Binomial.log_prob()
Binomial.logits
Binomial.mean
Binomial.mode
Binomial.param_shape
Binomial.probs
Binomial.sample()
Binomial.support
Binomial.variance




Categorical
Categorical
Categorical.arg_constraints
Categorical.entropy()
Categorical.enumerate_support()
Categorical.expand()
Categorical.has_enumerate_support
Categorical.log_prob()
Categorical.logits
Categorical.mean
Categorical.mode
Categorical.param_shape
Categorical.probs
Categorical.sample()
Categorical.support
Categorical.variance




Cauchy
Cauchy
Cauchy.arg_constraints
Cauchy.cdf()
Cauchy.entropy()
Cauchy.expand()
Cauchy.has_rsample
Cauchy.icdf()
Cauchy.log_prob()
Cauchy.mean
Cauchy.mode
Cauchy.rsample()
Cauchy.support
Cauchy.variance




Chi2
Chi2
Chi2.arg_constraints
Chi2.df
Chi2.expand()




ContinuousBernoulli
ContinuousBernoulli
ContinuousBernoulli.arg_constraints
ContinuousBernoulli.cdf()
ContinuousBernoulli.entropy()
ContinuousBernoulli.expand()
ContinuousBernoulli.has_rsample
ContinuousBernoulli.icdf()
ContinuousBernoulli.log_prob()
ContinuousBernoulli.logits
ContinuousBernoulli.mean
ContinuousBernoulli.param_shape
ContinuousBernoulli.probs
ContinuousBernoulli.rsample()
ContinuousBernoulli.sample()
ContinuousBernoulli.stddev
ContinuousBernoulli.support
ContinuousBernoulli.variance




Dirichlet
Dirichlet
Dirichlet.arg_constraints
Dirichlet.entropy()
Dirichlet.expand()
Dirichlet.has_rsample
Dirichlet.log_prob()
Dirichlet.mean
Dirichlet.mode
Dirichlet.rsample()
Dirichlet.support
Dirichlet.variance




Exponential
Exponential
Exponential.arg_constraints
Exponential.cdf()
Exponential.entropy()
Exponential.expand()
Exponential.has_rsample
Exponential.icdf()
Exponential.log_prob()
Exponential.mean
Exponential.mode
Exponential.rsample()
Exponential.stddev
Exponential.support
Exponential.variance




FisherSnedecor
FisherSnedecor
FisherSnedecor.arg_constraints
FisherSnedecor.expand()
FisherSnedecor.has_rsample
FisherSnedecor.log_prob()
FisherSnedecor.mean
FisherSnedecor.mode
FisherSnedecor.rsample()
FisherSnedecor.support
FisherSnedecor.variance




Gamma
Gamma
Gamma.arg_constraints
Gamma.cdf()
Gamma.entropy()
Gamma.expand()
Gamma.has_rsample
Gamma.log_prob()
Gamma.mean
Gamma.mode
Gamma.rsample()
Gamma.support
Gamma.variance




Geometric
Geometric
Geometric.arg_constraints
Geometric.entropy()
Geometric.expand()
Geometric.log_prob()
Geometric.logits
Geometric.mean
Geometric.mode
Geometric.probs
Geometric.sample()
Geometric.support
Geometric.variance




Gumbel
Gumbel
Gumbel.arg_constraints
Gumbel.entropy()
Gumbel.expand()
Gumbel.log_prob()
Gumbel.mean
Gumbel.mode
Gumbel.stddev
Gumbel.support
Gumbel.variance




HalfCauchy
HalfCauchy
HalfCauchy.arg_constraints
HalfCauchy.cdf()
HalfCauchy.entropy()
HalfCauchy.expand()
HalfCauchy.has_rsample
HalfCauchy.icdf()
HalfCauchy.log_prob()
HalfCauchy.mean
HalfCauchy.mode
HalfCauchy.scale
HalfCauchy.support
HalfCauchy.variance




HalfNormal
HalfNormal
HalfNormal.arg_constraints
HalfNormal.cdf()
HalfNormal.entropy()
HalfNormal.expand()
HalfNormal.has_rsample
HalfNormal.icdf()
HalfNormal.log_prob()
HalfNormal.mean
HalfNormal.mode
HalfNormal.scale
HalfNormal.support
HalfNormal.variance




Independent
Independent
Independent.arg_constraints
Independent.entropy()
Independent.enumerate_support()
Independent.expand()
Independent.has_enumerate_support
Independent.has_rsample
Independent.log_prob()
Independent.mean
Independent.mode
Independent.rsample()
Independent.sample()
Independent.support
Independent.variance




InverseGamma
InverseGamma
InverseGamma.arg_constraints
InverseGamma.concentration
InverseGamma.entropy()
InverseGamma.expand()
InverseGamma.has_rsample
InverseGamma.mean
InverseGamma.mode
InverseGamma.rate
InverseGamma.support
InverseGamma.variance




Kumaraswamy
Kumaraswamy
Kumaraswamy.arg_constraints
Kumaraswamy.entropy()
Kumaraswamy.expand()
Kumaraswamy.has_rsample
Kumaraswamy.mean
Kumaraswamy.mode
Kumaraswamy.support
Kumaraswamy.variance




LKJCholesky
LKJCholesky
LKJCholesky.arg_constraints
LKJCholesky.expand()
LKJCholesky.log_prob()
LKJCholesky.sample()
LKJCholesky.support




Laplace
Laplace
Laplace.arg_constraints
Laplace.cdf()
Laplace.entropy()
Laplace.expand()
Laplace.has_rsample
Laplace.icdf()
Laplace.log_prob()
Laplace.mean
Laplace.mode
Laplace.rsample()
Laplace.stddev
Laplace.support
Laplace.variance




LogNormal
LogNormal
LogNormal.arg_constraints
LogNormal.entropy()
LogNormal.expand()
LogNormal.has_rsample
LogNormal.loc
LogNormal.mean
LogNormal.mode
LogNormal.scale
LogNormal.support
LogNormal.variance




LowRankMultivariateNormal
LowRankMultivariateNormal
LowRankMultivariateNormal.arg_constraints
LowRankMultivariateNormal.covariance_matrix
LowRankMultivariateNormal.entropy()
LowRankMultivariateNormal.expand()
LowRankMultivariateNormal.has_rsample
LowRankMultivariateNormal.log_prob()
LowRankMultivariateNormal.mean
LowRankMultivariateNormal.mode
LowRankMultivariateNormal.precision_matrix
LowRankMultivariateNormal.rsample()
LowRankMultivariateNormal.scale_tril
LowRankMultivariateNormal.support
LowRankMultivariateNormal.variance




MixtureSameFamily
MixtureSameFamily
MixtureSameFamily.arg_constraints
MixtureSameFamily.cdf()
MixtureSameFamily.component_distribution
MixtureSameFamily.expand()
MixtureSameFamily.has_rsample
MixtureSameFamily.log_prob()
MixtureSameFamily.mean
MixtureSameFamily.mixture_distribution
MixtureSameFamily.sample()
MixtureSameFamily.support
MixtureSameFamily.variance




Multinomial
Multinomial
Multinomial.arg_constraints
Multinomial.entropy()
Multinomial.expand()
Multinomial.log_prob()
Multinomial.logits
Multinomial.mean
Multinomial.param_shape
Multinomial.probs
Multinomial.sample()
Multinomial.support
Multinomial.total_count
Multinomial.variance




MultivariateNormal
MultivariateNormal
MultivariateNormal.arg_constraints
MultivariateNormal.covariance_matrix
MultivariateNormal.entropy()
MultivariateNormal.expand()
MultivariateNormal.has_rsample
MultivariateNormal.log_prob()
MultivariateNormal.mean
MultivariateNormal.mode
MultivariateNormal.precision_matrix
MultivariateNormal.rsample()
MultivariateNormal.scale_tril
MultivariateNormal.support
MultivariateNormal.variance




NegativeBinomial
NegativeBinomial
NegativeBinomial.arg_constraints
NegativeBinomial.expand()
NegativeBinomial.log_prob()
NegativeBinomial.logits
NegativeBinomial.mean
NegativeBinomial.mode
NegativeBinomial.param_shape
NegativeBinomial.probs
NegativeBinomial.sample()
NegativeBinomial.support
NegativeBinomial.variance




Normal
Normal
Normal.arg_constraints
Normal.cdf()
Normal.entropy()
Normal.expand()
Normal.has_rsample
Normal.icdf()
Normal.log_prob()
Normal.mean
Normal.mode
Normal.rsample()
Normal.sample()
Normal.stddev
Normal.support
Normal.variance




OneHotCategorical
OneHotCategorical
OneHotCategorical.arg_constraints
OneHotCategorical.entropy()
OneHotCategorical.enumerate_support()
OneHotCategorical.expand()
OneHotCategorical.has_enumerate_support
OneHotCategorical.log_prob()
OneHotCategorical.logits
OneHotCategorical.mean
OneHotCategorical.mode
OneHotCategorical.param_shape
OneHotCategorical.probs
OneHotCategorical.sample()
OneHotCategorical.support
OneHotCategorical.variance




Pareto
Pareto
Pareto.arg_constraints
Pareto.entropy()
Pareto.expand()
Pareto.mean
Pareto.mode
Pareto.support
Pareto.variance




Poisson
Poisson
Poisson.arg_constraints
Poisson.expand()
Poisson.log_prob()
Poisson.mean
Poisson.mode
Poisson.sample()
Poisson.support
Poisson.variance




RelaxedBernoulli
RelaxedBernoulli
RelaxedBernoulli.arg_constraints
RelaxedBernoulli.expand()
RelaxedBernoulli.has_rsample
RelaxedBernoulli.logits
RelaxedBernoulli.probs
RelaxedBernoulli.support
RelaxedBernoulli.temperature




LogitRelaxedBernoulli
LogitRelaxedBernoulli
LogitRelaxedBernoulli.arg_constraints
LogitRelaxedBernoulli.expand()
LogitRelaxedBernoulli.log_prob()
LogitRelaxedBernoulli.logits
LogitRelaxedBernoulli.param_shape
LogitRelaxedBernoulli.probs
LogitRelaxedBernoulli.rsample()
LogitRelaxedBernoulli.support




RelaxedOneHotCategorical
RelaxedOneHotCategorical
RelaxedOneHotCategorical.arg_constraints
RelaxedOneHotCategorical.expand()
RelaxedOneHotCategorical.has_rsample
RelaxedOneHotCategorical.logits
RelaxedOneHotCategorical.probs
RelaxedOneHotCategorical.support
RelaxedOneHotCategorical.temperature




StudentT
StudentT
StudentT.arg_constraints
StudentT.entropy()
StudentT.expand()
StudentT.has_rsample
StudentT.log_prob()
StudentT.mean
StudentT.mode
StudentT.rsample()
StudentT.support
StudentT.variance




TransformedDistribution
TransformedDistribution
TransformedDistribution.arg_constraints
TransformedDistribution.cdf()
TransformedDistribution.expand()
TransformedDistribution.has_rsample
TransformedDistribution.icdf()
TransformedDistribution.log_prob()
TransformedDistribution.rsample()
TransformedDistribution.sample()
TransformedDistribution.support




Uniform
Uniform
Uniform.arg_constraints
Uniform.cdf()
Uniform.entropy()
Uniform.expand()
Uniform.has_rsample
Uniform.icdf()
Uniform.log_prob()
Uniform.mean
Uniform.mode
Uniform.rsample()
Uniform.stddev
Uniform.support
Uniform.variance




VonMises
VonMises
VonMises.arg_constraints
VonMises.expand()
VonMises.has_rsample
VonMises.log_prob()
VonMises.mean
VonMises.mode
VonMises.sample()
VonMises.support
VonMises.variance




Weibull
Weibull
Weibull.arg_constraints
Weibull.entropy()
Weibull.expand()
Weibull.mean
Weibull.mode
Weibull.support
Weibull.variance




Wishart
Wishart
Wishart.arg_constraints
Wishart.covariance_matrix
Wishart.entropy()
Wishart.expand()
Wishart.has_rsample
Wishart.log_prob()
Wishart.mean
Wishart.mode
Wishart.precision_matrix
Wishart.rsample()
Wishart.scale_tril
Wishart.support
Wishart.variance




KL Divergence
kl_divergence()
register_kl()


Transforms
AbsTransform
AffineTransform
CatTransform
ComposeTransform
CorrCholeskyTransform
CumulativeDistributionTransform
ExpTransform
IndependentTransform
LowerCholeskyTransform
PositiveDefiniteTransform
PowerTransform
ReshapeTransform
SigmoidTransform
SoftplusTransform
TanhTransform
SoftmaxTransform
StackTransform
StickBreakingTransform
Transform
Transform.inv
Transform.sign
Transform.log_abs_det_jacobian()
Transform.forward_shape()
Transform.inverse_shape()




Constraints
Constraint
Constraint.check()


cat
dependent_property
greater_than
greater_than_eq
independent
integer_interval
interval
half_open_interval
is_dependent()
less_than
multinomial
stack


Constraint Registry
ConstraintRegistry
ConstraintRegistry.register()
torch.compiler
Read More
torch.fft
Fast Fourier Transforms
Helper Functions
torch.func
What are composable function transforms?
Why composable function transforms?
Read More
torch.futures
Future
Future.add_done_callback()
Future.done()
Future.set_exception()
Future.set_result()
Future.then()
Future.value()
Future.wait()


collect_all()
wait_all()
torch.fx
Overview
Writing Transformations
A Quick Primer on Graphs
Graph Manipulation
Direct Graph Manipulation
Subgraph Rewriting With replace_pattern()
Graph Manipulation Examples


Proxy/Retracing
The Interpreter Pattern
Examples of the Interpreter Pattern




Debugging
Introduction
Common Pitfalls in Transform Authoring
Checking Correctness of Modules
Debugging the Generated Code
Use pdb
Print the Generated Code
Use the to_folder Function From GraphModule


Debugging the Transformation
Available Debuggers


Limitations of Symbolic Tracing
Dynamic Control Flow
Static Control Flow


Non-torch Functions
Customizing Tracing with the Tracer class
Leaf Modules


Miscellanea


API Reference
symbolic_trace()
wrap()
GraphModule
GraphModule.__init__()
GraphModule.add_submodule()
GraphModule.code
GraphModule.delete_all_unused_submodules()
GraphModule.delete_submodule()
GraphModule.graph
GraphModule.print_readable()
GraphModule.recompile()
GraphModule.to_folder()


Graph
Graph.__init__()
Graph.call_function()
Graph.call_method()
Graph.call_module()
Graph.create_node()
Graph.eliminate_dead_code()
Graph.erase_node()
Graph.find_nodes()
Graph.get_attr()
Graph.graph_copy()
Graph.inserting_after()
Graph.inserting_before()
Graph.lint()
Graph.node_copy()
Graph.nodes
Graph.on_generate_code()
Graph.output()
Graph.output_node()
Graph.placeholder()
Graph.print_tabular()
Graph.process_inputs()
Graph.process_outputs()
Graph.python_code()
Graph.set_codegen()


Node
Node.all_input_nodes
Node.append()
Node.args
Node.format_node()
Node.insert_arg()
Node.is_impure()
Node.kwargs
Node.next
Node.normalized_arguments()
Node.prepend()
Node.prev
Node.replace_all_uses_with()
Node.replace_input_with()
Node.stack_trace
Node.update_arg()
Node.update_kwarg()


Tracer
Tracer.call_module()
Tracer.create_arg()
Tracer.create_args_for_root()
Tracer.create_node()
Tracer.create_proxy()
Tracer.get_fresh_qualname()
Tracer.getattr()
Tracer.is_leaf_module()
Tracer.iter()
Tracer.keys()
Tracer.path_of_module()
Tracer.proxy()
Tracer.to_bool()
Tracer.trace()


Proxy
Interpreter
Interpreter.boxed_run()
Interpreter.call_function()
Interpreter.call_method()
Interpreter.call_module()
Interpreter.fetch_args_kwargs_from_env()
Interpreter.fetch_attr()
Interpreter.get_attr()
Interpreter.map_nodes_to_values()
Interpreter.output()
Interpreter.placeholder()
Interpreter.run()
Interpreter.run_node()


Transformer
Transformer.call_function()
Transformer.call_module()
Transformer.get_attr()
Transformer.placeholder()
Transformer.transform()


replace_pattern()
torch.fx.experimental
torch.fx.experimental.symbolic_shapes
torch.fx.experimental.proxy_tensor
torch.hub
Publishing models
How to implement an entrypoint?
Important Notice


Loading models from Hub
list()
help()
load()
download_url_to_file()
load_state_dict_from_url()
Running a loaded model:
Where are my downloaded models saved?
get_dir()
set_dir()


Caching logic
Known limitations:
TorchScript
Creating TorchScript Code
Mixing Tracing and Scripting
TorchScript Language
Built-in Functions and Modules
PyTorch Functions and Modules
Python Functions and Modules
Python Language Reference Comparison


Debugging
Disable JIT for Debugging
Inspecting Code
Interpreting Graphs
Tracer
Tracing Edge Cases
Automatic Trace Checking
Tracer Warnings




Frequently Asked Questions
Known Issues
Appendix
Migrating to PyTorch 1.2 Recursive Scripting API
Modules
export()


Functions
TorchScript Classes
Attributes
Constants
Variables


Fusion Backends
References
torch.linalg
Matrix Properties
Decompositions
Solvers
Inverses
Matrix Functions
Matrix Products
Tensor Operations
Misc
Experimental Functions
torch.monitor
API Reference
Aggregation
Aggregation.name


Stat
Stat.__init__()
Stat.add()
Stat.count
Stat.get()
Stat.name


data_value_t
Event
Event.__init__()
Event.data
Event.name
Event.timestamp


EventHandlerHandle
log_event()
register_event_handler()
unregister_event_handler()
TensorboardEventHandler
TensorboardEventHandler.__init__()
torch.signal
torch.signal.windows
torch.special
Functions
airy_ai()
bessel_j0()
bessel_j1()
digamma()
entr()
erf()
erfc()
erfcx()
erfinv()
exp2()
expit()
expm1()
gammainc()
gammaincc()
gammaln()
i0()
i0e()
i1()
i1e()
log1p()
log_ndtr()
log_softmax()
logit()
logsumexp()
multigammaln()
ndtr()
ndtri()
polygamma()
psi()
round()
scaled_modified_bessel_k0()
scaled_modified_bessel_k1()
sinc()
softmax()
spherical_bessel_j0()
xlog1py()
xlogy()
zeta()
torch.overrides
Functions
get_ignored_functions()
get_overridable_functions()
resolve_name()
get_testing_overrides()
handle_torch_function()
has_torch_function()
is_tensor_like()
is_tensor_method_or_property()
wrap_torch_function()
torch.package
Tutorials
Packaging your first model


How do I…
See what is inside a package?
Treat the package like a ZIP archive
Use the file_structure() API


See why a given module was included as a dependency?
Include arbitrary resources with my package and access them later?
Customize how a class is packaged?
Test in my source code whether or not it is executing inside a package?
Patch code into a package?
Access package contents from packaged code?
Distinguish between packaged code and non-packaged code?
Re-export an imported object?
Package a TorchScript module?


Explanation
torch.package Format Overview
Framework files
User files


How torch.package finds your code’s dependencies
Analyzing an object’s dependencies
Analyzing a module’s dependencies


Dependency Management
intern
extern
mock
Refactoring
Patterns


torch.package sharp edges
Avoid global state in your modules
Types are not shared between packages and the loading environment


How torch.package keeps packages isolated from each other
Mangling




API Reference
PackagingError
EmptyMatchError
PackageExporter
PackageExporter.__init__()
PackageExporter.add_dependency()
PackageExporter.all_paths()
PackageExporter.close()
PackageExporter.denied_modules()
PackageExporter.deny()
PackageExporter.dependency_graph_string()
PackageExporter.extern()
PackageExporter.externed_modules()
PackageExporter.get_rdeps()
PackageExporter.get_unique_id()
PackageExporter.intern()
PackageExporter.interned_modules()
PackageExporter.mock()
PackageExporter.mocked_modules()
PackageExporter.register_extern_hook()
PackageExporter.register_intern_hook()
PackageExporter.register_mock_hook()
PackageExporter.save_binary()
PackageExporter.save_module()
PackageExporter.save_pickle()
PackageExporter.save_source_file()
PackageExporter.save_source_string()
PackageExporter.save_text()


PackageImporter
PackageImporter.__init__()
PackageImporter.file_structure()
PackageImporter.id()
PackageImporter.import_module()
PackageImporter.load_binary()
PackageImporter.load_pickle()
PackageImporter.load_text()
PackageImporter.python_version()


Directory
Directory.has_file()
torch.profiler
Overview
API Reference
_KinetoProfile
_KinetoProfile.add_metadata()
_KinetoProfile.add_metadata_json()
_KinetoProfile.events()
_KinetoProfile.export_chrome_trace()
_KinetoProfile.export_memory_timeline()
_KinetoProfile.export_stacks()
_KinetoProfile.key_averages()
_KinetoProfile.preset_metadata_json()
_KinetoProfile.toggle_collection_dynamic()


profile
profile.get_trace_id()
profile.set_custom_trace_id_callback()
profile.step()


ProfilerAction
ProfilerActivity
ProfilerActivity.name


schedule()
tensorboard_trace_handler()


Intel Instrumentation and Tracing Technology APIs
is_available()
mark()
range_push()
range_pop()
torch.nn.init
calculate_gain()
uniform_()
normal_()
constant_()
ones_()
zeros_()
eye_()
dirac_()
xavier_uniform_()
xavier_normal_()
kaiming_uniform_()
kaiming_normal_()
trunc_normal_()
orthogonal_()
sparse_()
torch.nn.attention
Utils
Submodules
torch.onnx
Overview
TorchDynamo-based ONNX Exporter
TorchScript-based ONNX Exporter
Contributing / Developing
torch.optim
How to use an optimizer
Constructing it
Per-parameter options
Taking an optimization step
optimizer.step()
optimizer.step(closure)




Base class
Optimizer


Algorithms
How to adjust learning rate
How to utilize named parameters to load optimizer state dict
Weight Averaging (SWA and EMA)
Constructing averaged models
Custom averaging strategies
SWA learning rate schedules
Taking care of batch normalization
Putting it all together: SWA
Putting it all together: EMA
get_ema_multi_avg_fn()
update_bn()
Complex Numbers
Creating Complex Tensors
Transition from the old representation
Accessing real and imag
Angle and abs
Linear Algebra
Serialization
Autograd
Optimizers
DDP Communication Hooks
How to Use a Communication Hook?
What Does a Communication Hook Operate On?
GradBucket
index()
buffer()
gradients()
is_last()
set_buffer()
parameters()


Default Communication Hooks
allreduce_hook()
fp16_compress_hook()
bf16_compress_hook()
fp16_compress_wrapper()
bf16_compress_wrapper()


PowerSGD Communication Hook
PowerSGD State
PowerSGDState


PowerSGD Hooks
powerSGD_hook()
batched_powerSGD_hook()




Debugging Communication Hooks
noop_hook()


Checkpointing of Communication Hooks
PowerSGDState.__getstate__()
PowerSGDState.__setstate__()


Acknowledgements
Quantization
Introduction to Quantization
Quantization API Summary
Eager Mode Quantization
Post Training Dynamic Quantization
Post Training Static Quantization
Quantization Aware Training for Static Quantization
Model Preparation for Eager Mode Static Quantization


(Prototype - maintenance mode) FX Graph Mode Quantization
(Prototype) PyTorch 2 Export Quantization


Quantization Stack
Quantized Model
Quantized Tensor
Quantize and Dequantize
Quantized Operators/Modules
Quantized Engine


Quantization Flow
Observer and FakeQuantize
QConfig
General Quantization Flow




Quantization Support Matrix
Quantization Mode Support
Quantization Flow Support
Backend/Hardware Support
Note for native CPU backends


Operator Support


Quantization API Reference
Quantization Backend Configuration
Quantization Accuracy Debugging
Quantization Customizations
Quantization Custom Module API


Best Practices
Frequently Asked Questions
Common Errors
Passing a non-quantized Tensor into a quantized kernel
Passing a quantized Tensor into a non-quantized kernel
Saving and Loading Quantized models
Symbolic Trace Error when using FX Graph Mode Quantization
Distributed RPC Framework
Basics
RPC
init_rpc()
rpc_sync()
rpc_async()
remote()
get_worker_info()
shutdown()
WorkerInfo
WorkerInfo.id
WorkerInfo.name


async_execution()
Backends
BackendType
RpcBackendOptions
RpcBackendOptions.init_method
RpcBackendOptions.rpc_timeout


TensorPipe Backend
TensorPipeRpcBackendOptions
TensorPipeRpcBackendOptions.device_maps
TensorPipeRpcBackendOptions.devices
TensorPipeRpcBackendOptions.init_method
TensorPipeRpcBackendOptions.num_worker_threads
TensorPipeRpcBackendOptions.rpc_timeout
TensorPipeRpcBackendOptions.set_device_map()
TensorPipeRpcBackendOptions.set_devices()








RRef
PyRRef
PyRRef.backward()
PyRRef.confirmed_by_owner()
PyRRef.is_owner()
PyRRef.local_value()
PyRRef.owner()
PyRRef.owner_name()
PyRRef.remote()
PyRRef.rpc_async()
PyRRef.rpc_sync()
PyRRef.to_here()




RemoteModule
RemoteModule
RemoteModule.get_module_rref()
RemoteModule.remote_parameters()




Distributed Autograd Framework
backward()
context
get_gradients()


Distributed Optimizer
Design Notes
Tutorials
torch.random
fork_rng()
get_rng_state()
initial_seed()
manual_seed()
seed()
set_rng_state()
torch.masked
Introduction
Motivation
What is a MaskedTensor?


Supported Operators
Unary Operators


Binary Operators


Reductions
View and select functions
torch.nested
Introduction
Construction
Data Layout and Shape
Supported Operations
Viewing nested tensor constituents
Conversions to / from padded
Shape manipulations
Attention mechanisms


Usage with torch.compile
Troubleshooting
Unimplemented ops
Ragged structure incompatibility
Data dependent operation within torch.compile


Contributions
Detailed Docs for Construction and Conversion Functions
nested_tensor()
nested_tensor_from_jagged()
as_nested_tensor()
to_padded_tensor()
masked_select()
narrow()
torch.Size
Size
Size.count()
Size.index()
Size.numel()
torch.sparse
Why and when to use sparsity
Functionality overview
Operator overview
Sparse Semi-Structured Tensors
Constructing Sparse Semi-Structured Tensors
Sparse Semi-Structured Tensor Operations
Accelerating nn.Linear with semi-structured sparsity


Sparse COO tensors
Construction
Sparse hybrid COO tensors
Uncoalesced sparse COO tensors
Working with sparse COO tensors


Sparse Compressed Tensors
Sparse CSR Tensor
Construction of CSR tensors
CSR Tensor Operations


Sparse CSC Tensor
Construction of CSC tensors


Sparse BSR Tensor
Construction of BSR tensors


Sparse BSC Tensor
Construction of BSC tensors


Tools for working with sparse compressed tensors
Construction of sparse compressed tensors




Supported operations
Linear Algebra operations
Tensor methods and sparse


Torch functions specific to sparse Tensors
Other functions


Zero-preserving unary functions
torch.Storage
Untyped Storage API
Special cases
UntypedStorage
UntypedStorage.bfloat16()
UntypedStorage.bool()
UntypedStorage.byte()
UntypedStorage.byteswap()
UntypedStorage.char()
UntypedStorage.clone()
UntypedStorage.complex_double()
UntypedStorage.complex_float()
UntypedStorage.copy_()
UntypedStorage.cpu()
UntypedStorage.cuda()
UntypedStorage.data_ptr()
UntypedStorage.device
UntypedStorage.double()
UntypedStorage.element_size()
UntypedStorage.filename
UntypedStorage.fill_()
UntypedStorage.float()
UntypedStorage.float8_e4m3fn()
UntypedStorage.float8_e4m3fnuz()
UntypedStorage.float8_e5m2()
UntypedStorage.float8_e5m2fnuz()
UntypedStorage.from_buffer()
UntypedStorage.from_file()
UntypedStorage.get_device()
UntypedStorage.half()
UntypedStorage.hpu()
UntypedStorage.int()
UntypedStorage.is_cuda
UntypedStorage.is_hpu
UntypedStorage.is_pinned()
UntypedStorage.is_shared()
UntypedStorage.is_sparse
UntypedStorage.is_sparse_csr
UntypedStorage.long()
UntypedStorage.mps()
UntypedStorage.nbytes()
UntypedStorage.new()
UntypedStorage.pin_memory()
UntypedStorage.resizable()
UntypedStorage.resize_()
UntypedStorage.share_memory_()
UntypedStorage.short()
UntypedStorage.size()
UntypedStorage.to()
UntypedStorage.tolist()
UntypedStorage.type()
UntypedStorage.untyped()




Legacy Typed Storage
TypedStorage
TypedStorage.bfloat16()
TypedStorage.bool()
TypedStorage.byte()
TypedStorage.char()
TypedStorage.clone()
TypedStorage.complex_double()
TypedStorage.complex_float()
TypedStorage.copy_()
TypedStorage.cpu()
TypedStorage.cuda()
TypedStorage.data_ptr()
TypedStorage.device
TypedStorage.double()
TypedStorage.dtype
TypedStorage.element_size()
TypedStorage.filename
TypedStorage.fill_()
TypedStorage.float()
TypedStorage.float8_e4m3fn()
TypedStorage.float8_e4m3fnuz()
TypedStorage.float8_e5m2()
TypedStorage.float8_e5m2fnuz()
TypedStorage.from_buffer()
TypedStorage.from_file()
TypedStorage.get_device()
TypedStorage.half()
TypedStorage.hpu()
TypedStorage.int()
TypedStorage.is_cuda
TypedStorage.is_hpu
TypedStorage.is_pinned()
TypedStorage.is_shared()
TypedStorage.is_sparse
TypedStorage.long()
TypedStorage.nbytes()
TypedStorage.pickle_storage_type()
TypedStorage.pin_memory()
TypedStorage.resizable()
TypedStorage.resize_()
TypedStorage.share_memory_()
TypedStorage.short()
TypedStorage.size()
TypedStorage.to()
TypedStorage.tolist()
TypedStorage.type()
TypedStorage.untyped()


DoubleStorage
DoubleStorage.dtype


FloatStorage
FloatStorage.dtype


HalfStorage
HalfStorage.dtype


LongStorage
LongStorage.dtype


IntStorage
IntStorage.dtype


ShortStorage
ShortStorage.dtype


CharStorage
CharStorage.dtype


ByteStorage
ByteStorage.dtype


BoolStorage
BoolStorage.dtype


BFloat16Storage
BFloat16Storage.dtype


ComplexDoubleStorage
ComplexDoubleStorage.dtype


ComplexFloatStorage
ComplexFloatStorage.dtype


QUInt8Storage
QUInt8Storage.dtype


QInt8Storage
QInt8Storage.dtype


QInt32Storage
QInt32Storage.dtype


QUInt4x2Storage
QUInt4x2Storage.dtype


QUInt2x4Storage
QUInt2x4Storage.dtype
torch.testing
assert_close()
make_tensor()
assert_allclose()
torch.utils
Benchmark Utils - torch.utils.benchmark
Timer
Timer.adaptive_autorange()
Timer.blocked_autorange()
Timer.collect_callgrind()
Timer.timeit()


Measurement
Measurement.merge()
Measurement.significant_figures


CallgrindStats
CallgrindStats.as_standardized()
CallgrindStats.counts()
CallgrindStats.delta()
CallgrindStats.stats()


FunctionCounts
FunctionCounts.denoise()
FunctionCounts.filter()
FunctionCounts.transform()


Compare
Compare.colorize()
Compare.extend_results()
Compare.highlight_warnings()
Compare.print()
Compare.trim_significant_figures()
torch.utils.bottleneck
torch.utils.checkpoint
checkpoint()
checkpoint_sequential()
set_checkpoint_debug_enabled()
CheckpointPolicy
SelectiveCheckpointContext
create_selective_checkpoint_contexts()
torch.utils.cpp_extension
CppExtension()
CUDAExtension()
SyclExtension()
BuildExtension()
load()
load_inline()
include_paths()
get_compiler_abi_compatibility_and_version()
verify_ninja_availability()
is_ninja_available()
torch.utils.data
Dataset Types
Map-style datasets
Iterable-style datasets


Data Loading Order and Sampler
Loading Batched and Non-Batched Data
Automatic batching (default)
Disable automatic batching
Working with collate_fn


Single- and Multi-process Data Loading
Single-process data loading (default)
Multi-process data loading
Platform-specific behaviors
Randomness in multi-process data loading




Memory Pinning
DataLoader
Dataset
IterableDataset
TensorDataset
StackDataset
ConcatDataset
ChainDataset
Subset
collate()
default_collate()
default_convert()
get_worker_info()
random_split()
Sampler
SequentialSampler
RandomSampler
SubsetRandomSampler
WeightedRandomSampler
BatchSampler
DistributedSampler
torch.utils.deterministic
fill_uninitialized_memory
JIT Utils - torch.utils.jit
torch.utils.dlpack
from_dlpack()
to_dlpack()
torch.utils.model_zoo
load_url()
torch.utils.tensorboard
SummaryWriter
SummaryWriter.__init__()
SummaryWriter.add_scalar()
SummaryWriter.add_scalars()
SummaryWriter.add_histogram()
SummaryWriter.add_image()
SummaryWriter.add_images()
SummaryWriter.add_figure()
SummaryWriter.add_video()
SummaryWriter.add_audio()
SummaryWriter.add_text()
SummaryWriter.add_graph()
SummaryWriter.add_embedding()
SummaryWriter.add_pr_curve()
SummaryWriter.add_custom_scalars()
SummaryWriter.add_mesh()
SummaryWriter.add_hparams()
SummaryWriter.flush()
SummaryWriter.close()
torch.utils.module_tracker
ModuleTracker
Type Info
torch.finfo
torch.finfo


torch.iinfo
torch.iinfo
Named Tensors
Creating named tensors
Named dimensions
Name propagation semantics
match semantics
Basic name inference rules


Explicit alignment by names
Manipulating dimensions
Autograd support
Currently supported operations and subsystems
Operators
Subsystems


Named tensor API reference
Tensor.names
Tensor.rename()
Tensor.rename_()
Tensor.refine_names()
Tensor.align_as()
Tensor.align_to()
Named Tensors operator coverage
Keeps input names
Removes dimensions
Unifies names from inputs
Permutes dimensions
Contracts away dims
Factory functions
out function and in-place variants
torch.__config__
show()
parallel_info()
torch.__future__
set_overwrite_module_params_on_conversion()
get_overwrite_module_params_on_conversion()
set_swap_module_params_on_conversion()
get_swap_module_params_on_conversion()
torch._logging
Torch Environment Variables
PyTorch documentation


Indices and tables
torch.is_tensor
is_tensor()
torch.is_storage
is_storage()
torch.is_complex
is_complex()
torch.is_conj
is_conj()
torch.is_floating_point
is_floating_point()
torch.is_nonzero
is_nonzero()
torch.set_default_dtype
set_default_dtype()
torch.get_default_dtype
get_default_dtype()
torch.set_default_device
set_default_device()
torch.get_default_device
get_default_device()
torch.set_default_tensor_type
set_default_tensor_type()
torch.numel
numel()
torch.set_printoptions
set_printoptions()
torch.set_flush_denormal
set_flush_denormal()
torch.rand
rand()
torch.rand_like
rand_like()
torch.randn
randn()
torch.randn_like
randn_like()
torch.randint
randint()
torch.randint_like
randint_like()
torch.randperm
randperm()
torch.empty
empty()
torch.tensor
tensor()
torch.sparse_coo_tensor
sparse_coo_tensor()
torch.sparse_csr_tensor
sparse_csr_tensor()
torch.sparse_csc_tensor
sparse_csc_tensor()
torch.sparse_bsr_tensor
sparse_bsr_tensor()
torch.sparse_bsc_tensor
sparse_bsc_tensor()
torch.asarray
asarray()
torch.as_tensor
as_tensor()
torch.as_strided
as_strided()
torch.from_file
from_file()
torch.from_numpy
from_numpy()
torch.from_dlpack
from_dlpack()
torch.frombuffer
frombuffer()
torch.zeros
zeros()
torch.zeros_like
zeros_like()
torch.ones
ones()
torch.ones_like
ones_like()
torch.arange
arange()
torch.range
range()
torch.linspace
linspace()
torch.logspace
logspace()
torch.eye
eye()
torch.empty_like
empty_like()
torch.empty_strided
empty_strided()
torch.full
full()
torch.full_like
full_like()
torch.quantize_per_tensor
quantize_per_tensor()
torch.quantize_per_channel
quantize_per_channel()
torch.dequantize
dequantize()
torch.complex
complex()
torch.real
real()
torch.imag
imag()
torch.polar
polar()
torch.abs
abs()
torch.angle
angle()
torch.heaviside
heaviside()
torch.adjoint
adjoint()
torch.argwhere
argwhere()
torch.cat
cat()
torch.concat
concat()
torch.concatenate
concatenate()
torch.conj
conj()
torch.chunk
chunk()
torch.dsplit
dsplit()
torch.column_stack
column_stack()
torch.dstack
dstack()
torch.gather
gather()
torch.hsplit
hsplit()
torch.hstack
hstack()
torch.index_add
index_add()
torch.Tensor.index_add_
Tensor.index_add_()
torch.index_copy
index_copy()
torch.index_reduce
index_reduce()
torch.Tensor.index_reduce_
Tensor.index_reduce_()
torch.index_select
index_select()
torch.masked_select
masked_select()
torch.movedim
movedim()
torch.moveaxis
moveaxis()
torch.narrow
narrow()
torch.narrow_copy
narrow_copy()
torch.Tensor.narrow
Tensor.narrow()
torch.nonzero
nonzero()
torch.permute
permute()
torch.reshape
reshape()
torch.row_stack
row_stack()
torch.vstack
vstack()
torch.select
select()
torch.scatter
scatter()
torch.Tensor.scatter_
Tensor.scatter_()
torch.diagonal_scatter
diagonal_scatter()
torch.select_scatter
select_scatter()
torch.slice_scatter
slice_scatter()
torch.scatter_add
scatter_add()
torch.Tensor.scatter_add_
Tensor.scatter_add_()
torch.scatter_reduce
scatter_reduce()
torch.Tensor.scatter_reduce_
Tensor.scatter_reduce_()
torch.split
split()
torch.squeeze
squeeze()
torch.stack
stack()
torch.swapaxes
swapaxes()
torch.transpose
transpose()
torch.swapdims
swapdims()
torch.t
t()
torch.take
take()
torch.take_along_dim
take_along_dim()
torch.tensor_split
tensor_split()
torch.tile
tile()
torch.unbind
unbind()
torch.unravel_index
unravel_index()
torch.unsqueeze
unsqueeze()
torch.vsplit
vsplit()
torch.where
where()
Stream
Stream
Stream.query()
Stream.record_event()
Stream.synchronize()
Stream.wait_event()
Stream.wait_stream()
Event
Event
Event.elapsed_time()
Event.query()
Event.record()
Event.synchronize()
Event.wait()
torch.accelerator.current_accelerator
current_accelerator()
torch.accelerator.is_available
is_available()
Generator
Generator
Generator.clone_state()
Generator.device
Generator.get_state()
Generator.graphsafe_get_state()
Generator.graphsafe_set_state()
Generator.initial_seed()
Generator.manual_seed()
Generator.seed()
Generator.set_state()
torch.seed
seed()
torch.manual_seed
manual_seed()
torch.initial_seed
initial_seed()
torch.get_rng_state
get_rng_state()
torch.set_rng_state
set_rng_state()
torch.bernoulli
bernoulli()
torch.multinomial
multinomial()
torch.normal
normal()
torch.poisson
poisson()
torch.Tensor.bernoulli_
Tensor.bernoulli_()
torch.Tensor.cauchy_
Tensor.cauchy_()
torch.Tensor.exponential_
Tensor.exponential_()
torch.Tensor.geometric_
Tensor.geometric_()
torch.Tensor.log_normal_
Tensor.log_normal_()
torch.Tensor.normal_
Tensor.normal_()
torch.Tensor.random_
Tensor.random_()
torch.Tensor.uniform_
Tensor.uniform_()
SobolEngine
SobolEngine
SobolEngine.draw()
SobolEngine.draw_base2()
SobolEngine.fast_forward()
SobolEngine.reset()
torch.save
save()
torch.load
load()
torch.get_num_threads
get_num_threads()
torch.set_num_threads
set_num_threads()
torch.get_num_interop_threads
get_num_interop_threads()
torch.set_num_interop_threads
set_num_interop_threads()
no_grad
no_grad
enable_grad
enable_grad
set_grad_enabled
set_grad_enabled
set_grad_enabled.clone()
torch.is_grad_enabled
is_grad_enabled()
inference_mode
inference_mode
inference_mode.clone()
torch.is_inference_mode_enabled
is_inference_mode_enabled()
torch.absolute
absolute()
torch.acos
acos()
torch.arccos
arccos()
torch.acosh
acosh()
torch.arccosh
arccosh()
torch.add
add()
torch.addcdiv
addcdiv()
torch.addcmul
addcmul()
torch.asin
asin()
torch.arcsin
arcsin()
torch.asinh
asinh()
torch.arcsinh
arcsinh()
torch.atan
atan()
torch.arctan
arctan()
torch.atanh
atanh()
torch.arctanh
arctanh()
torch.atan2
atan2()
torch.arctan2
arctan2()
torch.bitwise_not
bitwise_not()
torch.bitwise_and
bitwise_and()
torch.bitwise_or
bitwise_or()
torch.bitwise_xor
bitwise_xor()
torch.bitwise_left_shift
bitwise_left_shift()
torch.bitwise_right_shift
bitwise_right_shift()
torch.ceil
ceil()
torch.clamp
clamp()
torch.min
min()
torch.max
max()
torch.clip
clip()
torch.conj_physical
conj_physical()
torch.copysign
copysign()
torch.cos
cos()
torch.cosh
cosh()
torch.deg2rad
deg2rad()
torch.div
div()
torch.divide
divide()
torch.digamma
digamma()
torch.erf
erf()
torch.erfc
erfc()
torch.erfinv
erfinv()
torch.exp
exp()
torch.exp2
exp2()
torch.expm1
expm1()
torch.fake_quantize_per_channel_affine
fake_quantize_per_channel_affine()
torch.fake_quantize_per_tensor_affine
fake_quantize_per_tensor_affine()
torch.fix
fix()
torch.trunc
trunc()
torch.float_power
float_power()
torch.floor
floor()
torch.floor_divide
floor_divide()
torch.fmod
fmod()
torch.frac
frac()
torch.frexp
frexp()
torch.gradient
gradient()
torch.ldexp
ldexp()
torch.lerp
lerp()
torch.lgamma
lgamma()
torch.log
log()
torch.log10
log10()
torch.log1p
log1p()
torch.log2
log2()
torch.logaddexp
logaddexp()
torch.logaddexp2
logaddexp2()
torch.logical_and
logical_and()
torch.logical_not
logical_not()
torch.logical_or
logical_or()
torch.logical_xor
logical_xor()
torch.logit
logit()
torch.hypot
hypot()
torch.i0
i0()
torch.igamma
igamma()
torch.igammac
igammac()
torch.mul
mul()
torch.multiply
multiply()
torch.mvlgamma
mvlgamma()
torch.nan_to_num
nan_to_num()
torch.neg
neg()
torch.negative
negative()
torch.nextafter
nextafter()
torch.polygamma
polygamma()
torch.positive
positive()
torch.pow
pow()
torch.quantized_batch_norm
quantized_batch_norm()
torch.quantized_max_pool1d
quantized_max_pool1d()
torch.quantized_max_pool2d
quantized_max_pool2d()
torch.rad2deg
rad2deg()
torch.reciprocal
reciprocal()
torch.remainder
remainder()
torch.round
round()
torch.rsqrt
rsqrt()
torch.sigmoid
sigmoid()
torch.sign
sign()
torch.sgn
sgn()
torch.signbit
signbit()
torch.sin
sin()
torch.sinc
sinc()
torch.sinh
sinh()
torch.softmax
softmax()
torch.nn.functional.softmax
softmax()
torch.sqrt
sqrt()
torch.square
square()
torch.sub
sub()
torch.subtract
subtract()
torch.tan
tan()
torch.tanh
tanh()
torch.true_divide
true_divide()
torch.xlogy
xlogy()
torch.argmax
argmax()
torch.argmin
argmin()
torch.amax
amax()
torch.amin
amin()
torch.aminmax
aminmax()
torch.all
all()
torch.any
any()
torch.dist
dist()
torch.logsumexp
logsumexp()
torch.mean
mean()
torch.nanmean
nanmean()
torch.median
median()
torch.nanmedian
nanmedian()
torch.mode
mode()
torch.norm
norm()
torch.nansum
nansum()
torch.prod
prod()
torch.quantile
quantile()
torch.nanquantile
nanquantile()
torch.std
std()
torch.std_mean
std_mean()
torch.sum
sum()
torch.unique
unique()
torch.unique_consecutive
unique_consecutive()
torch.var
var()
torch.var_mean
var_mean()
torch.count_nonzero
count_nonzero()
torch.allclose
allclose()
torch.argsort
argsort()
torch.eq
eq()
torch.equal
equal()
torch.ge
ge()
torch.greater_equal
greater_equal()
torch.gt
gt()
torch.greater
greater()
torch.isclose
isclose()
torch.isfinite
isfinite()
torch.isin
isin()
torch.isinf
isinf()
torch.isposinf
isposinf()
torch.isneginf
isneginf()
torch.isnan
isnan()
torch.isreal
isreal()
torch.kthvalue
kthvalue()
torch.le
le()
torch.less_equal
less_equal()
torch.lt
lt()
torch.less
less()
torch.maximum
maximum()
torch.minimum
minimum()
torch.fmax
fmax()
torch.fmin
fmin()
torch.ne
ne()
torch.not_equal
not_equal()
torch.sort
sort()
torch.topk
topk()
torch.msort
msort()
torch.stft
stft()
torch.istft
istft()
torch.bartlett_window
bartlett_window()
torch.blackman_window
blackman_window()
torch.hamming_window
hamming_window()
torch.hann_window
hann_window()
torch.kaiser_window
kaiser_window()
torch.atleast_1d
atleast_1d()
torch.atleast_2d
atleast_2d()
torch.atleast_3d
atleast_3d()
torch.bincount
bincount()
torch.block_diag
block_diag()
torch.broadcast_tensors
broadcast_tensors()
torch.broadcast_to
broadcast_to()
torch.broadcast_shapes
broadcast_shapes()
torch.bucketize
bucketize()
torch.cartesian_prod
cartesian_prod()
torch.cdist
cdist()
torch.clone
clone()
torch.combinations
combinations()
torch.corrcoef
corrcoef()
torch.cov
cov()
torch.cross
cross()
torch.cummax
cummax()
torch.cummin
cummin()
torch.cumprod
cumprod()
torch.cumsum
cumsum()
torch.diag
diag()
torch.diag_embed
diag_embed()
torch.diagflat
diagflat()
torch.diagonal
diagonal()
torch.diff
diff()
torch.einsum
einsum()
torch.flatten
flatten()
torch.flip
flip()
torch.fliplr
fliplr()
torch.flipud
flipud()
torch.kron
kron()
torch.rot90
rot90()
torch.gcd
gcd()
torch.histc
histc()
torch.histogram
histogram()
torch.histogramdd
histogramdd()
torch.meshgrid
meshgrid()
torch.lcm
lcm()
torch.logcumsumexp
logcumsumexp()
torch.ravel
ravel()
torch.renorm
renorm()
torch.repeat_interleave
repeat_interleave()
torch.roll
roll()
torch.searchsorted
searchsorted()
torch.tensordot
tensordot()
torch.trace
trace()
torch.tril
tril()
torch.tril_indices
tril_indices()
torch.triu
triu()
torch.triu_indices
triu_indices()
torch.unflatten
unflatten()
torch.vander
vander()
torch.view_as_real
view_as_real()
torch.view_as_complex
view_as_complex()
torch.resolve_conj
resolve_conj()
torch.resolve_neg
resolve_neg()
torch.addbmm
addbmm()
torch.addmm
addmm()
torch.addmv
addmv()
torch.addr
addr()
torch.baddbmm
baddbmm()
torch.bmm
bmm()
torch.chain_matmul
chain_matmul()
torch.cholesky
cholesky()
torch.cholesky_inverse
cholesky_inverse()
torch.cholesky_solve
cholesky_solve()
torch.dot
dot()
torch.geqrf
geqrf()
torch.ger
ger()
torch.outer
outer()
torch.inner
inner()
torch.inverse
inverse()
torch.linalg.inv
inv()
torch.det
det()
torch.linalg.det
det()
torch.logdet
logdet()
torch.slogdet
slogdet()
torch.linalg.slogdet
slogdet()
torch.lu
lu()
torch.lu_solve
lu_solve()
torch.linalg.lu_factor
lu_factor()
torch.lu_unpack
lu_unpack()
torch.matmul
matmul()
torch.matrix_power
matrix_power()
torch.linalg.matrix_power
matrix_power()
torch.matrix_exp
matrix_exp()
torch.linalg.matrix_exp
matrix_exp()
torch.mm
mm()
torch.mv
mv()
torch.orgqr
orgqr()
torch.linalg.householder_product
householder_product()
torch.ormqr
ormqr()
torch.pinverse
pinverse()
torch.linalg.pinv
pinv()
torch.qr
qr()
torch.svd
svd()
torch.svd_lowrank
svd_lowrank()
torch.pca_lowrank
pca_lowrank()
torch.lobpcg
lobpcg()
torch.trapz
trapz()
torch.trapezoid
trapezoid()
torch.cumulative_trapezoid
cumulative_trapezoid()
torch.triangular_solve
triangular_solve()
torch.vdot
vdot()
torch._foreach_abs
_foreach_abs()
torch._foreach_abs_
_foreach_abs_()
torch._foreach_acos
_foreach_acos()
torch._foreach_acos_
_foreach_acos_()
torch._foreach_asin
_foreach_asin()
torch._foreach_asin_
_foreach_asin_()
torch._foreach_atan
_foreach_atan()
torch._foreach_atan_
_foreach_atan_()
torch._foreach_ceil
_foreach_ceil()
torch._foreach_ceil_
_foreach_ceil_()
torch._foreach_cos
_foreach_cos()
torch._foreach_cos_
_foreach_cos_()
torch._foreach_cosh
_foreach_cosh()
torch._foreach_cosh_
_foreach_cosh_()
torch._foreach_erf
_foreach_erf()
torch._foreach_erf_
_foreach_erf_()
torch._foreach_erfc
_foreach_erfc()
torch._foreach_erfc_
_foreach_erfc_()
torch._foreach_exp
_foreach_exp()
torch._foreach_exp_
_foreach_exp_()
torch._foreach_expm1
_foreach_expm1()
torch._foreach_expm1_
_foreach_expm1_()
torch._foreach_floor
_foreach_floor()
torch._foreach_floor_
_foreach_floor_()
torch._foreach_log
_foreach_log()
torch._foreach_log_
_foreach_log_()
torch._foreach_log10
_foreach_log10()
torch._foreach_log10_
_foreach_log10_()
torch._foreach_log1p
_foreach_log1p()
torch._foreach_log1p_
_foreach_log1p_()
torch._foreach_log2
_foreach_log2()
torch._foreach_log2_
_foreach_log2_()
torch._foreach_neg
_foreach_neg()
torch._foreach_neg_
_foreach_neg_()
torch._foreach_tan
_foreach_tan()
torch._foreach_tan_
_foreach_tan_()
torch._foreach_sin
_foreach_sin()
torch._foreach_sin_
_foreach_sin_()
torch._foreach_sinh
_foreach_sinh()
torch._foreach_sinh_
_foreach_sinh_()
torch._foreach_round
_foreach_round()
torch._foreach_round_
_foreach_round_()
torch._foreach_sqrt
_foreach_sqrt()
torch._foreach_sqrt_
_foreach_sqrt_()
torch._foreach_lgamma
_foreach_lgamma()
torch._foreach_lgamma_
_foreach_lgamma_()
torch._foreach_frac
_foreach_frac()
torch._foreach_frac_
_foreach_frac_()
torch._foreach_reciprocal
_foreach_reciprocal()
torch._foreach_reciprocal_
_foreach_reciprocal_()
torch._foreach_sigmoid
_foreach_sigmoid()
torch._foreach_sigmoid_
_foreach_sigmoid_()
torch._foreach_trunc
_foreach_trunc()
torch._foreach_trunc_
_foreach_trunc_()
torch._foreach_zero_
_foreach_zero_()
torch.compiled_with_cxx11_abi
compiled_with_cxx11_abi()
torch.result_type
result_type()
torch.can_cast
can_cast()
torch.promote_types
promote_types()
torch.use_deterministic_algorithms
use_deterministic_algorithms()
torch.are_deterministic_algorithms_enabled
are_deterministic_algorithms_enabled()
torch.is_deterministic_algorithms_warn_only_enabled
is_deterministic_algorithms_warn_only_enabled()
torch.set_deterministic_debug_mode
set_deterministic_debug_mode()
torch.get_deterministic_debug_mode
get_deterministic_debug_mode()
torch.set_float32_matmul_precision
set_float32_matmul_precision()
torch.get_float32_matmul_precision
get_float32_matmul_precision()
torch.set_warn_always
set_warn_always()
torch.get_device_module
get_device_module()
torch.is_warn_always_enabled
is_warn_always_enabled()
torch.vmap
vmap()
torch._assert
_assert()

torch.sym_float
sym_float()
torch.sym_fresh_size
sym_fresh_size()
torch.sym_int
sym_int()
torch.sym_max
sym_max()
torch.sym_min
sym_min()
torch.sym_not
sym_not()
torch.sym_ite
sym_ite()
torch.sym_sum
sym_sum()
torch.cond
cond()
torch.compile
compile()
torch.nn.utils.clip_grad_norm_
clip_grad_norm_()
torch.nn.utils.clip_grad_value_
clip_grad_value_()
torch.autograd.grad
grad()
DataParallel
DataParallel
DistributedDataParallel
DistributedDataParallel
DistributedDataParallel.join()
DistributedDataParallel.join_hook()
DistributedDataParallel.no_sync()
DistributedDataParallel.register_comm_hook()
Dropout
Dropout
BatchNorm2d
BatchNorm2d
torch.Tensor.register_hook
Tensor.register_hook()
torch.Tensor.register_post_accumulate_grad_hook
Tensor.register_post_accumulate_grad_hook()
torch.autograd.graph.Node.register_hook
Node.register_hook()
torch.autograd.graph.Node.register_prehook
Node.register_prehook()
torch.autograd.backward
backward()
torch.nn.modules.module.register_module_full_backward_hook
register_module_full_backward_hook()
device
device
torch.Tensor.copy_
Tensor.copy_()
torch.Tensor.to
Tensor.to()
torch.Tensor.cuda
Tensor.cuda()
torch.cuda.synchronize
synchronize()
Event
Event
Event.elapsed_time()
Event.from_ipc_handle()
Event.ipc_handle()
Event.query()
Event.record()
Event.synchronize()
Event.wait()
Stream
Stream
Stream.query()
Stream.record_event()
Stream.synchronize()
Stream.wait_event()
Stream.wait_stream()
torch.Tensor.record_stream
Tensor.record_stream()
torch.Tensor.backward
Tensor.backward()
torch.cuda.memory_allocated
memory_allocated()
torch.cuda.max_memory_allocated
max_memory_allocated()
torch.cuda.memory_reserved
memory_reserved()
torch.cuda.max_memory_reserved
max_memory_reserved()
torch.cuda.empty_cache
empty_cache()
torch.cuda.memory_stats
memory_stats()
torch.cuda.memory_snapshot
memory_snapshot()
torch.cuda.memory_summary
memory_summary()
CUDAPluggableAllocator
CUDAPluggableAllocator
torch.cuda.change_current_allocator
change_current_allocator()
MemPool
MemPool
MemPool.allocator
MemPool.id
MemPool.snapshot()
MemPool.use_count()
torch.fft.fft
fft()
torch.cuda.is_available
is_available()
torch.Tensor.pin_memory
Tensor.pin_memory()
Multiprocessing package - torch.multiprocessing
Strategy management
get_all_sharing_strategies()
get_sharing_strategy()
set_sharing_strategy()


Sharing CUDA tensors
Sharing strategies
File descriptor - file_descriptor
File system - file_system


Spawning subprocesses
spawn()
SpawnContext
SpawnContext.join()
CUDAGraph
CUDAGraph
CUDAGraph.capture_begin()
CUDAGraph.capture_end()
CUDAGraph.debug_dump()
CUDAGraph.enable_debug_mode()
CUDAGraph.pool()
CUDAGraph.replay()
CUDAGraph.reset()
graph
graph
torch.cuda.make_graphed_callables
make_graphed_callables()
Module
Module
Module.add_module()
Module.apply()
Module.bfloat16()
Module.buffers()
Module.children()
Module.compile()
Module.cpu()
Module.cuda()
Module.double()
Module.eval()
Module.extra_repr()
Module.float()
Module.forward()
Module.get_buffer()
Module.get_extra_state()
Module.get_parameter()
Module.get_submodule()
Module.half()
Module.ipu()
Module.load_state_dict()
Module.modules()
Module.mtia()
Module.named_buffers()
Module.named_children()
Module.named_modules()
Module.named_parameters()
Module.parameters()
Module.register_backward_hook()
Module.register_buffer()
Module.register_forward_hook()
Module.register_forward_pre_hook()
Module.register_full_backward_hook()
Module.register_full_backward_pre_hook()
Module.register_load_state_dict_post_hook()
Module.register_load_state_dict_pre_hook()
Module.register_module()
Module.register_parameter()
Module.register_state_dict_post_hook()
Module.register_state_dict_pre_hook()
Module.requires_grad_()
Module.set_extra_state()
Module.set_submodule()
Module.share_memory()
Module.state_dict()
Module.to()
Module.to_empty()
Module.train()
Module.type()
Module.xpu()
Module.zero_grad()
Linear
Linear
torch.autograd.Function.forward
Function.forward()
torch.autograd.Function.backward
Function.backward()
torch.autograd.function.FunctionCtx.save_for_backward
FunctionCtx.save_for_backward()
torch.autograd.function.FunctionCtx.mark_dirty
FunctionCtx.mark_dirty()
torch.autograd.function.FunctionCtx.mark_non_differentiable
FunctionCtx.mark_non_differentiable()
torch.autograd.function.FunctionCtx.set_materialize_grads
FunctionCtx.set_materialize_grads()
torch.autograd.function.once_differentiable
once_differentiable()
torch.func API Reference
Function Transforms
Utilities for working with torch.nn.Modules


Debug utilities
torch.autograd.Function.jvp
Function.jvp()
torch.func.grad
grad()
torch.func.vjp
vjp()
torch.autograd.Function.vmap
Function.vmap()
torch.func.jvp
jvp()
torch.func.jacrev
jacrev()
torch.func.jacfwd
jacfwd()
torch.func.hessian
hessian()
torch.nn.functional.torch.nn.parallel.data_parallel
data_parallel()
torch.nn.utils.rnn.pad_packed_sequence
pad_packed_sequence()
torch.jit.save
save()
Parameter
Parameter
Sequential
Sequential
Sequential.append()
ReLU
ReLU
ModuleList
ModuleList
ModuleList.append()
ModuleList.extend()
ModuleList.insert()
ModuleDict
ModuleDict
ModuleDict.clear()
ModuleDict.items()
ModuleDict.keys()
ModuleDict.pop()
ModuleDict.update()
ModuleDict.values()
torch.nn.utils.skip_init
skip_init()
torch.nn.modules.module.register_module_forward_pre_hook
register_module_forward_pre_hook()
torch.nn.modules.module.register_module_forward_hook
register_module_forward_hook()
torch.nn.modules.module.register_module_full_backward_pre_hook
register_module_full_backward_pre_hook()
torch.Tensor.grad
Tensor.grad
torch.linalg.svdvals
svdvals()
torch.linalg.cond
cond()
RNN
RNN
LSTM
LSTM
torch.Tensor.resize_
Tensor.resize_()
torch.jit.load
load()

Buffer
Buffer
UninitializedParameter
UninitializedParameter
UninitializedParameter.cls_to_become
UninitializedBuffer
UninitializedBuffer
ParameterList
ParameterList
ParameterList.append()
ParameterList.extend()
ParameterDict
ParameterDict
ParameterDict.clear()
ParameterDict.copy()
ParameterDict.fromkeys()
ParameterDict.get()
ParameterDict.items()
ParameterDict.keys()
ParameterDict.pop()
ParameterDict.popitem()
ParameterDict.setdefault()
ParameterDict.update()
ParameterDict.values()
torch.nn.modules.module.register_module_backward_hook
register_module_backward_hook()
torch.nn.modules.module.register_module_buffer_registration_hook
register_module_buffer_registration_hook()
torch.nn.modules.module.register_module_module_registration_hook
register_module_module_registration_hook()
torch.nn.modules.module.register_module_parameter_registration_hook
register_module_parameter_registration_hook()
Conv1d
Conv1d
Conv2d
Conv2d
Conv3d
Conv3d
ConvTranspose1d
ConvTranspose1d
ConvTranspose2d
ConvTranspose2d
ConvTranspose3d
ConvTranspose3d
LazyConv1d
LazyConv1d
LazyConv1d.cls_to_become
LazyConv2d
LazyConv2d
LazyConv2d.cls_to_become
LazyConv3d
LazyConv3d
LazyConv3d.cls_to_become
LazyConvTranspose1d
LazyConvTranspose1d
LazyConvTranspose1d.cls_to_become
LazyConvTranspose2d
LazyConvTranspose2d
LazyConvTranspose2d.cls_to_become
LazyConvTranspose3d
LazyConvTranspose3d
LazyConvTranspose3d.cls_to_become
Unfold
Unfold
Fold
Fold
MaxPool1d
MaxPool1d
MaxPool2d
MaxPool2d
MaxPool3d
MaxPool3d
MaxUnpool1d
MaxUnpool1d
MaxUnpool2d
MaxUnpool2d
MaxUnpool3d
MaxUnpool3d
AvgPool1d
AvgPool1d
AvgPool2d
AvgPool2d
AvgPool3d
AvgPool3d
FractionalMaxPool2d
FractionalMaxPool2d
FractionalMaxPool3d
FractionalMaxPool3d
LPPool1d
LPPool1d
LPPool2d
LPPool2d
LPPool3d
LPPool3d
AdaptiveMaxPool1d
AdaptiveMaxPool1d
AdaptiveMaxPool2d
AdaptiveMaxPool2d
AdaptiveMaxPool3d
AdaptiveMaxPool3d
AdaptiveAvgPool1d
AdaptiveAvgPool1d
AdaptiveAvgPool2d
AdaptiveAvgPool2d
AdaptiveAvgPool3d
AdaptiveAvgPool3d
ReflectionPad1d
ReflectionPad1d
ReflectionPad2d
ReflectionPad2d
ReflectionPad3d
ReflectionPad3d
ReplicationPad1d
ReplicationPad1d
ReplicationPad2d
ReplicationPad2d
ReplicationPad3d
ReplicationPad3d
ZeroPad1d
ZeroPad1d
ZeroPad2d
ZeroPad2d
ZeroPad3d
ZeroPad3d
ConstantPad1d
ConstantPad1d
ConstantPad2d
ConstantPad2d
ConstantPad3d
ConstantPad3d
CircularPad1d
CircularPad1d
CircularPad2d
CircularPad2d
CircularPad3d
CircularPad3d
ELU
ELU
Hardshrink
Hardshrink
Hardsigmoid
Hardsigmoid
Hardtanh
Hardtanh
Hardswish
Hardswish
LeakyReLU
LeakyReLU
LogSigmoid
LogSigmoid
MultiheadAttention
MultiheadAttention
MultiheadAttention.forward()
MultiheadAttention.merge_masks()
PReLU
PReLU
ReLU6
ReLU6
RReLU
RReLU
SELU
SELU
CELU
CELU
GELU
GELU
Sigmoid
Sigmoid
SiLU
SiLU
Mish
Mish
Softplus
Softplus
Softshrink
Softshrink
Softsign
Softsign
Tanh
Tanh
Tanhshrink
Tanhshrink
Threshold
Threshold
GLU
GLU
Softmin
Softmin
Softmax
Softmax
Softmax2d
Softmax2d
LogSoftmax
LogSoftmax
AdaptiveLogSoftmaxWithLoss
AdaptiveLogSoftmaxWithLoss
AdaptiveLogSoftmaxWithLoss.log_prob()
AdaptiveLogSoftmaxWithLoss.predict()
BatchNorm1d
BatchNorm1d
BatchNorm3d
BatchNorm3d
LazyBatchNorm1d
LazyBatchNorm1d
LazyBatchNorm1d.cls_to_become
LazyBatchNorm2d
LazyBatchNorm2d
LazyBatchNorm2d.cls_to_become
LazyBatchNorm3d
LazyBatchNorm3d
LazyBatchNorm3d.cls_to_become
GroupNorm
GroupNorm
SyncBatchNorm
SyncBatchNorm
SyncBatchNorm.convert_sync_batchnorm()
InstanceNorm1d
InstanceNorm1d
InstanceNorm2d
InstanceNorm2d
InstanceNorm3d
InstanceNorm3d
LazyInstanceNorm1d
LazyInstanceNorm1d
LazyInstanceNorm1d.cls_to_become
LazyInstanceNorm2d
LazyInstanceNorm2d
LazyInstanceNorm2d.cls_to_become
LazyInstanceNorm3d
LazyInstanceNorm3d
LazyInstanceNorm3d.cls_to_become
LayerNorm
LayerNorm
LocalResponseNorm
LocalResponseNorm
RMSNorm
RMSNorm
RMSNorm.extra_repr()
RMSNorm.forward()
RMSNorm.reset_parameters()
RNNBase
RNNBase
RNNBase.flatten_parameters()
GRU
GRU
RNNCell
RNNCell
LSTMCell
LSTMCell
GRUCell
GRUCell
Transformer
Transformer
Transformer.forward()
Transformer.generate_square_subsequent_mask()
TransformerEncoder
TransformerEncoder
TransformerEncoder.forward()
TransformerDecoder
TransformerDecoder
TransformerDecoder.forward()
TransformerEncoderLayer
TransformerEncoderLayer
TransformerEncoderLayer.forward()
TransformerDecoderLayer
TransformerDecoderLayer
TransformerDecoderLayer.forward()
Identity
Identity
Bilinear
Bilinear
LazyLinear
LazyLinear
LazyLinear.cls_to_become
Dropout1d
Dropout1d
Dropout2d
Dropout2d
Dropout3d
Dropout3d
AlphaDropout
AlphaDropout
FeatureAlphaDropout
FeatureAlphaDropout
Embedding
Embedding
Embedding.from_pretrained()
EmbeddingBag
EmbeddingBag
EmbeddingBag.forward()
EmbeddingBag.from_pretrained()
CosineSimilarity
CosineSimilarity
PairwiseDistance
PairwiseDistance
L1Loss
L1Loss
MSELoss
MSELoss
CrossEntropyLoss
CrossEntropyLoss
CTCLoss
CTCLoss
NLLLoss
NLLLoss
PoissonNLLLoss
PoissonNLLLoss
GaussianNLLLoss
GaussianNLLLoss
KLDivLoss
KLDivLoss
BCELoss
BCELoss
BCEWithLogitsLoss
BCEWithLogitsLoss
MarginRankingLoss
MarginRankingLoss
HingeEmbeddingLoss
HingeEmbeddingLoss
MultiLabelMarginLoss
MultiLabelMarginLoss
HuberLoss
HuberLoss
SmoothL1Loss
SmoothL1Loss
SoftMarginLoss
SoftMarginLoss
MultiLabelSoftMarginLoss
MultiLabelSoftMarginLoss
CosineEmbeddingLoss
CosineEmbeddingLoss
MultiMarginLoss
MultiMarginLoss
TripletMarginLoss
TripletMarginLoss
TripletMarginWithDistanceLoss
TripletMarginWithDistanceLoss
PixelShuffle
PixelShuffle
PixelUnshuffle
PixelUnshuffle
Upsample
Upsample
UpsamplingNearest2d
UpsamplingNearest2d
UpsamplingBilinear2d
UpsamplingBilinear2d
ChannelShuffle
ChannelShuffle
torch.nn.utils.clip_grad_norm
clip_grad_norm()
torch.nn.utils.get_total_norm
get_total_norm()
torch.nn.utils.clip_grads_with_norm_
clip_grads_with_norm_()
torch.nn.utils.parameters_to_vector
parameters_to_vector()
torch.nn.utils.vector_to_parameters
vector_to_parameters()
torch.nn.utils.fuse_conv_bn_eval
fuse_conv_bn_eval()
torch.nn.utils.fuse_conv_bn_weights
fuse_conv_bn_weights()
torch.nn.utils.fuse_linear_bn_eval
fuse_linear_bn_eval()
torch.nn.utils.fuse_linear_bn_weights
fuse_linear_bn_weights()
torch.nn.utils.convert_conv2d_weight_memory_format
convert_conv2d_weight_memory_format()
torch.nn.utils.convert_conv3d_weight_memory_format
convert_conv3d_weight_memory_format()
torch.nn.utils.weight_norm
weight_norm()
torch.nn.utils.remove_weight_norm
remove_weight_norm()
torch.nn.utils.spectral_norm
spectral_norm()
torch.nn.utils.remove_spectral_norm
remove_spectral_norm()
BasePruningMethod
BasePruningMethod
BasePruningMethod.apply()
BasePruningMethod.apply_mask()
BasePruningMethod.compute_mask()
BasePruningMethod.prune()
BasePruningMethod.remove()
PruningContainer
PruningContainer
PruningContainer.add_pruning_method()
PruningContainer.apply()
PruningContainer.apply_mask()
PruningContainer.compute_mask()
PruningContainer.prune()
PruningContainer.remove()
Identity
Identity
Identity.apply()
Identity.apply_mask()
Identity.prune()
Identity.remove()
RandomUnstructured
RandomUnstructured
RandomUnstructured.apply()
RandomUnstructured.apply_mask()
RandomUnstructured.prune()
RandomUnstructured.remove()
L1Unstructured
L1Unstructured
L1Unstructured.apply()
L1Unstructured.apply_mask()
L1Unstructured.prune()
L1Unstructured.remove()
RandomStructured
RandomStructured
RandomStructured.apply()
RandomStructured.apply_mask()
RandomStructured.compute_mask()
RandomStructured.prune()
RandomStructured.remove()
LnStructured
LnStructured
LnStructured.apply()
LnStructured.apply_mask()
LnStructured.compute_mask()
LnStructured.prune()
LnStructured.remove()
CustomFromMask
CustomFromMask
CustomFromMask.apply()
CustomFromMask.apply_mask()
CustomFromMask.prune()
CustomFromMask.remove()
torch.nn.utils.prune.identity
identity()
torch.nn.utils.prune.random_unstructured
random_unstructured()
torch.nn.utils.prune.l1_unstructured
l1_unstructured()
torch.nn.utils.prune.random_structured
random_structured()
torch.nn.utils.prune.ln_structured
ln_structured()
torch.nn.utils.prune.global_unstructured
global_unstructured()
torch.nn.utils.prune.custom_from_mask
custom_from_mask()
torch.nn.utils.prune.remove
remove()
torch.nn.utils.prune.is_pruned
is_pruned()
torch.nn.utils.parametrizations.orthogonal
orthogonal()
torch.nn.utils.parametrizations.weight_norm
weight_norm()
torch.nn.utils.parametrizations.spectral_norm
spectral_norm()
torch.nn.utils.parametrize.register_parametrization
register_parametrization()
torch.nn.utils.parametrize.remove_parametrizations
remove_parametrizations()
torch.nn.utils.parametrize.cached
cached()
torch.nn.utils.parametrize.is_parametrized
is_parametrized()
ParametrizationList
ParametrizationList
ParametrizationList.right_inverse()
torch.nn.utils.stateless.functional_call
functional_call()
PackedSequence
PackedSequence
PackedSequence.batch_sizes
PackedSequence.count()
PackedSequence.data
PackedSequence.index()
PackedSequence.is_cuda
PackedSequence.is_pinned()
PackedSequence.sorted_indices
PackedSequence.to()
PackedSequence.unsorted_indices
torch.nn.utils.rnn.pack_padded_sequence
pack_padded_sequence()
torch.nn.utils.rnn.pad_sequence
pad_sequence()
torch.nn.utils.rnn.pack_sequence
pack_sequence()
torch.nn.utils.rnn.unpack_sequence
unpack_sequence()
torch.nn.utils.rnn.unpad_sequence
unpad_sequence()
Flatten
Flatten
Unflatten
Unflatten
Unflatten.NamedShape
LazyModuleMixin
LazyModuleMixin
LazyModuleMixin.has_uninitialized_params()
LazyModuleMixin.initialize_parameters()
RMSNorm
RMSNorm
RMSNorm.extra_repr()
RMSNorm.forward()
RMSNorm.reset_parameters()
torch.nn.functional.conv1d
conv1d()
torch.nn.functional.conv2d
conv2d()
torch.nn.functional.conv3d
conv3d()
torch.nn.functional.conv_transpose1d
conv_transpose1d()
torch.nn.functional.conv_transpose2d
conv_transpose2d()
torch.nn.functional.conv_transpose3d
conv_transpose3d()
torch.nn.functional.unfold
unfold()
torch.nn.functional.fold
fold()
torch.nn.functional.avg_pool1d
avg_pool1d()
torch.nn.functional.avg_pool2d
avg_pool2d()
torch.nn.functional.avg_pool3d
avg_pool3d()
torch.nn.functional.max_pool1d
max_pool1d()
torch.nn.functional.max_pool2d
max_pool2d()
torch.nn.functional.max_pool3d
max_pool3d()
torch.nn.functional.max_unpool1d
max_unpool1d()
torch.nn.functional.max_unpool2d
max_unpool2d()
torch.nn.functional.max_unpool3d
max_unpool3d()
torch.nn.functional.lp_pool1d
lp_pool1d()
torch.nn.functional.lp_pool2d
lp_pool2d()
torch.nn.functional.lp_pool3d
lp_pool3d()
torch.nn.functional.adaptive_max_pool1d
adaptive_max_pool1d()
torch.nn.functional.adaptive_max_pool2d
adaptive_max_pool2d()
torch.nn.functional.adaptive_max_pool3d
adaptive_max_pool3d()
torch.nn.functional.adaptive_avg_pool1d
adaptive_avg_pool1d()
torch.nn.functional.adaptive_avg_pool2d
adaptive_avg_pool2d()
torch.nn.functional.adaptive_avg_pool3d
adaptive_avg_pool3d()
torch.nn.functional.fractional_max_pool2d
fractional_max_pool2d()
torch.nn.functional.fractional_max_pool3d
fractional_max_pool3d()
torch.nn.attention.bias
CausalBias
torch.nn.functional.scaled_dot_product_attention
scaled_dot_product_attention()
torch.nn.functional.threshold
threshold()
torch.nn.functional.threshold_
threshold_()
torch.nn.functional.relu
relu()
torch.nn.functional.relu_
relu_()
torch.nn.functional.hardtanh
hardtanh()
torch.nn.functional.hardtanh_
hardtanh_()
torch.nn.functional.hardswish
hardswish()
torch.nn.functional.relu6
relu6()
torch.nn.functional.elu
elu()
torch.nn.functional.elu_
elu_()
torch.nn.functional.selu
selu()
torch.nn.functional.celu
celu()
torch.nn.functional.leaky_relu
leaky_relu()
torch.nn.functional.leaky_relu_
leaky_relu_()
torch.nn.functional.prelu
prelu()
torch.nn.functional.rrelu
rrelu()
torch.nn.functional.rrelu_
rrelu_()
torch.nn.functional.glu
glu()
torch.nn.functional.gelu
gelu()
torch.nn.functional.logsigmoid
logsigmoid()
torch.nn.functional.hardshrink
hardshrink()
torch.nn.functional.tanhshrink
tanhshrink()
torch.nn.functional.softsign
softsign()
torch.nn.functional.softplus
softplus()
torch.nn.functional.softmin
softmin()
torch.nn.functional.softshrink
softshrink()
torch.nn.functional.gumbel_softmax
gumbel_softmax()
torch.nn.functional.log_softmax
log_softmax()
torch.nn.functional.tanh
tanh()
torch.nn.functional.sigmoid
sigmoid()
torch.nn.functional.hardsigmoid
hardsigmoid()
torch.nn.functional.silu
silu()
torch.nn.functional.mish
mish()
torch.nn.functional.batch_norm
batch_norm()
torch.nn.functional.group_norm
group_norm()
torch.nn.functional.instance_norm
instance_norm()
torch.nn.functional.layer_norm
layer_norm()
torch.nn.functional.local_response_norm
local_response_norm()
torch.nn.functional.rms_norm
rms_norm()
torch.nn.functional.normalize
normalize()
torch.nn.functional.linear
linear()
torch.nn.functional.bilinear
bilinear()
torch.nn.functional.dropout
dropout()
torch.nn.functional.alpha_dropout
alpha_dropout()
torch.nn.functional.feature_alpha_dropout
feature_alpha_dropout()
torch.nn.functional.dropout1d
dropout1d()
torch.nn.functional.dropout2d
dropout2d()
torch.nn.functional.dropout3d
dropout3d()
torch.nn.functional.embedding
embedding()
torch.nn.functional.embedding_bag
embedding_bag()
torch.nn.functional.one_hot
one_hot()
torch.nn.functional.pairwise_distance
pairwise_distance()
torch.nn.functional.cosine_similarity
cosine_similarity()
torch.nn.functional.pdist
pdist()
torch.nn.functional.binary_cross_entropy
binary_cross_entropy()
torch.nn.functional.binary_cross_entropy_with_logits
binary_cross_entropy_with_logits()
torch.nn.functional.poisson_nll_loss
poisson_nll_loss()
torch.nn.functional.cosine_embedding_loss
cosine_embedding_loss()
torch.nn.functional.cross_entropy
cross_entropy()
torch.nn.functional.ctc_loss
ctc_loss()
torch.nn.functional.gaussian_nll_loss
gaussian_nll_loss()
torch.nn.functional.hinge_embedding_loss
hinge_embedding_loss()
torch.nn.functional.kl_div
kl_div()
torch.nn.functional.l1_loss
l1_loss()
torch.nn.functional.mse_loss
mse_loss()
torch.nn.functional.margin_ranking_loss
margin_ranking_loss()
torch.nn.functional.multilabel_margin_loss
multilabel_margin_loss()
torch.nn.functional.multilabel_soft_margin_loss
multilabel_soft_margin_loss()
torch.nn.functional.multi_margin_loss
multi_margin_loss()
torch.nn.functional.nll_loss
nll_loss()
torch.nn.functional.huber_loss
huber_loss()
torch.nn.functional.smooth_l1_loss
smooth_l1_loss()
torch.nn.functional.soft_margin_loss
soft_margin_loss()
torch.nn.functional.triplet_margin_loss
triplet_margin_loss()
torch.nn.functional.triplet_margin_with_distance_loss
triplet_margin_with_distance_loss()
torch.nn.functional.pixel_shuffle
pixel_shuffle()
torch.nn.functional.pixel_unshuffle
pixel_unshuffle()
torch.nn.functional.pad
pad()
torch.nn.functional.interpolate
interpolate()
torch.nn.functional.upsample
upsample()
torch.nn.functional.upsample_nearest
upsample_nearest()
torch.nn.functional.upsample_bilinear
upsample_bilinear()
torch.nn.functional.grid_sample
grid_sample()
torch.nn.functional.affine_grid
affine_grid()
torch.Tensor.requires_grad_
Tensor.requires_grad_()
torch.Tensor.detach
Tensor.detach()
torch.Tensor.item
Tensor.item()
torch.Tensor.new_tensor
Tensor.new_tensor()
torch.Tensor.new_full
Tensor.new_full()
torch.Tensor.new_empty
Tensor.new_empty()
torch.Tensor.new_ones
Tensor.new_ones()
torch.Tensor.new_zeros
Tensor.new_zeros()
torch.Tensor.is_cuda
Tensor.is_cuda
torch.Tensor.is_quantized
Tensor.is_quantized
torch.Tensor.is_meta
Tensor.is_meta
torch.Tensor.device
Tensor.device
torch.Tensor.ndim
Tensor.ndim
torch.Tensor.dim
Tensor.dim()
torch.Tensor.real
Tensor.real
torch.Tensor.imag
Tensor.imag
torch.Tensor.nbytes
Tensor.nbytes
torch.Tensor.itemsize
Tensor.itemsize
torch.Tensor.element_size
Tensor.element_size()
torch.Tensor.abs
Tensor.abs()
torch.Tensor.abs_
Tensor.abs_()
torch.Tensor.absolute
Tensor.absolute()
torch.Tensor.absolute_
Tensor.absolute_()
torch.Tensor.acos
Tensor.acos()
torch.Tensor.acos_
Tensor.acos_()
torch.Tensor.arccos
Tensor.arccos()
torch.Tensor.arccos_
Tensor.arccos_()
torch.Tensor.add
Tensor.add()
torch.Tensor.add_
Tensor.add_()
torch.Tensor.addbmm
Tensor.addbmm()
torch.Tensor.addbmm_
Tensor.addbmm_()
torch.Tensor.addcdiv
Tensor.addcdiv()
torch.Tensor.addcdiv_
Tensor.addcdiv_()
torch.Tensor.addcmul
Tensor.addcmul()
torch.Tensor.addcmul_
Tensor.addcmul_()
torch.Tensor.addmm
Tensor.addmm()
torch.Tensor.addmm_
Tensor.addmm_()
torch.Tensor.sspaddmm
Tensor.sspaddmm()
torch.sspaddmm
sspaddmm()
torch.Tensor.addmv
Tensor.addmv()
torch.Tensor.addmv_
Tensor.addmv_()
torch.Tensor.addr
Tensor.addr()
torch.Tensor.addr_
Tensor.addr_()
torch.Tensor.adjoint
Tensor.adjoint()
torch.Tensor.allclose
Tensor.allclose()
torch.Tensor.amax
Tensor.amax()
torch.Tensor.amin
Tensor.amin()
torch.Tensor.aminmax
Tensor.aminmax()
torch.Tensor.angle
Tensor.angle()
torch.Tensor.apply_
Tensor.apply_()
torch.Tensor.argmax
Tensor.argmax()
torch.Tensor.argmin
Tensor.argmin()
torch.Tensor.argsort
Tensor.argsort()
torch.Tensor.argwhere
Tensor.argwhere()
torch.Tensor.asin
Tensor.asin()
torch.Tensor.asin_
Tensor.asin_()
torch.Tensor.arcsin
Tensor.arcsin()
torch.Tensor.arcsin_
Tensor.arcsin_()
torch.Tensor.as_strided
Tensor.as_strided()
torch.Tensor.atan
Tensor.atan()
torch.Tensor.atan_
Tensor.atan_()
torch.Tensor.arctan
Tensor.arctan()
torch.Tensor.arctan_
Tensor.arctan_()
torch.Tensor.atan2
Tensor.atan2()
torch.Tensor.atan2_
Tensor.atan2_()
torch.Tensor.arctan2
Tensor.arctan2()
torch.Tensor.arctan2_
Tensor.arctan2_()
torch.Tensor.all
Tensor.all()
torch.Tensor.any
Tensor.any()
torch.Tensor.baddbmm
Tensor.baddbmm()
torch.Tensor.baddbmm_
Tensor.baddbmm_()
torch.Tensor.bernoulli
Tensor.bernoulli()
torch.Tensor.bfloat16
Tensor.bfloat16()
torch.Tensor.bincount
Tensor.bincount()
torch.Tensor.bitwise_not
Tensor.bitwise_not()
torch.Tensor.bitwise_not_
Tensor.bitwise_not_()
torch.Tensor.bitwise_and
Tensor.bitwise_and()
torch.Tensor.bitwise_and_
Tensor.bitwise_and_()
torch.Tensor.bitwise_or
Tensor.bitwise_or()
torch.Tensor.bitwise_or_
Tensor.bitwise_or_()
torch.Tensor.bitwise_xor
Tensor.bitwise_xor()
torch.Tensor.bitwise_xor_
Tensor.bitwise_xor_()
torch.Tensor.bitwise_left_shift
Tensor.bitwise_left_shift()
torch.Tensor.bitwise_left_shift_
Tensor.bitwise_left_shift_()
torch.Tensor.bitwise_right_shift
Tensor.bitwise_right_shift()
torch.Tensor.bitwise_right_shift_
Tensor.bitwise_right_shift_()
torch.Tensor.bmm
Tensor.bmm()
torch.Tensor.bool
Tensor.bool()
torch.Tensor.byte
Tensor.byte()
torch.Tensor.broadcast_to
Tensor.broadcast_to()
torch.Tensor.ceil
Tensor.ceil()
torch.Tensor.ceil_
Tensor.ceil_()
torch.Tensor.char
Tensor.char()
torch.Tensor.cholesky
Tensor.cholesky()
torch.Tensor.cholesky_inverse
Tensor.cholesky_inverse()
torch.Tensor.cholesky_solve
Tensor.cholesky_solve()
torch.Tensor.chunk
Tensor.chunk()
torch.Tensor.clamp
Tensor.clamp()
torch.Tensor.clamp_
Tensor.clamp_()
torch.Tensor.clip
Tensor.clip()
torch.Tensor.clip_
Tensor.clip_()
torch.Tensor.clone
Tensor.clone()
torch.Tensor.contiguous
Tensor.contiguous()
torch.Tensor.conj
Tensor.conj()
torch.Tensor.conj_physical
Tensor.conj_physical()
torch.Tensor.conj_physical_
Tensor.conj_physical_()
torch.Tensor.resolve_conj
Tensor.resolve_conj()
torch.Tensor.resolve_neg
Tensor.resolve_neg()
torch.Tensor.copysign
Tensor.copysign()
torch.Tensor.copysign_
Tensor.copysign_()
torch.Tensor.cos
Tensor.cos()
torch.Tensor.cos_
Tensor.cos_()
torch.Tensor.cosh
Tensor.cosh()
torch.Tensor.cosh_
Tensor.cosh_()
torch.Tensor.corrcoef
Tensor.corrcoef()
torch.Tensor.count_nonzero
Tensor.count_nonzero()
torch.Tensor.cov
Tensor.cov()
torch.Tensor.acosh
Tensor.acosh()
torch.Tensor.acosh_
Tensor.acosh_()
torch.Tensor.arccosh
Tensor.arccosh()
torch.Tensor.arccosh_
Tensor.arccosh_()
torch.Tensor.cpu
Tensor.cpu()
torch.Tensor.cross
Tensor.cross()
torch.Tensor.logcumsumexp
Tensor.logcumsumexp()
torch.Tensor.cummax
Tensor.cummax()
torch.Tensor.cummin
Tensor.cummin()
torch.Tensor.cumprod
Tensor.cumprod()
torch.Tensor.cumprod_
Tensor.cumprod_()
torch.Tensor.cumsum
Tensor.cumsum()
torch.Tensor.cumsum_
Tensor.cumsum_()
torch.Tensor.chalf
Tensor.chalf()
torch.Tensor.cfloat
Tensor.cfloat()
torch.Tensor.cdouble
Tensor.cdouble()
torch.Tensor.data_ptr
Tensor.data_ptr()
torch.Tensor.deg2rad
Tensor.deg2rad()
torch.Tensor.dequantize
Tensor.dequantize()
torch.Tensor.det
Tensor.det()
torch.Tensor.dense_dim
Tensor.dense_dim()
torch.Tensor.detach_
Tensor.detach_()
torch.Tensor.diag
Tensor.diag()
torch.Tensor.diag_embed
Tensor.diag_embed()
torch.Tensor.diagflat
Tensor.diagflat()
torch.Tensor.diagonal
Tensor.diagonal()
torch.Tensor.diagonal_scatter
Tensor.diagonal_scatter()
torch.Tensor.fill_diagonal_
Tensor.fill_diagonal_()
torch.Tensor.fmax
Tensor.fmax()
torch.Tensor.fmin
Tensor.fmin()
torch.Tensor.diff
Tensor.diff()
torch.Tensor.digamma
Tensor.digamma()
torch.Tensor.digamma_
Tensor.digamma_()
torch.Tensor.dim_order
Tensor.dim_order()
torch.Tensor.dist
Tensor.dist()
torch.Tensor.div
Tensor.div()
torch.Tensor.div_
Tensor.div_()
torch.Tensor.divide
Tensor.divide()
torch.Tensor.divide_
Tensor.divide_()
torch.Tensor.dot
Tensor.dot()
torch.Tensor.double
Tensor.double()
torch.Tensor.dsplit
Tensor.dsplit()
torch.Tensor.eq
Tensor.eq()
torch.Tensor.eq_
Tensor.eq_()
torch.Tensor.equal
Tensor.equal()
torch.Tensor.erf
Tensor.erf()
torch.Tensor.erf_
Tensor.erf_()
torch.Tensor.erfc
Tensor.erfc()
torch.Tensor.erfc_
Tensor.erfc_()
torch.Tensor.erfinv
Tensor.erfinv()
torch.Tensor.erfinv_
Tensor.erfinv_()
torch.Tensor.exp
Tensor.exp()
torch.Tensor.exp_
Tensor.exp_()
torch.Tensor.expm1
Tensor.expm1()
torch.Tensor.expm1_
Tensor.expm1_()
torch.Tensor.expand
Tensor.expand()
torch.Tensor.expand_as
Tensor.expand_as()
torch.Tensor.fix
Tensor.fix()
torch.Tensor.fix_
Tensor.fix_()
torch.Tensor.fill_
Tensor.fill_()
torch.Tensor.flatten
Tensor.flatten()
torch.Tensor.flip
Tensor.flip()
torch.Tensor.fliplr
Tensor.fliplr()
torch.Tensor.flipud
Tensor.flipud()
torch.Tensor.float
Tensor.float()
torch.Tensor.float_power
Tensor.float_power()
torch.Tensor.float_power_
Tensor.float_power_()
torch.Tensor.floor
Tensor.floor()
torch.Tensor.floor_
Tensor.floor_()
torch.Tensor.floor_divide
Tensor.floor_divide()
torch.Tensor.floor_divide_
Tensor.floor_divide_()
torch.Tensor.fmod
Tensor.fmod()
torch.Tensor.fmod_
Tensor.fmod_()
torch.Tensor.frac
Tensor.frac()
torch.Tensor.frac_
Tensor.frac_()
torch.Tensor.frexp
Tensor.frexp()
torch.Tensor.gather
Tensor.gather()
torch.Tensor.gcd
Tensor.gcd()
torch.Tensor.gcd_
Tensor.gcd_()
torch.Tensor.ge
Tensor.ge()
torch.Tensor.ge_
Tensor.ge_()
torch.Tensor.greater_equal
Tensor.greater_equal()
torch.Tensor.greater_equal_
Tensor.greater_equal_()
torch.Tensor.geqrf
Tensor.geqrf()
torch.Tensor.ger
Tensor.ger()
torch.Tensor.get_device
Tensor.get_device()
torch.Tensor.gt
Tensor.gt()
torch.Tensor.gt_
Tensor.gt_()
torch.Tensor.greater
Tensor.greater()
torch.Tensor.greater_
Tensor.greater_()
torch.Tensor.half
Tensor.half()
torch.Tensor.hardshrink
Tensor.hardshrink()
torch.Tensor.heaviside
Tensor.heaviside()
torch.Tensor.histc
Tensor.histc()
torch.Tensor.histogram
Tensor.histogram()
torch.Tensor.hsplit
Tensor.hsplit()
torch.Tensor.hypot
Tensor.hypot()
torch.Tensor.hypot_
Tensor.hypot_()
torch.Tensor.i0
Tensor.i0()
torch.Tensor.i0_
Tensor.i0_()
torch.Tensor.igamma
Tensor.igamma()
torch.Tensor.igamma_
Tensor.igamma_()
torch.Tensor.igammac
Tensor.igammac()
torch.Tensor.igammac_
Tensor.igammac_()
torch.Tensor.index_add
Tensor.index_add()
torch.Tensor.index_copy_
Tensor.index_copy_()
torch.Tensor.index_copy
Tensor.index_copy()
torch.Tensor.index_fill_
Tensor.index_fill_()
torch.Tensor.index_fill
Tensor.index_fill()
torch.Tensor.index_put_
Tensor.index_put_()
torch.Tensor.index_put
Tensor.index_put()
torch.Tensor.index_reduce
Tensor.index_reduce()
torch.Tensor.index_select
Tensor.index_select()
torch.Tensor.indices
Tensor.indices()
torch.Tensor.inner
Tensor.inner()
torch.Tensor.int
Tensor.int()
torch.Tensor.int_repr
Tensor.int_repr()
torch.Tensor.inverse
Tensor.inverse()
torch.Tensor.isclose
Tensor.isclose()
torch.Tensor.isfinite
Tensor.isfinite()
torch.Tensor.isinf
Tensor.isinf()
torch.Tensor.isposinf
Tensor.isposinf()
torch.Tensor.isneginf
Tensor.isneginf()
torch.Tensor.isnan
Tensor.isnan()
torch.Tensor.is_contiguous
Tensor.is_contiguous()
torch.Tensor.is_complex
Tensor.is_complex()
torch.Tensor.is_conj
Tensor.is_conj()
torch.Tensor.is_floating_point
Tensor.is_floating_point()
torch.Tensor.is_inference
Tensor.is_inference()
torch.Tensor.is_leaf
Tensor.is_leaf
torch.Tensor.is_pinned
Tensor.is_pinned()
torch.Tensor.is_set_to
Tensor.is_set_to()
torch.Tensor.is_shared
Tensor.is_shared()
torch.Tensor.is_signed
Tensor.is_signed()
torch.Tensor.is_sparse
Tensor.is_sparse
torch.Tensor.istft
Tensor.istft()
torch.Tensor.isreal
Tensor.isreal()
torch.Tensor.kthvalue
Tensor.kthvalue()
torch.Tensor.lcm
Tensor.lcm()
torch.Tensor.lcm_
Tensor.lcm_()
torch.Tensor.ldexp
Tensor.ldexp()
torch.Tensor.ldexp_
Tensor.ldexp_()
torch.Tensor.le
Tensor.le()
torch.Tensor.le_
Tensor.le_()
torch.Tensor.less_equal
Tensor.less_equal()
torch.Tensor.less_equal_
Tensor.less_equal_()
torch.Tensor.lerp
Tensor.lerp()
torch.Tensor.lerp_
Tensor.lerp_()
torch.Tensor.lgamma
Tensor.lgamma()
torch.Tensor.lgamma_
Tensor.lgamma_()
torch.Tensor.log
Tensor.log()
torch.Tensor.log_
Tensor.log_()
torch.Tensor.logdet
Tensor.logdet()
torch.Tensor.log10
Tensor.log10()
torch.Tensor.log10_
Tensor.log10_()
torch.Tensor.log1p
Tensor.log1p()
torch.Tensor.log1p_
Tensor.log1p_()
torch.Tensor.log2
Tensor.log2()
torch.Tensor.log2_
Tensor.log2_()
torch.Tensor.logaddexp
Tensor.logaddexp()
torch.Tensor.logaddexp2
Tensor.logaddexp2()
torch.Tensor.logsumexp
Tensor.logsumexp()
torch.Tensor.logical_and
Tensor.logical_and()
torch.Tensor.logical_and_
Tensor.logical_and_()
torch.Tensor.logical_not
Tensor.logical_not()
torch.Tensor.logical_not_
Tensor.logical_not_()
torch.Tensor.logical_or
Tensor.logical_or()
torch.Tensor.logical_or_
Tensor.logical_or_()
torch.Tensor.logical_xor
Tensor.logical_xor()
torch.Tensor.logical_xor_
Tensor.logical_xor_()
torch.Tensor.logit
Tensor.logit()
torch.Tensor.logit_
Tensor.logit_()
torch.Tensor.long
Tensor.long()
torch.Tensor.lt
Tensor.lt()
torch.Tensor.lt_
Tensor.lt_()
torch.Tensor.less
Tensor.less()
torch.Tensor.less_
Tensor.less_()
torch.Tensor.lu
Tensor.lu()
torch.Tensor.lu_solve
Tensor.lu_solve()
torch.Tensor.as_subclass
Tensor.as_subclass()
torch.Tensor.map_
Tensor.map_()
torch.Tensor.masked_scatter_
Tensor.masked_scatter_()
torch.Tensor.masked_scatter
Tensor.masked_scatter()
torch.Tensor.masked_fill_
Tensor.masked_fill_()
torch.Tensor.masked_fill
Tensor.masked_fill()
torch.Tensor.masked_select
Tensor.masked_select()
torch.Tensor.matmul
Tensor.matmul()
torch.Tensor.matrix_power
Tensor.matrix_power()
torch.Tensor.matrix_exp
Tensor.matrix_exp()
torch.Tensor.max
Tensor.max()
torch.Tensor.maximum
Tensor.maximum()
torch.Tensor.mean
Tensor.mean()
torch.Tensor.module_load
Tensor.module_load()
torch.Tensor.nanmean
Tensor.nanmean()
torch.Tensor.median
Tensor.median()
torch.Tensor.nanmedian
Tensor.nanmedian()
torch.Tensor.min
Tensor.min()
torch.Tensor.minimum
Tensor.minimum()
torch.Tensor.mm
Tensor.mm()
torch.Tensor.smm
Tensor.smm()
torch.smm
smm()
torch.Tensor.mode
Tensor.mode()
torch.Tensor.movedim
Tensor.movedim()
torch.Tensor.moveaxis
Tensor.moveaxis()
torch.Tensor.msort
Tensor.msort()
torch.Tensor.mul
Tensor.mul()
torch.Tensor.mul_
Tensor.mul_()
torch.Tensor.multiply
Tensor.multiply()
torch.Tensor.multiply_
Tensor.multiply_()
torch.Tensor.multinomial
Tensor.multinomial()
torch.Tensor.mv
Tensor.mv()
torch.Tensor.mvlgamma
Tensor.mvlgamma()
torch.Tensor.mvlgamma_
Tensor.mvlgamma_()
torch.Tensor.nansum
Tensor.nansum()
torch.Tensor.narrow_copy
Tensor.narrow_copy()
torch.Tensor.ndimension
Tensor.ndimension()
torch.Tensor.nan_to_num
Tensor.nan_to_num()
torch.Tensor.nan_to_num_
Tensor.nan_to_num_()
torch.Tensor.ne
Tensor.ne()
torch.Tensor.ne_
Tensor.ne_()
torch.Tensor.not_equal
Tensor.not_equal()
torch.Tensor.not_equal_
Tensor.not_equal_()
torch.Tensor.neg
Tensor.neg()
torch.Tensor.neg_
Tensor.neg_()
torch.Tensor.negative
Tensor.negative()
torch.Tensor.negative_
Tensor.negative_()
torch.Tensor.nelement
Tensor.nelement()
torch.Tensor.numel
Tensor.numel()
torch.Tensor.nextafter
Tensor.nextafter()
torch.Tensor.nextafter_
Tensor.nextafter_()
torch.Tensor.nonzero
Tensor.nonzero()
torch.Tensor.norm
Tensor.norm()
torch.Tensor.numpy
Tensor.numpy()
torch.Tensor.orgqr
Tensor.orgqr()
torch.Tensor.ormqr
Tensor.ormqr()
torch.Tensor.outer
Tensor.outer()
torch.Tensor.permute
Tensor.permute()
torch.Tensor.pinverse
Tensor.pinverse()
torch.Tensor.polygamma
Tensor.polygamma()
torch.Tensor.polygamma_
Tensor.polygamma_()
torch.Tensor.positive
Tensor.positive()
torch.Tensor.pow
Tensor.pow()
torch.Tensor.pow_
Tensor.pow_()
torch.Tensor.prod
Tensor.prod()
torch.Tensor.put_
Tensor.put_()
torch.Tensor.qr
Tensor.qr()
torch.Tensor.qscheme
Tensor.qscheme()
torch.Tensor.quantile
Tensor.quantile()
torch.Tensor.nanquantile
Tensor.nanquantile()
torch.Tensor.q_scale
Tensor.q_scale()
torch.Tensor.q_zero_point
Tensor.q_zero_point()
torch.Tensor.q_per_channel_scales
Tensor.q_per_channel_scales()
torch.Tensor.q_per_channel_zero_points
Tensor.q_per_channel_zero_points()
torch.Tensor.q_per_channel_axis
Tensor.q_per_channel_axis()
torch.Tensor.rad2deg
Tensor.rad2deg()
torch.Tensor.ravel
Tensor.ravel()
torch.Tensor.reciprocal
Tensor.reciprocal()
torch.Tensor.reciprocal_
Tensor.reciprocal_()
torch.Tensor.remainder
Tensor.remainder()
torch.Tensor.remainder_
Tensor.remainder_()
torch.Tensor.renorm
Tensor.renorm()
torch.Tensor.renorm_
Tensor.renorm_()
torch.Tensor.repeat
Tensor.repeat()
torch.Tensor.repeat_interleave
Tensor.repeat_interleave()
torch.Tensor.requires_grad
Tensor.requires_grad
torch.Tensor.reshape
Tensor.reshape()
torch.Tensor.reshape_as
Tensor.reshape_as()
torch.Tensor.resize_as_
Tensor.resize_as_()
torch.Tensor.retain_grad
Tensor.retain_grad()
torch.Tensor.retains_grad
Tensor.retains_grad
torch.Tensor.roll
Tensor.roll()
torch.Tensor.rot90
Tensor.rot90()
torch.Tensor.round
Tensor.round()
torch.Tensor.round_
Tensor.round_()
torch.Tensor.rsqrt
Tensor.rsqrt()
torch.Tensor.rsqrt_
Tensor.rsqrt_()
torch.Tensor.scatter
Tensor.scatter()
torch.Tensor.scatter_add
Tensor.scatter_add()
torch.Tensor.scatter_reduce
Tensor.scatter_reduce()
torch.Tensor.select
Tensor.select()
torch.Tensor.select_scatter
Tensor.select_scatter()
torch.Tensor.set_
Tensor.set_()
torch.Tensor.share_memory_
Tensor.share_memory_()
torch.Tensor.short
Tensor.short()
torch.Tensor.sigmoid
Tensor.sigmoid()
torch.Tensor.sigmoid_
Tensor.sigmoid_()
torch.Tensor.sign
Tensor.sign()
torch.Tensor.sign_
Tensor.sign_()
torch.Tensor.signbit
Tensor.signbit()
torch.Tensor.sgn
Tensor.sgn()
torch.Tensor.sgn_
Tensor.sgn_()
torch.Tensor.sin
Tensor.sin()
torch.Tensor.sin_
Tensor.sin_()
torch.Tensor.sinc
Tensor.sinc()
torch.Tensor.sinc_
Tensor.sinc_()
torch.Tensor.sinh
Tensor.sinh()
torch.Tensor.sinh_
Tensor.sinh_()
torch.Tensor.asinh
Tensor.asinh()
torch.Tensor.asinh_
Tensor.asinh_()
torch.Tensor.arcsinh
Tensor.arcsinh()
torch.Tensor.arcsinh_
Tensor.arcsinh_()
torch.Tensor.shape
Tensor.shape
torch.Tensor.size
Tensor.size()
torch.Tensor.slogdet
Tensor.slogdet()
torch.Tensor.slice_scatter
Tensor.slice_scatter()
torch.Tensor.softmax
Tensor.softmax()
torch.Tensor.sort
Tensor.sort()
torch.Tensor.split
Tensor.split()
torch.Tensor.sparse_mask
Tensor.sparse_mask()
torch.Tensor.sparse_dim
Tensor.sparse_dim()
torch.Tensor.sqrt
Tensor.sqrt()
torch.Tensor.sqrt_
Tensor.sqrt_()
torch.Tensor.square
Tensor.square()
torch.Tensor.square_
Tensor.square_()
torch.Tensor.squeeze
Tensor.squeeze()
torch.Tensor.squeeze_
Tensor.squeeze_()
torch.Tensor.std
Tensor.std()
torch.Tensor.stft
Tensor.stft()
torch.Tensor.storage
Tensor.storage()
torch.Tensor.untyped_storage
Tensor.untyped_storage()
torch.Tensor.storage_offset
Tensor.storage_offset()
torch.Tensor.storage_type
Tensor.storage_type()
torch.Tensor.stride
Tensor.stride()
torch.Tensor.sub
Tensor.sub()
torch.Tensor.sub_
Tensor.sub_()
torch.Tensor.subtract
Tensor.subtract()
torch.Tensor.subtract_
Tensor.subtract_()
torch.Tensor.sum
Tensor.sum()
torch.Tensor.sum_to_size
Tensor.sum_to_size()
torch.Tensor.svd
Tensor.svd()
torch.Tensor.swapaxes
Tensor.swapaxes()
torch.Tensor.swapdims
Tensor.swapdims()
torch.Tensor.t
Tensor.t()
torch.Tensor.t_
Tensor.t_()
torch.Tensor.tensor_split
Tensor.tensor_split()
torch.Tensor.tile
Tensor.tile()
torch.Tensor.to_mkldnn
Tensor.to_mkldnn()
torch.Tensor.take
Tensor.take()
torch.Tensor.take_along_dim
Tensor.take_along_dim()
torch.Tensor.tan
Tensor.tan()
torch.Tensor.tan_
Tensor.tan_()
torch.Tensor.tanh
Tensor.tanh()
torch.Tensor.tanh_
Tensor.tanh_()
torch.Tensor.atanh
Tensor.atanh()
torch.Tensor.atanh_
Tensor.atanh_()
torch.Tensor.arctanh
Tensor.arctanh()
torch.Tensor.arctanh_
Tensor.arctanh_()
torch.Tensor.tolist
Tensor.tolist()
torch.Tensor.topk
Tensor.topk()
torch.Tensor.to_dense
Tensor.to_dense()
torch.Tensor.to_sparse
Tensor.to_sparse()
torch.Tensor.to_sparse_csr
Tensor.to_sparse_csr()
torch.Tensor.to_sparse_csc
Tensor.to_sparse_csc()
torch.Tensor.to_sparse_bsr
Tensor.to_sparse_bsr()
torch.Tensor.to_sparse_bsc
Tensor.to_sparse_bsc()
torch.Tensor.trace
Tensor.trace()
torch.Tensor.transpose
Tensor.transpose()
torch.Tensor.transpose_
Tensor.transpose_()
torch.Tensor.triangular_solve
Tensor.triangular_solve()
torch.Tensor.tril
Tensor.tril()
torch.Tensor.tril_
Tensor.tril_()
torch.Tensor.triu
Tensor.triu()
torch.Tensor.triu_
Tensor.triu_()
torch.Tensor.true_divide
Tensor.true_divide()
torch.Tensor.true_divide_
Tensor.true_divide_()
torch.Tensor.trunc
Tensor.trunc()
torch.Tensor.trunc_
Tensor.trunc_()
torch.Tensor.type
Tensor.type()
torch.Tensor.type_as
Tensor.type_as()
torch.Tensor.unbind
Tensor.unbind()
torch.Tensor.unflatten
Tensor.unflatten()
torch.Tensor.unfold
Tensor.unfold()
torch.Tensor.unique
Tensor.unique()
torch.Tensor.unique_consecutive
Tensor.unique_consecutive()
torch.Tensor.unsqueeze
Tensor.unsqueeze()
torch.Tensor.unsqueeze_
Tensor.unsqueeze_()
torch.Tensor.values
Tensor.values()
torch.Tensor.var
Tensor.var()
torch.Tensor.vdot
Tensor.vdot()
torch.Tensor.view
Tensor.view()
torch.Tensor.view_as
Tensor.view_as()
torch.Tensor.vsplit
Tensor.vsplit()
torch.Tensor.where
Tensor.where()
torch.Tensor.xlogy
Tensor.xlogy()
torch.Tensor.xlogy_
Tensor.xlogy_()
torch.Tensor.xpu
Tensor.xpu()
torch.Tensor.zero_
Tensor.zero_()
torch.cuda.set_device
set_device()
torch.cuda.current_device
current_device()





dual_level
dual_level
torch.autograd.forward_ad.make_dual
make_dual()
torch.autograd.forward_ad.unpack_dual
unpack_dual()
torch.autograd.forward_ad.enter_dual_level
enter_dual_level()
torch.autograd.forward_ad.exit_dual_level
exit_dual_level()
UnpackedDualTensor
UnpackedDualTensor
UnpackedDualTensor.count()
UnpackedDualTensor.index()
UnpackedDualTensor.primal
UnpackedDualTensor.tangent
torch.autograd.functional.jacobian
jacobian()
torch.autograd.functional.hessian
hessian()
torch.autograd.functional.vjp
vjp()
torch.autograd.functional.jvp
jvp()
torch.autograd.functional.vhp
vhp()
torch.autograd.functional.hvp
hvp()

BackwardCFunction
BackwardCFunction
BackwardCFunction.apply()
BackwardCFunction.apply_jvp()
BackwardCFunction.mark_dirty()
BackwardCFunction.mark_non_differentiable()
BackwardCFunction.save_for_backward()
BackwardCFunction.save_for_forward()
BackwardCFunction.set_materialize_grads()
InplaceFunction
InplaceFunction
InplaceFunction.backward()
InplaceFunction.forward()
InplaceFunction.jvp()
InplaceFunction.mark_dirty()
InplaceFunction.mark_non_differentiable()
InplaceFunction.save_for_backward()
InplaceFunction.save_for_forward()
InplaceFunction.set_materialize_grads()
InplaceFunction.setup_context()
InplaceFunction.vjp()
InplaceFunction.vmap()
NestedIOFunction
NestedIOFunction
NestedIOFunction.backward()
NestedIOFunction.backward_extended()
NestedIOFunction.forward()
NestedIOFunction.forward_extended()
NestedIOFunction.jvp()
NestedIOFunction.mark_dirty()
NestedIOFunction.mark_non_differentiable()
NestedIOFunction.save_for_backward()
NestedIOFunction.save_for_forward()
NestedIOFunction.saved_tensors
NestedIOFunction.set_materialize_grads()
NestedIOFunction.setup_context()
NestedIOFunction.vjp()
NestedIOFunction.vmap()
torch.autograd.gradcheck.gradcheck
gradcheck()
torch.autograd.gradcheck.gradgradcheck
gradgradcheck()
torch.autograd.gradcheck.GradcheckError
GradcheckError

torch.autograd.profiler.profile.export_chrome_trace
profile.export_chrome_trace()
torch.autograd.profiler.profile.key_averages
profile.key_averages()
torch.autograd.profiler.profile.self_cpu_time_total
profile.self_cpu_time_total
torch.autograd.profiler.profile.total_average
profile.total_average()
torch.autograd.profiler.parse_nvprof_trace
parse_nvprof_trace()
EnforceUnique
EnforceUnique
EnforceUnique.see()
KinetoStepTracker
KinetoStepTracker
KinetoStepTracker.current_step()
KinetoStepTracker.erase_step_count()
KinetoStepTracker.increment_step()
KinetoStepTracker.init_step_count()
record_function
record_function
Interval
Interval
Interval.elapsed_us()
Kernel
Kernel
Kernel.count()
Kernel.device
Kernel.duration
Kernel.index()
Kernel.name
MemRecordsAcc
MemRecordsAcc
MemRecordsAcc.in_interval()
StringTable
StringTable
StringTable.clear()
StringTable.copy()
StringTable.default_factory
StringTable.fromkeys()
StringTable.get()
StringTable.items()
StringTable.keys()
StringTable.pop()
StringTable.popitem()
StringTable.setdefault()
StringTable.update()
StringTable.values()
torch.autograd.profiler.load_nvprof
load_nvprof()

set_multithreading_enabled
set_multithreading_enabled
set_multithreading_enabled.clone()
torch.autograd.graph.Node.name
Node.name()
torch.autograd.graph.Node.metadata
Node.metadata()
torch.autograd.graph.Node.next_functions
Node.next_functions
torch.autograd.graph.increment_version
increment_version()


torch.accelerator.device_count
device_count()
torch.accelerator.set_device_index
set_device_index()
torch.accelerator.set_device_idx
set_device_idx()
torch.accelerator.current_device_index
current_device_index()
torch.accelerator.current_device_idx
current_device_idx()
torch.accelerator.set_stream
set_stream()
torch.accelerator.current_stream
current_stream()
torch.accelerator.synchronize
synchronize()
torch.cpu.current_device
current_device()
torch.cpu.current_stream
current_stream()
Stream
Stream
torch.cpu.is_available
is_available()
torch.cpu.synchronize
synchronize()
torch.cpu.stream
stream()
torch.cpu.set_device
set_device()
torch.cpu.device_count
device_count()
StreamContext
StreamContext
StreamContext
StreamContext
torch.cuda.can_device_access_peer
can_device_access_peer()
torch.cuda.current_blas_handle
current_blas_handle()
torch.cuda.current_stream
current_stream()
torch.cuda.cudart
cudart()
torch.cuda.default_stream
default_stream()
torch.cuda.device_count
device_count()
torch.cuda.device_memory_used
device_memory_used()
device_of
device_of
torch.cuda.get_arch_list
get_arch_list()
torch.cuda.get_device_capability
get_device_capability()
torch.cuda.get_device_name
get_device_name()
torch.cuda.get_device_properties
get_device_properties()
torch.cuda.get_gencode_flags
get_gencode_flags()
torch.cuda.get_stream_from_external
get_stream_from_external()
torch.cuda.get_sync_debug_mode
get_sync_debug_mode()
torch.cuda.init
init()
torch.cuda.ipc_collect
ipc_collect()
torch.cuda.is_initialized
is_initialized()
torch.cuda.is_tf32_supported
is_tf32_supported()
torch.cuda.memory_usage
memory_usage()
torch.cuda.set_stream
set_stream()
torch.cuda.set_sync_debug_mode
set_sync_debug_mode()
torch.cuda.stream
stream()
torch.cuda.utilization
utilization()
torch.cuda.temperature
temperature()
torch.cuda.power_draw
power_draw()
torch.cuda.clock_rate
clock_rate()
torch.cuda.OutOfMemoryError
OutOfMemoryError
torch.cuda.get_rng_state
get_rng_state()
torch.cuda.get_rng_state_all
get_rng_state_all()
torch.cuda.set_rng_state
set_rng_state()
torch.cuda.set_rng_state_all
set_rng_state_all()
torch.cuda.manual_seed
manual_seed()
torch.cuda.manual_seed_all
manual_seed_all()
torch.cuda.seed
seed()
torch.cuda.seed_all
seed_all()
torch.cuda.initial_seed
initial_seed()
torch.cuda.comm.broadcast
broadcast()
torch.cuda.comm.broadcast_coalesced
broadcast_coalesced()
torch.cuda.comm.reduce_add
reduce_add()
torch.cuda.comm.scatter
scatter()
torch.cuda.comm.gather
gather()
ExternalStream
ExternalStream
ExternalStream.query()
ExternalStream.record_event()
ExternalStream.synchronize()
ExternalStream.wait_event()
ExternalStream.wait_stream()
torch.cuda.is_current_stream_capturing
is_current_stream_capturing()
torch.cuda.graph_pool_handle
graph_pool_handle()
torch.cuda.get_per_process_memory_fraction
get_per_process_memory_fraction()
torch.cuda.list_gpu_processes
list_gpu_processes()
torch.cuda.mem_get_info
mem_get_info()
torch.cuda.host_memory_stats
host_memory_stats()
torch.cuda.reset_max_memory_allocated
reset_max_memory_allocated()
torch.cuda.set_per_process_memory_fraction
set_per_process_memory_fraction()
torch.cuda.memory_cached
memory_cached()
torch.cuda.max_memory_cached
max_memory_cached()
torch.cuda.reset_max_memory_cached
reset_max_memory_cached()
torch.cuda.reset_peak_memory_stats
reset_peak_memory_stats()
torch.cuda.reset_peak_host_memory_stats
reset_peak_host_memory_stats()
torch.cuda.caching_allocator_alloc
caching_allocator_alloc()
torch.cuda.caching_allocator_delete
caching_allocator_delete()
torch.cuda.get_allocator_backend
get_allocator_backend()
MemPoolContext
MemPoolContext
MemPoolContext.active_pool()
torch.cuda.memory.caching_allocator_enable
caching_allocator_enable()

torch.cuda.nvtx.mark
mark()
torch.cuda.nvtx.range_push
range_push()
torch.cuda.nvtx.range_pop
range_pop()
torch.cuda.nvtx.range
range()
torch.cuda.jiterator._create_jit_fn
_create_jit_fn()
torch.cuda.jiterator._create_multi_output_jit_fn
_create_multi_output_jit_fn()
TunableOp
Overview
Enabling TunableOp and Tuning Separately
File Input and Output
A Note on Tuning Behavior, Warmup, and Cache Effects
Current Tunable Operators
TunableGemm for ROCm


Offline Tuning
Motivation
Workflow


Tuning Context
Environment Variable Interface




API Reference
enable()
is_enabled()
tuning_enable()
tuning_is_enabled()
record_untuned_enable()
record_untuned_is_enabled()
set_max_tuning_duration()
get_max_tuning_duration()
set_max_tuning_iterations()
get_max_tuning_iterations()
set_filename()
get_filename()
get_results()
get_validators()
write_file_on_exit()
write_file()
read_file()
tune_gemm_in_file()
mgpu_tune_gemm_in_file()
set_rotating_buffer_size()
get_rotating_buffer_size()
CUDA Stream Sanitizer
Overview
Usage
API Reference
enable_cuda_sanitizer()
GdsFile
GdsFile
GdsFile.deregister_handle()
GdsFile.load_storage()
GdsFile.register_handle()
GdsFile.save_storage()
torch.cuda.gds.gds_register_buffer
gds_register_buffer()
torch.cuda.gds.gds_deregister_buffer
gds_deregister_buffer()
torch.mps.device_count
device_count()
torch.mps.synchronize
synchronize()
torch.mps.get_rng_state
get_rng_state()
torch.mps.set_rng_state
set_rng_state()
torch.mps.manual_seed
manual_seed()
torch.mps.seed
seed()
torch.mps.empty_cache
empty_cache()
torch.mps.set_per_process_memory_fraction
set_per_process_memory_fraction()
torch.mps.current_allocated_memory
current_allocated_memory()
torch.mps.driver_allocated_memory
driver_allocated_memory()
torch.mps.recommended_max_memory
recommended_max_memory()
torch.mps.compile_shader
compile_shader()
torch.mps.profiler.start
start()
torch.mps.profiler.stop
stop()
torch.mps.profiler.profile
profile()
torch.mps.profiler.is_capturing_metal
is_capturing_metal()
torch.mps.profiler.is_metal_capture_enabled
is_metal_capture_enabled()
torch.mps.profiler.metal_capture
metal_capture()
Event
Event
Event.elapsed_time()
Event.query()
Event.record()
Event.synchronize()
Event.wait()
torch.xpu.is_available
is_available()
StreamContext
StreamContext
torch.xpu.current_device
current_device()
torch.xpu.current_stream
current_stream()
Stream
Stream
Stream.query()
Stream.record_event()
Stream.synchronize()
Stream.wait_event()
Stream.wait_stream()
device
device
torch.xpu.device_count
device_count()
device_of
device_of
torch.xpu.get_arch_list
get_arch_list()
torch.xpu.get_device_capability
get_device_capability()
torch.xpu.get_device_name
get_device_name()
torch.xpu.get_device_properties
get_device_properties()
torch.xpu.get_gencode_flags
get_gencode_flags()
torch.xpu.get_stream_from_external
get_stream_from_external()
torch.xpu.init
init()
torch.xpu.is_initialized
is_initialized()
torch.xpu.set_device
set_device()
torch.xpu.set_stream
set_stream()
torch.xpu.stream
stream()
torch.xpu.synchronize
synchronize()
torch.xpu.get_rng_state
get_rng_state()
torch.xpu.get_rng_state_all
get_rng_state_all()
torch.xpu.initial_seed
initial_seed()
torch.xpu.manual_seed
manual_seed()
torch.xpu.manual_seed_all
manual_seed_all()
torch.xpu.seed
seed()
torch.xpu.seed_all
seed_all()
torch.xpu.set_rng_state
set_rng_state()
torch.xpu.set_rng_state_all
set_rng_state_all()
Event
Event
Event.elapsed_time()
Event.query()
Event.record()
Event.synchronize()
Event.wait()
torch.xpu.empty_cache
empty_cache()
torch.xpu.max_memory_allocated
max_memory_allocated()
torch.xpu.max_memory_reserved
max_memory_reserved()
torch.xpu.mem_get_info
mem_get_info()
torch.xpu.memory_allocated
memory_allocated()
torch.xpu.memory_reserved
memory_reserved()
torch.xpu.memory_stats
memory_stats()
torch.xpu.memory_stats_as_nested_dict
memory_stats_as_nested_dict()
torch.xpu.reset_accumulated_memory_stats
reset_accumulated_memory_stats()
torch.xpu.reset_peak_memory_stats
reset_peak_memory_stats()
StreamContext
StreamContext
torch.mtia.current_device
current_device()
torch.mtia.current_stream
current_stream()
Stream
Stream
Stream.query()
Stream.record_event()
Stream.synchronize()
Stream.wait_event()
Stream.wait_stream()
torch.mtia.default_stream
default_stream()
torch.mtia.device_count
device_count()
torch.mtia.init
init()
torch.mtia.is_available
is_available()
torch.mtia.is_initialized
is_initialized()
torch.mtia.memory_stats
memory_stats()
torch.mtia.get_device_capability
get_device_capability()
torch.mtia.empty_cache
empty_cache()
torch.mtia.record_memory_history
record_memory_history()
torch.mtia.snapshot
snapshot()
torch.mtia.set_device
set_device()
torch.mtia.set_stream
set_stream()
torch.mtia.stream
stream()
torch.mtia.synchronize
synchronize()
device
device
torch.mtia.set_rng_state
set_rng_state()
torch.mtia.get_rng_state
get_rng_state()
torch.mtia.DeferredMtiaCallError
DeferredMtiaCallError
Event
Event
Event.elapsed_time()
Event.query()
Event.record()
Event.synchronize()
Event.wait()
torch.mtia.memory.memory_stats
memory_stats()


torch.linalg.inv_ex
inv_ex()
torch.linalg.cholesky
cholesky()
torch.linalg.cholesky_ex
cholesky_ex()
torch.linalg.lu
lu()
torch.linalg.lu_solve
lu_solve()
torch.linalg.qr
qr()
torch.linalg.eigh
eigh()
torch.linalg.svd
svd()









torch.export IR Specification
Assumptions
What is Export IR
ExportedProgram
Graph
Node
call_function
Metadata


placeholder
output
get_attr


References
SymInt
FakeTensor
Pytree-able Types
torch.jit.script
script()
torch.jit.trace
trace()
ExportDB
Supported
assume_constant_result
autograd_function
class_method
cond_branch_class_method
cond_branch_nested_function
cond_branch_nonlocal_variables
cond_closed_over_variable
cond_operands
cond_predicate
constrain_as_size_example
constrain_as_value_example
decorator
dictionary
dynamic_shape_assert
dynamic_shape_constructor
dynamic_shape_if_guard
dynamic_shape_map
dynamic_shape_slicing
dynamic_shape_view
fn_with_kwargs
list_contains
list_unpack
nested_function
null_context_manager
pytree_flatten
scalar_output
specialized_attribute
static_for_loop
static_if
tensor_setattr
type_reflection_method
user_input_mutation


Not Supported Yet
dynamic_shape_round
model_attr_mutation
optional_input
unsupported_operator
Control Flow - Cond
Examples
Invariants of torch.ops.higher_order.cond
API Reference
cond()
torch.export Programming Model
Basics of Tracing
Strict vs. Non-Strict Tracing


Values: Static vs. Dynamic
Static Values
Dynamic Values
Which values are static vs. dynamic?


Input types
Custom Input Types
Optional input types


Control Flow: Static vs. Dynamic
Static Control Flow
Dynamic Control Flow: Shape-Dependent vs. Data-Dependent
Dynamic Shape-Dependent Control Flow
Dynamic Data-Dependent Control Flow




Basics of Symbolic Shapes
Fake Implementations of PyTorch Operators
Shape Propagation: Backed vs. Unbacked Dynamic Shapes


Control Flow: Guards and Assertions


Allowed PyTorch operators
Custom operators


Module State: Reads vs. Updates
Access rules
Effects of functionalization
Writing Graph Transformations on ATen IR
Passes
Transformer
One-to-One Pass
One-to-X Pass
One-to-None Pass
Utilizing Local Information


Subgraph Rewriter


Pass Manager
Partitioner
Subgraph Matcher
Capability Based Partitioner
IRs
Core Aten IR
Prims IR
Dynamo Overview
Dynamo Internals
What is a guard?
What is Dynamo doing?
How to inspect artifacts generated by Dynamo?
Dynamo Deep-Dive
A Gentle Introduction to Dynamo
PEP 523: Adding a frame evaluation API to CPython
Implementing CPython in Python
Generating the Output Graph
Making Dynamo Sound: Guards
Symbolic Shapes
Static by default
0, 1 are always specialized
Duck shaping
Guards on symbolic ints


Making Dynamo Complete: Graph Breaks
Conclusion
Footnotes
Dynamic shapes
Motivation
Abridged public API
The Guard Model
Overall architecture
Abridged internal API
DimDynamic policy
Unbacked SymInts
Fake tensor
Motivation
Related work
Overall architecture
API: the important bits
Details
About the tensor subclass
How is each individual operator implemented?
How does the converter work?
Performance characteristics
Fake tensor of fake tensor?
Interaction with dynamic shapes
Other resources










torchrun (Elastic Launch)
Usage
Single-node multi-worker
Stacked single-node multi-worker
Fault tolerant (fixed sized number of workers, no elasticity, tolerates 3 failures)
Elastic (min=1, max=4, tolerates up to 3 membership changes or failures)


Note on rendezvous backend
Definitions
Environment Variables
Deployment
Failure Modes
Membership Changes
Important Notices


Quickstart
Train script
Examples
Elastic Agent
Server
Concepts
ElasticAgent
ElasticAgent.get_worker_group()
ElasticAgent.run()


WorkerSpec
WorkerSpec.get_entrypoint_name()


WorkerState
WorkerState.is_running()


Worker
WorkerGroup


Implementations
LocalElasticAgent


Extending the Agent
SimpleElasticAgent
SimpleElasticAgent._assign_worker_ranks()
SimpleElasticAgent._exit_barrier()
SimpleElasticAgent._initialize_workers()
SimpleElasticAgent._monitor_workers()
SimpleElasticAgent._rendezvous()
SimpleElasticAgent._restart_workers()
SimpleElasticAgent._shutdown()
SimpleElasticAgent._start_workers()
SimpleElasticAgent._stop_workers()


RunResult


Watchdog in the Agent
Health Check Server
HealthCheckServer
HealthCheckServer.start()
HealthCheckServer.stop()


create_healthcheck_server()
Multiprocessing
Starting Multiple Workers
start_processes()


Process Context
PContext
MultiprocessContext
SubprocessContext
RunProcsResult
DefaultLogsSpecs
DefaultLogsSpecs.reify()


LogsDest
LogsSpecs
LogsSpecs.reify()
Error Propagation
Methods and Classes
record()
ChildFailedError
ErrorHandler
ProcessFailure
Rendezvous
Registry
RendezvousParameters
RendezvousParameters.get()
RendezvousParameters.get_as_bool()
RendezvousParameters.get_as_int()


RendezvousHandlerRegistry


Handler
RendezvousHandler
RendezvousHandler.get_backend()
RendezvousHandler.get_run_id()
RendezvousHandler.is_closed()
RendezvousHandler.next_rendezvous()
RendezvousHandler.num_nodes_waiting()
RendezvousHandler.set_closed()
RendezvousHandler.shutdown()
RendezvousHandler.use_agent_store




Dataclasses
RendezvousInfo
RendezvousStoreInfo
RendezvousStoreInfo.build()




Exceptions
RendezvousError
RendezvousClosedError
RendezvousTimeoutError
RendezvousConnectionError
RendezvousStateError
RendezvousGracefulExitError


Implementations
Dynamic Rendezvous
create_handler()
DynamicRendezvousHandler
DynamicRendezvousHandler.from_backend()


RendezvousBackend
RendezvousBackend.get_state()
RendezvousBackend.name
RendezvousBackend.set_state()


RendezvousTimeout
RendezvousTimeout.close
RendezvousTimeout.heartbeat
RendezvousTimeout.join
RendezvousTimeout.last_call


C10d Backend
create_backend()
C10dRendezvousBackend
C10dRendezvousBackend.get_state()
C10dRendezvousBackend.name
C10dRendezvousBackend.set_state()




Etcd Backend
create_backend()
EtcdRendezvousBackend
EtcdRendezvousBackend.get_state()
EtcdRendezvousBackend.name
EtcdRendezvousBackend.set_state()






Etcd Rendezvous (Legacy)
EtcdRendezvousHandler


Etcd Store
EtcdStore
EtcdStore.add()
EtcdStore.check()
EtcdStore.get()
EtcdStore.set()
EtcdStore.wait()




Etcd Server
EtcdServer
Expiration Timers
Client Methods
configure()
expires()


Server/Client Implementations
LocalTimerServer
LocalTimerClient
FileTimerServer
FileTimerClient


Writing a custom timer server/client
TimerRequest
TimerServer
TimerServer.clear_timers()
TimerServer.get_expired_timers()
TimerServer.register_timers()


TimerClient
TimerClient.acquire()
TimerClient.release()




Debug info logging
log_debug_info_for_expired_timers()
Metrics
Metric Handlers
MetricHandler
ConsoleMetricHandler
NullMetricHandler


Methods
configure()
prof()
put_metric()
Events
API Methods
record()
construct_and_record_rdzv_event()
get_logging_handler()


Event Objects
Event
EventSource
EventMetadataValue
Subprocess Handling
Retrieve SubprocessHandler
get_subprocess_handler()


SubprocessHandler
SubprocessHandler
Control Plane
worker_main()
Customization
Launcher
Rendezvous Handler
Metric Handler
Events Handler
TorchElastic Kubernetes

torch.optim.Optimizer.state_dict
Optimizer.state_dict()





torch.optim.Optimizer.step
Optimizer.step()



























































ONNX Backend for TorchDynamo
is_onnxrt_backend_supported()
Getting Started
Using a pretrained model
Next Steps
torch.compiler API reference
torch.compiler.config
job_id
TorchDynamo APIs for fine-grained tracing
torch.compiler.disable
torch._dynamo.disallow_in_graph
torch.compiler.allow_in_graph
Limitations
AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models
Model Compilation
Inference in Python
Inference in C++
Troubleshooting
API Reference
aoti_compile_and_package()
aoti_load_package()
TorchInductor GPU Profiling
Relevant Environment Variables
Breakdown Model GPU Time
Benchmark Individual Triton Kernel
Profiling to understand torch.compile performance
What to use torch.profiler for:
Basics of using torch.profiler and viewing traces
Working around CUDA Graph profiling issues
Understanding compilation time
Finding graph breaks: “Torch-Compiled Region” and “CompiledFunction”
Operator Kernels
Launch overhead
Frequently Asked Questions
Does torch.compile support training?
Do you support Distributed code?
Do I still need to export whole graphs?
Why is my code crashing?
Why is compilation slow?
Why are you recompiling in production?
How are you speeding up my code?
Why am I not seeing speedups?
Graph Breaks
Identifying the cause of a graph break
Why didn’t my code recompile when I changed it?


Why am I getting incorrect results?
Why am I getting OOMs?
Does torch.func work with torch.compile (for grad and vmap transforms)?
Calling torch.func transform inside of a function handled with torch.compile
Compiling torch.func.grad with torch.compile
Compiling torch.vmap with torch.compile
Compiling functions besides the ones which are supported (escape hatch)


Does NumPy work with torch.compile?
Which NumPy features does torch.compile support?
Can I compile NumPy code using torch.compile?
Can I execute NumPy code on CUDA and compute gradients via torch.compile?
How do I debug NumPy code under torch.compile?
I torch.compile some NumPy code and I did not see any speed-up.


Which API to use for fine grain tracing?
How do I graph break on a function?
What’s the difference between torch._dynamo.disable and torch._dynamo.disallow_in_graph
What’s the difference between torch._dynamo.disable and torch._dynamo_skip
torch.compile Troubleshooting
Setting Expectations
Compile times


Terminology
Graph break
Guards
Recompilation
Dynamic Shapes


Logging Tools
tlparse / TORCH_TRACE
TORCH_LOGS
tlparse vs. TORCH_LOGS


Simple Workarounds
Where to apply torch.compile?
Disabling and Suppressing Errors
Resolving graph breaks
Data-dependent operations
Custom ops
Printing
Incorrect code


Dealing with recompilations
Is dynamic shapes enabled?
Changing the cache size limit
Wrapping constants with tensors




Reporting Issues
Ablation
Bisecting
Creating a reproducer
Minifier


Debugging Deeper
TorchDynamo
Logging what Dynamo is tracing
Breakpointing Dynamo tracing
Bytecode generation errors


AOTAutograd
Summary of TORCH_LOGS options


Related Articles
PyTorch 2.0 Performance Dashboard
How to read the dashboard?
What is measured on the dashboard?
Can I check if my PR affects TorchInductor’s performance on the dashboard before merging?
How can I run any performance test locally?
PyTorch 2.0 NNModule Support
NNModule Hooks Support
nn.Module.__call__ Hooks Usage and limitations
state_dict Hooks
Best Practices for Backends
x86 CPU
CUDAGraph Trees
Background
CUDAGraph
PyTorch CUDAGraph Integration
Make Graphed Callables
TorchDynamo Previous CUDA Graphs Integration


CUDAGraph Trees Integration
Input Mutation Support
Dynamic Shape Support
NCCL Support
Reasons for Skipping CUDAGraph
Limitations
Comparisons
Custom Backends
Overview
Registering Custom Backends
Custom Backends after AOTAutograd
Examples
Debugging Backend
Speedy Backend
Composable Backends
torch.fft.ifft
ifft()
torch.fft.fft2
fft2()
torch.fft.ifft2
ifft2()
torch.fft.fftn
fftn()
torch.fft.ifftn
ifftn()
torch.fft.rfft
rfft()
torch.fft.irfft
irfft()
torch.fft.rfft2
rfft2()
torch.fft.irfft2
irfft2()
torch.fft.rfftn
rfftn()
torch.fft.irfftn
irfftn()
torch.fft.hfft
hfft()
torch.fft.ihfft
ihfft()
torch.fft.hfft2
hfft2()
torch.fft.ihfft2
ihfft2()
torch.fft.hfftn
hfftn()
torch.fft.ihfftn
ihfftn()
torch.fft.fftfreq
fftfreq()
torch.fft.rfftfreq
rfftfreq()
torch.fft.fftshift
fftshift()
torch.fft.ifftshift
ifftshift()
torch.func.vmap
vmap()
torch.func Whirlwind Tour
What is torch.func?
Why composable function transforms?
What are the transforms?
grad() (gradient computation)
vmap() (auto-vectorization)
vjp() (vector-Jacobian product)
jvp() (Jacobian-vector product)
jacrev(), jacfwd(), and hessian()
UX Limitations
General limitations
torch.autograd APIs
vmap limitations
Mutation: Arbitrary mutation of Python data structures
Mutation: in-place PyTorch Operations
Mutation: out= PyTorch Operations
Data-dependent Python control flow
Data-dependent operations (.item())
Dynamic shape operations (nonzero and friends)


Randomness
Migrating from functorch to torch.func
function transforms
NN module utilities
functorch.make_functional
functorch.combine_state_for_ensemble


functorch.compile








ShapeEnv
ShapeEnv
ShapeEnv.add_var_to_val()
ShapeEnv.bind_symbols()
ShapeEnv.bound_sympy()
ShapeEnv.check_equal()
ShapeEnv.cleanup()
ShapeEnv.create_symbol()
ShapeEnv.create_symbolic_sizes_strides_storage_offset()
ShapeEnv.create_symboolnode()
ShapeEnv.create_symfloatnode()
ShapeEnv.create_symintnode()
ShapeEnv.create_unbacked_symbool()
ShapeEnv.create_unbacked_symfloat()
ShapeEnv.create_unbacked_symint()
ShapeEnv.create_unspecified_symbol()
ShapeEnv.create_unspecified_symint_and_symbol()
ShapeEnv.defer_runtime_assert()
ShapeEnv.deserialize_symexpr()
ShapeEnv.evaluate_guards_expression()
ShapeEnv.evaluate_guards_for_args()
ShapeEnv.evaluate_sym_node()
ShapeEnv.evaluate_symexpr()
ShapeEnv.format_guards()
ShapeEnv.freeze()
ShapeEnv.freeze_runtime_asserts()
ShapeEnv.get_axioms()
ShapeEnv.get_implications()
ShapeEnv.get_nontrivial_guards()
ShapeEnv.get_pruned_guards()
ShapeEnv.ignore_fresh_unbacked_symbols()
ShapeEnv.is_unbacked_symint()
ShapeEnv.produce_guards()
ShapeEnv.produce_guards_expression()
ShapeEnv.produce_guards_verbose()
ShapeEnv.replace()
ShapeEnv.set_unbacked_var_to_val()
ShapeEnv.simplify()
ShapeEnv.size_hint()
ShapeEnv.suppress_guards()
DimDynamic
DimDynamic
StrictMinMaxConstraint
StrictMinMaxConstraint
StrictMinMaxConstraint.render()
RelaxedUnspecConstraint
RelaxedUnspecConstraint
EqualityConstraint
EqualityConstraint
SymbolicContext
SymbolicContext
StatelessSymbolicContext
StatelessSymbolicContext
StatefulSymbolicContext
StatefulSymbolicContext
SubclassSymbolicContext
SubclassSymbolicContext
DimConstraints
DimConstraints
DimConstraints.add()
DimConstraints.add_equality()
DimConstraints.forced_specializations()
DimConstraints.prettify_results()
DimConstraints.rewrite_with_congruences()
DimConstraints.solve()
ShapeEnvSettings
ShapeEnvSettings
ConvertIntKey
ConvertIntKey
ConvertIntKey.get()
CallMethodKey
CallMethodKey
CallMethodKey.get()
PropagateUnbackedSymInts
PropagateUnbackedSymInts
PropagateUnbackedSymInts.boxed_run()
PropagateUnbackedSymInts.call_function()
PropagateUnbackedSymInts.call_method()
PropagateUnbackedSymInts.call_module()
PropagateUnbackedSymInts.fetch_args_kwargs_from_env()
PropagateUnbackedSymInts.fetch_attr()
PropagateUnbackedSymInts.get_attr()
PropagateUnbackedSymInts.map_nodes_to_values()
PropagateUnbackedSymInts.output()
PropagateUnbackedSymInts.placeholder()
PropagateUnbackedSymInts.run()
PropagateUnbackedSymInts.run_node()
DivideByKey
DivideByKey
DivideByKey.get()
InnerTensorKey
InnerTensorKey
InnerTensorKey.get()
torch.fx.experimental.symbolic_shapes.hint_int
hint_int()
torch.fx.experimental.symbolic_shapes.is_concrete_int
is_concrete_int()
torch.fx.experimental.symbolic_shapes.is_concrete_bool
is_concrete_bool()
torch.fx.experimental.symbolic_shapes.is_concrete_float
is_concrete_float()
torch.fx.experimental.symbolic_shapes.has_free_symbols
has_free_symbols()
torch.fx.experimental.symbolic_shapes.has_free_unbacked_symbols
has_free_unbacked_symbols()
torch.fx.experimental.symbolic_shapes.definitely_true
definitely_true()
torch.fx.experimental.symbolic_shapes.definitely_false
definitely_false()
torch.fx.experimental.symbolic_shapes.guard_size_oblivious
guard_size_oblivious()
torch.fx.experimental.symbolic_shapes.sym_eq
sym_eq()
torch.fx.experimental.symbolic_shapes.constrain_range
constrain_range()
torch.fx.experimental.symbolic_shapes.constrain_unify
constrain_unify()
torch.fx.experimental.symbolic_shapes.canonicalize_bool_expr
canonicalize_bool_expr()
torch.fx.experimental.symbolic_shapes.statically_known_true
statically_known_true()
torch.fx.experimental.symbolic_shapes.lru_cache
lru_cache()
torch.fx.experimental.symbolic_shapes.check_consistent
check_consistent()
torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings
compute_unbacked_bindings()
torch.fx.experimental.symbolic_shapes.rebind_unbacked
rebind_unbacked()
torch.fx.experimental.symbolic_shapes.resolve_unbacked_bindings
resolve_unbacked_bindings()
torch.fx.experimental.symbolic_shapes.is_accessor_node
is_accessor_node()
torch.fx.experimental.proxy_tensor.make_fx
make_fx()
torch.fx.experimental.proxy_tensor.handle_sym_dispatch
handle_sym_dispatch()
torch.fx.experimental.proxy_tensor.get_proxy_mode
get_proxy_mode()
torch.fx.experimental.proxy_tensor.maybe_enable_thunkify
maybe_enable_thunkify()
torch.fx.experimental.proxy_tensor.maybe_disable_thunkify
maybe_disable_thunkify()

TorchScript Language Reference
Terminology
Type System
TorchScript Types
Meta Types
Any Type
Operators Supported for Any Type
Design Notes


Primitive Types
Structural Types
Nominal Types
Built-in Class
Special Note on torch.nn.ModuleList and torch.nn.ModuleDict


Custom Class
Enum Type
TorchScript Module Class
Module Instance Class


Type Annotation
When to Annotate Types
Annotate Function Signature
Annotate Variables and Data Attributes
Local Variables
Instance Data Attributes


Type Annotation APIs
torch.jit.annotate(T, expr)


Type Annotation Appendix
TorchScript Type System Definition
Unsupported Typing Constructs




Expressions
Arithmetic Conversions
Atoms
Identifiers
Literals
Parenthesized Forms
List and Dictionary Displays


Primaries
Attribute References
Subscriptions
Slicings
Calls


Power Operator
Unary and Arithmetic Bitwise Operations
Binary Arithmetic Operations
Shifting Operations
Binary Bitwise Operations
Comparisons
Value Comparisons
Membership Test Operations
Identity Comparisons


Boolean Operations
Conditional Expressions
Expression Lists


Simple Statements
Expression Statements
Assignment Statements
Augmented Assignment Statements
Annotated Assignment Statements
The raise Statement
The assert Statement
The return Statement
The del Statement
The pass Statement
The print Statement
The break Statement
The continue Statement:


Compound Statements
The if Statement
Basic if/else Statement
Ternary if/else Statement


The while Statement
The for-in Statement
The with Statement
The tuple Statement
The getattr Statement
The hasattr Statement
The zip Statement
The enumerate Statement


Python Values
Resolution Rules
Python Built-in Functions Support
Python Built-in Values Support


torch.* APIs
Remote Procedure Calls
Asynchronous Execution
Type Annotations
Meta Programming
Type Refinement
ScriptFunction
ScriptFunction
ScriptFunction.get_debug_state()
ScriptFunction.save()
ScriptFunction.save_to_buffer()
torch.jit.script_if_tracing
script_if_tracing()
torch.jit.trace_module
trace_module()
ScriptModule
ScriptModule
ScriptModule.add_module()
ScriptModule.apply()
ScriptModule.bfloat16()
ScriptModule.buffers()
ScriptModule.children()
ScriptModule.code
ScriptModule.code_with_constants
ScriptModule.compile()
ScriptModule.cpu()
ScriptModule.cuda()
ScriptModule.double()
ScriptModule.eval()
ScriptModule.extra_repr()
ScriptModule.float()
ScriptModule.get_buffer()
ScriptModule.get_extra_state()
ScriptModule.get_parameter()
ScriptModule.get_submodule()
ScriptModule.graph
ScriptModule.half()
ScriptModule.inlined_graph
ScriptModule.ipu()
ScriptModule.load_state_dict()
ScriptModule.modules()
ScriptModule.mtia()
ScriptModule.named_buffers()
ScriptModule.named_children()
ScriptModule.named_modules()
ScriptModule.named_parameters()
ScriptModule.parameters()
ScriptModule.register_backward_hook()
ScriptModule.register_buffer()
ScriptModule.register_forward_hook()
ScriptModule.register_forward_pre_hook()
ScriptModule.register_full_backward_hook()
ScriptModule.register_full_backward_pre_hook()
ScriptModule.register_load_state_dict_post_hook()
ScriptModule.register_load_state_dict_pre_hook()
ScriptModule.register_module()
ScriptModule.register_parameter()
ScriptModule.register_state_dict_post_hook()
ScriptModule.register_state_dict_pre_hook()
ScriptModule.requires_grad_()
ScriptModule.save()
ScriptModule.set_extra_state()
ScriptModule.set_submodule()
ScriptModule.share_memory()
ScriptModule.state_dict()
ScriptModule.to()
ScriptModule.to_empty()
ScriptModule.train()
ScriptModule.type()
ScriptModule.xpu()
ScriptModule.zero_grad()
torch.jit.fork
fork()
torch.jit.wait
wait()
torch.jit.freeze
freeze()
torch.jit.optimize_for_inference
optimize_for_inference()
torch.jit.enable_onednn_fusion
enable_onednn_fusion()
torch.jit.onednn_fusion_enabled
onednn_fusion_enabled()
torch.jit.set_fusion_strategy
set_fusion_strategy()
strict_fusion
strict_fusion
torch.jit.ignore
ignore()
torch.jit.unused
unused()
torch.jit.interface
interface()
torch.jit.isinstance
isinstance()
Attribute
Attribute
Attribute.count()
Attribute.index()
Attribute.type
Attribute.value
torch.jit.annotate
annotate()
TorchScript Language Reference
Types
Unsupported Typing Constructs
Default Types
Optional Type Refinement
TorchScript Classes
TorchScript Enums
Named Tuples
Iterables


Expressions
Literals
List Construction
Tuple Construction
Dict Construction


Variables
Arithmetic Operators
Comparison Operators
Logical Operators
Subscripts and Slicing
Function Calls
Method Calls
Ternary Expressions
Casts
Accessing Module Parameters


Statements
Simple Assignments
Pattern Matching Assignments
Print Statements
If Statements
While Loops
For loops with range
For loops over tuples
For loops over constant nn.ModuleList
Break and Continue
Return


Variable Resolution
Use of Python Values
Functions
is_scripting()
is_tracing()


Attribute Lookup On Python Modules
Python-defined Constants
Module Attributes
TorchScript Builtins
Supported Tensor Methods
Supported PyTorch Functions
TorchScript Builtin Functions
Python Built-in Functions
math Module
TorchScript Unsupported PyTorch Constructs
Torch and Tensor Unsupported Attributes
Unsupported Tensor Methods
Unsupported Tensor Properties
Functions Not Correctly Bound on Torch
Ops With Divergent Schemas Between Torch & Python


PyTorch Unsupported Modules and Classes
Python Language Reference Coverage

torch.linalg.norm
norm()
torch.linalg.vector_norm
vector_norm()
torch.linalg.matrix_norm
matrix_norm()
torch.linalg.diagonal
diagonal()
torch.linalg.matrix_rank
matrix_rank()
torch.linalg.eig
eig()
torch.linalg.eigvals
eigvals()
torch.linalg.eigvalsh
eigvalsh()
torch.linalg.solve
solve()
torch.linalg.solve_triangular
solve_triangular()
torch.linalg.lstsq
lstsq()
torch.linalg.cross
cross()
torch.linalg.matmul
matmul()
torch.linalg.vecdot
vecdot()
torch.linalg.multi_dot
multi_dot()
torch.linalg.tensorinv
tensorinv()
torch.linalg.tensorsolve
tensorsolve()
torch.linalg.vander
vander()
torch.linalg.solve_ex
solve_ex()
torch.linalg.lu_factor_ex
lu_factor_ex()
torch.linalg.ldl_factor
ldl_factor()
torch.linalg.ldl_factor_ex
ldl_factor_ex()
torch.linalg.ldl_solve
ldl_solve()

torch.signal.windows.bartlett
bartlett()
torch.signal.windows.blackman
blackman()
torch.signal.windows.cosine
cosine()
torch.signal.windows.exponential
exponential()
torch.signal.windows.gaussian
gaussian()
torch.signal.windows.general_cosine
general_cosine()
torch.signal.windows.general_hamming
general_hamming()
torch.signal.windows.hamming
hamming()
torch.signal.windows.hann
hann()
torch.signal.windows.kaiser
kaiser()
torch.signal.windows.nuttall
nuttall()







torch.nn.attention.sdpa_kernel
sdpa_kernel()
SDPBackend
SDPBackend
SDPBackend.name
torch.nn.attention.flex_attention
flex_attention()
BlockMask Utilities
create_block_mask()
create_mask()
create_nested_block_mask()
and_masks()
or_masks()
noop_mask()


BlockMask
BlockMask
BlockMask.BLOCK_SIZE
BlockMask.as_tuple()
BlockMask.from_kv_blocks()
BlockMask.full_kv_indices
BlockMask.full_kv_num_blocks
BlockMask.full_q_indices
BlockMask.full_q_num_blocks
BlockMask.kv_indices
BlockMask.kv_num_blocks
BlockMask.mask_mod
BlockMask.numel()
BlockMask.q_indices
BlockMask.q_num_blocks
BlockMask.seq_lengths
BlockMask.shape
BlockMask.sparsity()
BlockMask.to()
BlockMask.to_dense()
BlockMask.to_string()
torch.nn.attention.experimental
TorchScript-based ONNX Exporter
Example: AlexNet from PyTorch to ONNX
Tracing vs Scripting
Avoiding Pitfalls
Avoid NumPy and built-in Python types
Avoid Tensor.data
Avoid in-place operations when using tensor.shape in tracing mode


Limitations
Types
Differences in Operator Implementations
Unsupported Tensor Indexing Patterns
Reads / Gets
Writes / Sets




Adding support for operators
ONNX exporter internals
ATen operators
List of supported operators
Adding support for an aten or quantized operator


torch.autograd.Functions
Static Symbolic Method
Inline Autograd Function


Custom operators
ONNX-script functions
C++ Operators


Discovering all unconvertible ATen ops at once


Frequently Asked Questions
Python API
Functions
export()
register_custom_op_symbolic()
unregister_custom_op_symbolic()
select_model_mode_for_export()
is_in_onnx_export()


Classes
TorchDynamo-based ONNX Exporter
Overview
Dependencies
A simple example
Inspecting the ONNX model using GUI
When the conversion fails
API Reference
dynamo_export()
ONNXProgram
ONNXProgram.apply_weights()
ONNXProgram.compute_values()
ONNXProgram.initialize_inference_session()
ONNXProgram.model_proto
ONNXProgram.optimize()
ONNXProgram.release()
ONNXProgram.save()


ExportOptions
enable_fake_mode()
ONNXRuntimeOptions
OnnxExporterError
OnnxRegistry
OnnxRegistry.get_op_functions()
OnnxRegistry.is_registered_op()
OnnxRegistry.opset_version
OnnxRegistry.register_op()


DiagnosticOptions

torch.optim.Optimizer.add_param_group
Optimizer.add_param_group()
torch.optim.Optimizer.load_state_dict
Optimizer.load_state_dict()
torch.optim.Optimizer.register_load_state_dict_pre_hook
Optimizer.register_load_state_dict_pre_hook()
torch.optim.Optimizer.register_load_state_dict_post_hook
Optimizer.register_load_state_dict_post_hook()
torch.optim.Optimizer.register_state_dict_pre_hook
Optimizer.register_state_dict_pre_hook()
torch.optim.Optimizer.register_state_dict_post_hook
Optimizer.register_state_dict_post_hook()
torch.optim.Optimizer.register_step_pre_hook
Optimizer.register_step_pre_hook()
torch.optim.Optimizer.register_step_post_hook
Optimizer.register_step_post_hook()
torch.optim.Optimizer.zero_grad
Optimizer.zero_grad()
Adadelta
Adadelta
Adadelta.add_param_group()
Adadelta.load_state_dict()
Adadelta.register_load_state_dict_post_hook()
Adadelta.register_load_state_dict_pre_hook()
Adadelta.register_state_dict_post_hook()
Adadelta.register_state_dict_pre_hook()
Adadelta.register_step_post_hook()
Adadelta.register_step_pre_hook()
Adadelta.state_dict()
Adadelta.step()
Adadelta.zero_grad()
Adafactor
Adafactor
Adafactor.add_param_group()
Adafactor.load_state_dict()
Adafactor.register_load_state_dict_post_hook()
Adafactor.register_load_state_dict_pre_hook()
Adafactor.register_state_dict_post_hook()
Adafactor.register_state_dict_pre_hook()
Adafactor.register_step_post_hook()
Adafactor.register_step_pre_hook()
Adafactor.state_dict()
Adafactor.step()
Adafactor.zero_grad()
Adagrad
Adagrad
Adagrad.add_param_group()
Adagrad.load_state_dict()
Adagrad.register_load_state_dict_post_hook()
Adagrad.register_load_state_dict_pre_hook()
Adagrad.register_state_dict_post_hook()
Adagrad.register_state_dict_pre_hook()
Adagrad.register_step_post_hook()
Adagrad.register_step_pre_hook()
Adagrad.state_dict()
Adagrad.step()
Adagrad.zero_grad()
Adam
Adam
Adam.add_param_group()
Adam.load_state_dict()
Adam.register_load_state_dict_post_hook()
Adam.register_load_state_dict_pre_hook()
Adam.register_state_dict_post_hook()
Adam.register_state_dict_pre_hook()
Adam.register_step_post_hook()
Adam.register_step_pre_hook()
Adam.state_dict()
Adam.step()
Adam.zero_grad()
AdamW
AdamW
AdamW.add_param_group()
AdamW.load_state_dict()
AdamW.register_load_state_dict_post_hook()
AdamW.register_load_state_dict_pre_hook()
AdamW.register_state_dict_post_hook()
AdamW.register_state_dict_pre_hook()
AdamW.register_step_post_hook()
AdamW.register_step_pre_hook()
AdamW.state_dict()
AdamW.step()
AdamW.zero_grad()
SparseAdam
SparseAdam
SparseAdam.add_param_group()
SparseAdam.load_state_dict()
SparseAdam.register_load_state_dict_post_hook()
SparseAdam.register_load_state_dict_pre_hook()
SparseAdam.register_state_dict_post_hook()
SparseAdam.register_state_dict_pre_hook()
SparseAdam.register_step_post_hook()
SparseAdam.register_step_pre_hook()
SparseAdam.state_dict()
SparseAdam.step()
SparseAdam.zero_grad()
Adamax
Adamax
Adamax.add_param_group()
Adamax.load_state_dict()
Adamax.register_load_state_dict_post_hook()
Adamax.register_load_state_dict_pre_hook()
Adamax.register_state_dict_post_hook()
Adamax.register_state_dict_pre_hook()
Adamax.register_step_post_hook()
Adamax.register_step_pre_hook()
Adamax.state_dict()
Adamax.step()
Adamax.zero_grad()
ASGD
ASGD
ASGD.add_param_group()
ASGD.load_state_dict()
ASGD.register_load_state_dict_post_hook()
ASGD.register_load_state_dict_pre_hook()
ASGD.register_state_dict_post_hook()
ASGD.register_state_dict_pre_hook()
ASGD.register_step_post_hook()
ASGD.register_step_pre_hook()
ASGD.state_dict()
ASGD.step()
ASGD.zero_grad()
LBFGS
LBFGS
LBFGS.add_param_group()
LBFGS.load_state_dict()
LBFGS.register_load_state_dict_post_hook()
LBFGS.register_load_state_dict_pre_hook()
LBFGS.register_state_dict_post_hook()
LBFGS.register_state_dict_pre_hook()
LBFGS.register_step_post_hook()
LBFGS.register_step_pre_hook()
LBFGS.state_dict()
LBFGS.step()
LBFGS.zero_grad()
NAdam
NAdam
NAdam.add_param_group()
NAdam.load_state_dict()
NAdam.register_load_state_dict_post_hook()
NAdam.register_load_state_dict_pre_hook()
NAdam.register_state_dict_post_hook()
NAdam.register_state_dict_pre_hook()
NAdam.register_step_post_hook()
NAdam.register_step_pre_hook()
NAdam.state_dict()
NAdam.step()
NAdam.zero_grad()
RAdam
RAdam
RAdam.add_param_group()
RAdam.load_state_dict()
RAdam.register_load_state_dict_post_hook()
RAdam.register_load_state_dict_pre_hook()
RAdam.register_state_dict_post_hook()
RAdam.register_state_dict_pre_hook()
RAdam.register_step_post_hook()
RAdam.register_step_pre_hook()
RAdam.state_dict()
RAdam.step()
RAdam.zero_grad()
RMSprop
RMSprop
RMSprop.add_param_group()
RMSprop.load_state_dict()
RMSprop.register_load_state_dict_post_hook()
RMSprop.register_load_state_dict_pre_hook()
RMSprop.register_state_dict_post_hook()
RMSprop.register_state_dict_pre_hook()
RMSprop.register_step_post_hook()
RMSprop.register_step_pre_hook()
RMSprop.state_dict()
RMSprop.step()
RMSprop.zero_grad()
Rprop
Rprop
Rprop.add_param_group()
Rprop.load_state_dict()
Rprop.register_load_state_dict_post_hook()
Rprop.register_load_state_dict_pre_hook()
Rprop.register_state_dict_post_hook()
Rprop.register_state_dict_pre_hook()
Rprop.register_step_post_hook()
Rprop.register_step_pre_hook()
Rprop.state_dict()
Rprop.step()
Rprop.zero_grad()
SGD
SGD
SGD.add_param_group()
SGD.load_state_dict()
SGD.register_load_state_dict_post_hook()
SGD.register_load_state_dict_pre_hook()
SGD.register_state_dict_post_hook()
SGD.register_state_dict_pre_hook()
SGD.register_step_post_hook()
SGD.register_step_pre_hook()
SGD.state_dict()
SGD.step()
SGD.zero_grad()
LRScheduler
LRScheduler
LRScheduler.get_last_lr()
LRScheduler.get_lr()
LRScheduler.load_state_dict()
LRScheduler.state_dict()
LRScheduler.step()
ReduceLROnPlateau
ReduceLROnPlateau
ReduceLROnPlateau.get_last_lr()
ReduceLROnPlateau.get_lr()
ReduceLROnPlateau.load_state_dict()
ReduceLROnPlateau.step()
LambdaLR
LambdaLR
LambdaLR.get_last_lr()
LambdaLR.get_lr()
LambdaLR.load_state_dict()
LambdaLR.state_dict()
LambdaLR.step()
MultiplicativeLR
MultiplicativeLR
MultiplicativeLR.get_last_lr()
MultiplicativeLR.get_lr()
MultiplicativeLR.load_state_dict()
MultiplicativeLR.state_dict()
MultiplicativeLR.step()
StepLR
StepLR
StepLR.get_last_lr()
StepLR.get_lr()
StepLR.load_state_dict()
StepLR.state_dict()
StepLR.step()
MultiStepLR
MultiStepLR
MultiStepLR.get_last_lr()
MultiStepLR.get_lr()
MultiStepLR.load_state_dict()
MultiStepLR.state_dict()
MultiStepLR.step()
ConstantLR
ConstantLR
ConstantLR.get_last_lr()
ConstantLR.get_lr()
ConstantLR.load_state_dict()
ConstantLR.state_dict()
ConstantLR.step()
LinearLR
LinearLR
LinearLR.get_last_lr()
LinearLR.get_lr()
LinearLR.load_state_dict()
LinearLR.state_dict()
LinearLR.step()
ExponentialLR
ExponentialLR
ExponentialLR.get_last_lr()
ExponentialLR.get_lr()
ExponentialLR.load_state_dict()
ExponentialLR.state_dict()
ExponentialLR.step()
PolynomialLR
PolynomialLR
PolynomialLR.get_last_lr()
PolynomialLR.get_lr()
PolynomialLR.load_state_dict()
PolynomialLR.state_dict()
PolynomialLR.step()
CosineAnnealingLR
CosineAnnealingLR
CosineAnnealingLR.get_last_lr()
CosineAnnealingLR.get_lr()
CosineAnnealingLR.load_state_dict()
CosineAnnealingLR.state_dict()
CosineAnnealingLR.step()
ChainedScheduler
ChainedScheduler
ChainedScheduler.get_last_lr()
ChainedScheduler.get_lr()
ChainedScheduler.load_state_dict()
ChainedScheduler.state_dict()
ChainedScheduler.step()
SequentialLR
SequentialLR
SequentialLR.get_last_lr()
SequentialLR.get_lr()
SequentialLR.load_state_dict()
SequentialLR.recursive_undo()
SequentialLR.state_dict()
SequentialLR.step()
CyclicLR
CyclicLR
CyclicLR.get_last_lr()
CyclicLR.get_lr()
CyclicLR.load_state_dict()
CyclicLR.scale_fn()
CyclicLR.step()
OneCycleLR
OneCycleLR
OneCycleLR.get_last_lr()
OneCycleLR.get_lr()
OneCycleLR.load_state_dict()
OneCycleLR.state_dict()
OneCycleLR.step()
CosineAnnealingWarmRestarts
CosineAnnealingWarmRestarts
CosineAnnealingWarmRestarts.get_last_lr()
CosineAnnealingWarmRestarts.get_lr()
CosineAnnealingWarmRestarts.load_state_dict()
CosineAnnealingWarmRestarts.state_dict()
CosineAnnealingWarmRestarts.step()
AveragedModel
AveragedModel
AveragedModel.add_module()
AveragedModel.apply()
AveragedModel.bfloat16()
AveragedModel.buffers()
AveragedModel.children()
AveragedModel.compile()
AveragedModel.cpu()
AveragedModel.cuda()
AveragedModel.double()
AveragedModel.eval()
AveragedModel.extra_repr()
AveragedModel.float()
AveragedModel.forward()
AveragedModel.get_buffer()
AveragedModel.get_extra_state()
AveragedModel.get_parameter()
AveragedModel.get_submodule()
AveragedModel.half()
AveragedModel.ipu()
AveragedModel.load_state_dict()
AveragedModel.modules()
AveragedModel.mtia()
AveragedModel.named_buffers()
AveragedModel.named_children()
AveragedModel.named_modules()
AveragedModel.named_parameters()
AveragedModel.parameters()
AveragedModel.register_backward_hook()
AveragedModel.register_buffer()
AveragedModel.register_forward_hook()
AveragedModel.register_forward_pre_hook()
AveragedModel.register_full_backward_hook()
AveragedModel.register_full_backward_pre_hook()
AveragedModel.register_load_state_dict_post_hook()
AveragedModel.register_load_state_dict_pre_hook()
AveragedModel.register_module()
AveragedModel.register_parameter()
AveragedModel.register_state_dict_post_hook()
AveragedModel.register_state_dict_pre_hook()
AveragedModel.requires_grad_()
AveragedModel.set_extra_state()
AveragedModel.set_submodule()
AveragedModel.share_memory()
AveragedModel.state_dict()
AveragedModel.to()
AveragedModel.to_empty()
AveragedModel.train()
AveragedModel.type()
AveragedModel.update_parameters()
AveragedModel.xpu()
AveragedModel.zero_grad()
SWALR
SWALR
SWALR.get_last_lr()
SWALR.get_lr()
SWALR.load_state_dict()
SWALR.state_dict()
SWALR.step()

JitScalarType
JitScalarType
JitScalarType.dtype()
JitScalarType.from_dtype()
JitScalarType.from_onnx_type()
JitScalarType.from_value()
JitScalarType.onnx_compatible()
JitScalarType.onnx_type()
JitScalarType.scalar_name()
JitScalarType.torch_name()



QuantStub
QuantStub
DeQuantStub
DeQuantStub
FloatFunctional
FloatFunctional
fuse_modules
fuse_modules
Quantization API Reference
torch.ao.quantization
Top level APIs
Preparing model for quantization
Utility functions


torch.ao.quantization.quantize_fx
torch.ao.quantization.qconfig_mapping
torch.ao.quantization.backend_config
torch.ao.quantization.fx.custom_config
torch.ao.quantization.quantizer
torch.ao.quantization.pt2e (quantization in pytorch 2.0 export implementation)
torch.ao.quantization.pt2e.export_utils
PT2 Export (pt2e) Numeric Debugger
torch (quantization related functions)
torch.Tensor (quantization related methods)
torch.ao.quantization.observer
torch.ao.quantization.fake_quantize
torch.ao.quantization.qconfig
torch.ao.nn.intrinsic
torch.ao.nn.intrinsic.qat
torch.ao.nn.intrinsic.quantized
torch.ao.nn.intrinsic.quantized.dynamic
torch.ao.nn.qat
torch.ao.nn.qat.dynamic
torch.ao.nn.quantized
torch.ao.nn.quantized.functional
torch.ao.nn.quantizable
torch.ao.nn.quantized.dynamic
Quantized dtypes and quantization schemes
Quantization Backend Configuration
Default values for native configurations
Quantization Accuracy Debugging
Data insensitive error
General tips
Int8 quantization tips


Data sensitive error
Implementation error


Numerical Debugging Tooling (prototype)




Remote Reference Protocol
Background
Assumptions
RRef Lifetime
Design Reasoning
Implementation


Protocol Scenarios
User Share RRef with Owner as Return Value
User Share RRef with Owner as Argument
Owner Share RRef with User
User Share RRef with User

Distributed Autograd Design
Background
Autograd recording during the forward pass
Distributed Autograd Context
Distributed Backward Pass
Computing dependencies
FAST mode algorithm
SMART mode algorithm


Distributed Optimizer
Simple end to end example

torch.ao.ns._numeric_suite_fx
OutputLogger
OutputLogger.forward()


OutputComparisonLogger
OutputComparisonLogger.forward()


NSTracer
NSTracer.is_leaf_module()


extract_weights()
add_loggers()
extract_logger_info()
add_shadow_loggers()
extract_shadow_logger_info()
extend_logger_results_with_comparison()
prepare_n_shadows_model()
loggers_set_enabled()
loggers_set_save_activations()
convert_n_shadows_model()
extract_results_n_shadows_model()
print_comparisons_n_shadows_model()


torch.ao.ns.fx.utils
compute_sqnr()
compute_normalized_l2_error()
compute_cosine_similarity()


check_sparse_tensor_invariants
check_sparse_tensor_invariants
check_sparse_tensor_invariants.disable()
check_sparse_tensor_invariants.enable()
check_sparse_tensor_invariants.is_enabled()
torch.Tensor.coalesce
Tensor.coalesce()
torch.Tensor.is_coalesced
Tensor.is_coalesced()
torch.sparse.softmax
softmax()
torch.sparse_compressed_tensor
sparse_compressed_tensor()
torch.sparse.mm
mm()
torch.hspmm
hspmm()
torch.sparse.addmm
addmm()
torch.sparse.spsolve
spsolve()
torch.Tensor.is_sparse_csr
Tensor.is_sparse_csr
torch.Tensor.to_sparse_coo
Tensor.to_sparse_coo()
torch.Tensor.sparse_resize_
Tensor.sparse_resize_()
torch.Tensor.sparse_resize_and_clear_
Tensor.sparse_resize_and_clear_()
torch.Tensor.crow_indices
Tensor.crow_indices()
torch.Tensor.col_indices
Tensor.col_indices()
torch.Tensor.row_indices
Tensor.row_indices()
torch.Tensor.ccol_indices
Tensor.ccol_indices()
torch.sparse.sum
sum()
torch.sparse.sampled_addmm
sampled_addmm()
torch.sparse.log_softmax
log_softmax()
torch.sparse.spdiags
spdiags()
torch.sparse.as_sparse_gradcheck
as_sparse_gradcheck()



torch.utils.rename_privateuse1_backend
rename_privateuse1_backend()
torch.utils.generate_methods_for_privateuse1_backend
generate_methods_for_privateuse1_backend()
torch.utils.get_cpp_backtrace
get_cpp_backtrace()
torch.utils.set_module
set_module()
torch.utils.swap_tensors
swap_tensors()
















torch._logging.set_logs
set_logs()
Threading Environment Variables
CUDA Environment Variables
MPS Environment Variables
Debugging Environment Variables
Miscellaneous Environment Variables
PYTORCH ProcessGroupNCCL Environment Variables























torch.func.grad_and_value
grad_and_value()
torch.func.linearize
linearize()
torch.func.functionalize
functionalize()
torch.func.functional_call
functional_call()
torch.func.stack_module_state
stack_module_state()
torch.func.replace_all_batch_norm_modules_
replace_all_batch_norm_modules_()
Patching Batch Norm
What’s happening?
How to fix
Option 1: Change the BatchNorm
Option 2: torchvision parameter
Option 3: functorch’s patching
Option 4: eval mode
torch.func.debug_unwrap
debug_unwrap()
































torch.nn.attention.bias.CausalBias
CausalBias
torch.nn.attention.bias.causal_lower_right
causal_lower_right()
torch.nn.attention.bias.causal_upper_left
causal_upper_left()
CausalVariant
CausalVariant























torch.escape-hatch
assume_constant_result
constrain_as_size_example
constrain_as_value_example
torch.cond
cond_branch_class_method
cond_branch_nested_function
cond_branch_nonlocal_variables
cond_closed_over_variable
cond_operands
cond_predicate
torch.dynamic-shape
cond_branch_class_method
cond_branch_nested_function
cond_branch_nonlocal_variables
cond_operands
cond_predicate
dynamic_shape_constructor
dynamic_shape_if_guard
dynamic_shape_map
dynamic_shape_round
dynamic_shape_slicing
dynamic_shape_view
list_contains
scalar_output
python.closure
cond_closed_over_variable
nested_function
torch.dynamic-value
constrain_as_size_example
constrain_as_value_example
python.data-structure
dictionary
fn_with_kwargs
list_contains
list_unpack
python.assert
dynamic_shape_assert
list_contains
python.control-flow
dynamic_shape_if_guard
list_unpack
static_for_loop
static_if
torch.map
dynamic_shape_map
python.builtin
dynamic_shape_round
tensor_setattr
type_reflection_method
python.object-model
model_attr_mutation
optional_input
python.context-manager
null_context_manager
torch.operator
unsupported_operator
torch.mutation
user_input_mutation



























torch.onnx.verification
verify_onnx_program()
VerificationInfo
VerificationInfo.from_tensors()


verify()
Deprecated
check_export_model_diff
GraphInfo
GraphInfoPrettyPrinter
OnnxBackend
OnnxTestCaseRepro
VerificationOptions
find_mismatch()
verify_aten_graph()
torch.compiler.compile
compile()
torch.compiler.reset
reset()
torch.compiler.allow_in_graph
allow_in_graph()
torch.compiler.substitute_in_graph
substitute_in_graph()
torch.compiler.assume_constant_result
assume_constant_result()
torch.compiler.list_backends
list_backends()
torch.compiler.disable
disable()
torch.compiler.set_stance
set_stance()
torch.compiler.cudagraph_mark_step_begin
cudagraph_mark_step_begin()
torch.compiler.is_compiling
is_compiling()
torch.compiler.is_dynamo_compiling
is_dynamo_compiling()
torch.compiler.is_exporting
is_exporting()
AOTInductor Minifier
Example Code
Minifier Launcher
Minified Result
PyTorch 2.0 Troubleshooting (old)
Diagnosing Runtime Errors
Torchdynamo Errors
Diagnosing TorchInductor Errors
Minifying TorchInductor Errors
Minifying Backend Compiler Errors


Performance Profiling
Accessing TorchDynamo Profiler
TorchInductor Debugging using TORCH_COMPILE_DEBUG
Graph Breaks


Identifying the Cause of a Graph Break
Excessive Recompilation


Accuracy Debugging
Extended Debugging
Cold Start Timing and Cache Corruption Debugging









ONNX supported TorchScript operators
Supported operators
Unsupported operators


Understanding TorchDynamo-based ONNX Exporter Memory Usage
TorchScript-based exporter
TorchDynamo-based exporter

















QuantWrapper
QuantWrapper

FXFloatFunctional
FXFloatFunctional
EmbeddingBag
EmbeddingBag
EmbeddingBag.from_float()

convert
convert
quantize
quantize
quantize_dynamic
quantize_dynamic
quantize_qat
quantize_qat
prepare
prepare
prepare_qat
prepare_qat
add_quant_dequant
add_quant_dequant
swap_module
swap_module
propagate_qconfig
propagate_qconfig_
default_eval_fn
default_eval_fn
prepare_fx
prepare_fx
prepare_qat_fx
prepare_qat_fx
convert_fx
convert_fx
fuse_fx
fuse_fx
QConfigMapping
QConfigMapping
QConfigMapping.from_dict()
QConfigMapping.set_global()
QConfigMapping.set_module_name()
QConfigMapping.set_module_name_object_type_order()
QConfigMapping.set_module_name_regex()
QConfigMapping.set_object_type()
QConfigMapping.to_dict()
get_default_qconfig_mapping
get_default_qconfig_mapping
get_default_qat_qconfig_mapping
get_default_qat_qconfig_mapping
BackendConfig
BackendConfig
BackendConfig.configs
BackendConfig.from_dict()
BackendConfig.set_backend_pattern_config()
BackendConfig.set_backend_pattern_configs()
BackendConfig.set_name()
BackendConfig.to_dict()
BackendPatternConfig
BackendPatternConfig
BackendPatternConfig.add_dtype_config()
BackendPatternConfig.from_dict()
BackendPatternConfig.set_dtype_configs()
BackendPatternConfig.set_fused_module()
BackendPatternConfig.set_fuser_method()
BackendPatternConfig.set_observation_type()
BackendPatternConfig.set_pattern()
BackendPatternConfig.set_qat_module()
BackendPatternConfig.set_reference_quantized_module()
BackendPatternConfig.set_root_module()
BackendPatternConfig.to_dict()
DTypeConfig
DTypeConfig
DTypeConfig.from_dict()
DTypeConfig.to_dict()
DTypeWithConstraints
DTypeWithConstraints
ObservationType
ObservationType
ObservationType.INPUT_OUTPUT_NOT_OBSERVED
ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
FuseCustomConfig
FuseCustomConfig
FuseCustomConfig.from_dict()
FuseCustomConfig.set_preserved_attributes()
FuseCustomConfig.to_dict()
PrepareCustomConfig
PrepareCustomConfig
PrepareCustomConfig.from_dict()
PrepareCustomConfig.set_float_to_observed_mapping()
PrepareCustomConfig.set_input_quantized_indexes()
PrepareCustomConfig.set_non_traceable_module_classes()
PrepareCustomConfig.set_non_traceable_module_names()
PrepareCustomConfig.set_output_quantized_indexes()
PrepareCustomConfig.set_preserved_attributes()
PrepareCustomConfig.set_standalone_module_class()
PrepareCustomConfig.set_standalone_module_name()
PrepareCustomConfig.to_dict()
ConvertCustomConfig
ConvertCustomConfig
ConvertCustomConfig.from_dict()
ConvertCustomConfig.set_observed_to_quantized_mapping()
ConvertCustomConfig.set_preserved_attributes()
ConvertCustomConfig.to_dict()
StandaloneModuleConfigEntry
StandaloneModuleConfigEntry
model_is_exported
model_is_exported
generate_numeric_debug_handle
generate_numeric_debug_handle
CUSTOM_KEY
CUSTOM_KEY
NUMERIC_DEBUG_HANDLE_KEY
NUMERIC_DEBUG_HANDLE_KEY
prepare_for_propagation_comparison
prepare_for_propagation_comparison
extract_results_from_loggers
extract_results_from_loggers
compare_results
compare_results
ObserverBase
ObserverBase
ObserverBase.with_args()
ObserverBase.with_callable_args()
MinMaxObserver
MinMaxObserver
MinMaxObserver.calculate_qparams()
MinMaxObserver.forward()
MinMaxObserver.reset_min_max_vals()
MovingAverageMinMaxObserver
MovingAverageMinMaxObserver
PerChannelMinMaxObserver
PerChannelMinMaxObserver
PerChannelMinMaxObserver.reset_min_max_vals()
MovingAveragePerChannelMinMaxObserver
MovingAveragePerChannelMinMaxObserver
HistogramObserver
HistogramObserver
PlaceholderObserver
PlaceholderObserver
RecordingObserver
RecordingObserver
NoopObserver
NoopObserver
get_observer_state_dict
get_observer_state_dict
load_observer_state_dict
load_observer_state_dict
default_observer
default_observer
default_placeholder_observer
default_placeholder_observer
default_debug_observer
default_debug_observer
default_weight_observer
default_weight_observer
default_histogram_observer
default_histogram_observer
default_per_channel_weight_observer
default_per_channel_weight_observer
default_dynamic_quant_observer
default_dynamic_quant_observer
default_float_qparams_observer
default_float_qparams_observer
AffineQuantizedObserverBase
AffineQuantizedObserverBase
AffineQuantizedObserverBase.calculate_qparams()
AffineQuantizedObserverBase.forward()
AffineQuantizedObserverBase.with_args()
Granularity
Granularity
MappingType
MappingType
PerAxis
PerAxis
PerBlock
PerBlock
PerGroup
PerGroup
PerRow
PerRow
PerTensor
PerTensor
PerToken
PerToken
TorchAODType
TorchAODType
ZeroPointDomain
ZeroPointDomain
get_block_size
get_block_size
FakeQuantizeBase
FakeQuantizeBase
FakeQuantize
FakeQuantize
FixedQParamsFakeQuantize
FixedQParamsFakeQuantize
FixedQParamsFakeQuantize.extra_repr()
FusedMovingAvgObsFakeQuantize
FusedMovingAvgObsFakeQuantize
default_fake_quant
default_fake_quant
default_weight_fake_quant
default_weight_fake_quant
default_per_channel_weight_fake_quant
default_per_channel_weight_fake_quant
default_histogram_fake_quant
default_histogram_fake_quant
default_fused_act_fake_quant
default_fused_act_fake_quant
default_fused_wt_fake_quant
default_fused_wt_fake_quant
default_fused_per_channel_wt_fake_quant
default_fused_per_channel_wt_fake_quant
disable_fake_quant
disable_fake_quant
enable_fake_quant
enable_fake_quant
disable_observer
disable_observer
enable_observer
enable_observer
QConfig
QConfig
default_qconfig
default_qconfig
default_debug_qconfig
default_debug_qconfig
default_per_channel_qconfig
default_per_channel_qconfig
default_dynamic_qconfig
default_dynamic_qconfig
float16_dynamic_qconfig
float16_dynamic_qconfig
float16_static_qconfig
float16_static_qconfig
per_channel_dynamic_qconfig
per_channel_dynamic_qconfig
float_qparams_weight_only_qconfig
float_qparams_weight_only_qconfig
default_qat_qconfig
default_qat_qconfig
default_weight_only_qconfig
default_weight_only_qconfig
default_activation_only_qconfig
default_activation_only_qconfig
default_qat_qconfig_v2
default_qat_qconfig_v2
ConvReLU1d
ConvReLU1d
ConvReLU2d
ConvReLU2d
ConvReLU3d
ConvReLU3d
LinearReLU
LinearReLU
ConvBn1d
ConvBn1d
ConvBn2d
ConvBn2d
ConvBn3d
ConvBn3d
ConvBnReLU1d
ConvBnReLU1d
ConvBnReLU2d
ConvBnReLU2d
ConvBnReLU3d
ConvBnReLU3d
BNReLU2d
BNReLU2d
BNReLU3d
BNReLU3d
LinearReLU
LinearReLU
ConvBn1d
ConvBn1d
ConvBnReLU1d
ConvBnReLU1d
ConvBn2d
ConvBn2d
ConvBnReLU2d
ConvBnReLU2d
ConvReLU2d
ConvReLU2d
ConvBn3d
ConvBn3d
ConvBnReLU3d
ConvBnReLU3d
ConvReLU3d
ConvReLU3d
update_bn_stats
update_bn_stats
freeze_bn_stats
freeze_bn_stats
BNReLU2d
BNReLU2d
BNReLU3d
BNReLU3d
ConvReLU1d
ConvReLU1d
ConvReLU2d
ConvReLU2d
ConvReLU3d
ConvReLU3d
LinearReLU
LinearReLU
LinearReLU
LinearReLU
Conv2d
Conv2d
Conv3d
Conv3d
Linear
Linear
Linear.from_float()
Linear
Linear
ReLU6
ReLU6
Hardswish
Hardswish
ELU
ELU
LeakyReLU
LeakyReLU
Sigmoid
Sigmoid
BatchNorm2d
BatchNorm2d
BatchNorm3d
BatchNorm3d
Conv1d
Conv1d
Conv1d.from_float()
Conv2d
Conv2d
Conv2d.from_float()
Conv3d
Conv3d
Conv3d.from_float()
ConvTranspose1d
ConvTranspose1d
ConvTranspose2d
ConvTranspose2d
ConvTranspose3d
ConvTranspose3d
Embedding
Embedding
Embedding.from_float()
QFunctional
QFunctional
Linear
Linear
Linear.from_float()
Linear.from_reference()
LayerNorm
LayerNorm
GroupNorm
GroupNorm
InstanceNorm1d
InstanceNorm1d
InstanceNorm2d
InstanceNorm2d
InstanceNorm3d
InstanceNorm3d
avg_pool2d
avg_pool2d
avg_pool3d
avg_pool3d
adaptive_avg_pool2d
adaptive_avg_pool2d
adaptive_avg_pool3d
adaptive_avg_pool3d
conv1d
conv1d
conv2d
conv2d
conv3d
conv3d
interpolate
interpolate
linear
linear
max_pool1d
max_pool1d
max_pool2d
max_pool2d
celu
celu
leaky_relu
leaky_relu
hardtanh
hardtanh
hardswish
hardswish
threshold
threshold
elu
elu
hardsigmoid
hardsigmoid
clamp
clamp
upsample
upsample
upsample_bilinear
upsample_bilinear
upsample_nearest
upsample_nearest
LSTM
LSTM
MultiheadAttention
MultiheadAttention
MultiheadAttention.dequantize()
MultiheadAttention.forward()
Linear
Linear
Linear.from_float()
Linear.from_reference()
LSTM
LSTM
GRU
GRU
RNNCell
RNNCell
LSTMCell
LSTMCell
GRUCell
GRUCell
torch.ao.ns._numeric_suite
compare_weights()
get_logger_dict()
Logger
Logger.forward()


ShadowLogger
ShadowLogger.forward()


OutputLogger
OutputLogger.forward()


Shadow
Shadow.forward()
Shadow.add()
Shadow.add_scalar()
Shadow.mul()
Shadow.mul_scalar()
Shadow.cat()
Shadow.add_relu()


prepare_model_with_stubs()
compare_model_stub()
get_matching_activations()
prepare_model_outputs()
compare_model_outputs()
















































