# Serverless Computing
### Assignment 3 for COS598D: System and Machine Learning

In this assignment, you are required to implement and evaluate batched matrix multiplications as serverless calls. You are going to use [Submitit](https://github.com/facebookincubator/submitit) as the serverless framework. We provide an example of a serverless call performing an add operation in `example.py`. You are going to implement batched matrix multiplications in `batchMM.py` as serverless calls on a single CPU, multiple CPUs, and GPU.

## Getting Started
 - Clone this repo
 - Install [Submitit](https://github.com/facebookincubator/submitit) on a server with [Slurm](https://slurm.schedmd.com/quickstart.html) (e.g. [Adroit](https://researchcomputing.princeton.edu/systems/adroit) Server, or you can install Slurm on your server.) 
 - Install [Pytorch](https://pytorch.org/)

## How to Run 
Run `python example.py` for an serverless call example performing an add operation with Submitit.
You will need to fill in the `batchMM.py` file and run it using the command `python batchMM.py`.

## Your Tasks

### 1. Implement a Batched Matrix Multiplication Serverless Call

Implement a batched matrix multiplication serverless function (`bmm_single`) on a single CPU. 
The serverless call takes batched matrices $A$, and batched matrices $B$ as input, computes batch matrix multiplication on $A$ and $B$ for 1,000 times on a single CPU, and returns batched output matrices $C$.
$A$, $B$, and $C$ are Pytorch tensors of sizes `[bs, M, N]`, `[bs, N, K]`, and `[bs, M, K]` correspondingly. 

Implement the function `call_singleCPU` to make the serverless function call with randomnly initialized input matrices and verify the result correctness with the matrix parameters: `bs = 5, 10, 15` and `M = N = K = 100, 500, 1000`.


### 2. Compare the performance of serverless call with bare metal time

Measure the time for your serverless call and its bare metal time, i.e., the actual time it takes to run on the CPU server.
For each case in the following table, run your serverless call 5 times and report the times for all the 5 runs.

***Performance of Bare Metal v.s. Serverless Call of Batched Matrix Multiplication (Single CPU)***

|   [bs, M, N, K]  |   Bare Metal Time (s) |   Serverless Call Time (s) |  
|----------------|----------------|-------------|
| [5, 1000, 1000, 1000] |  |    |      | 
| [10, 1000, 1000, 1000] |  |    |      |   
| [15, 1000, 1000, 1000] |  |    |      |

Discuss the following
- Difference between the bare metal time and serverless call time and the reason.
- Time variance across different runs and the reason.

### 3. Parallelize Batch Matrix Multiplication Serverless Call on multiple CPUs

Implement a serverless call `mm_single` performing one matrix multiplication for 1,000 times on a single CPU.

Implement function `call_multiCPU` to parallelize the batched matrix multiplication by making one serverless call (`mm_single`) for each matrix multiplication within the batch, i.e., making `bs` serverless calls in parallel for batch size `bs`, and verify the result correctness.

For each case in the following table, run your parallelized serverless calls 5 times and report the median time.

***Performance of Serverless Batched Matrix Multiplication (Single CPU v.s. Multiple CPUs)***

|   `[bs, M, N, K]`  |   Single CPU (s) |   Multiple (`bs`) CPUs (s) |  
|----------------|----------------|-------------|
| [5, 100, 100, 100] |  |    |      | 
| [10, 100, 100, 100] |  |    |      |   
| [15, 100, 100, 100] |  |    |      |
| [5, 500, 500, 500] |  |    |      | 
| [10, 500, 500, 500] |  |    |      |   
| [15, 500, 500, 500] |  |    |      |
| [5, 1000, 1000, 1000] |  |    |      | 
| [10, 1000, 1000, 1000] |  |    |      |   
| [15, 1000, 1000, 1000] |  |    |      |

Discuss the performance difference between the single-CPU and the multi-CPU serverless call.

### 4. Batch Matrix Multiplication Serverless Call on GPU

Implement the batched multiplication serverless call (`bmm_gpu`) for 1,000 times on GPU. 

Implement function `call_GPU` to make the serverless function call and verify its correctness.

For each case in the following table, run your serverless call 5 times and report the median time.

***Performance of Serverless Batched Matrix Multiplication (Single CPU v.s. Multiple CPUs v.s. GPU)***

|   `[bs, M, N, K]`  |  Single CPU (s) |   Multiple (`bs`) CPUs (s) |  GPU (s) |
|----------------|----------------|-------------|-------------|
| [5, 100, 100, 100] |  |    |      |     | 
| [10, 100, 100, 100] |  |    |      |       | 
| [15, 100, 100, 100] |  |    |      |    | 
| [5, 500, 500, 500] |  |    |      |    | 
| [10, 500, 500, 500] |  |    |      |      | 
| [15, 500, 500, 500] |  |    |      |   | 
| [5, 1000, 1000, 1000] |  |    |      |    | 
| [10, 1000, 1000, 1000] |  |    |      |      | 
| [15, 1000, 1000, 1000] |  |    |      |   | 

Discuss the following
- Performance comparison of the three cases.
- Pros and Cons for the three design choices: single CPU, multiple CPUs, and GPU.
- When serverless computing can be beneficial?

### 5. Batch Matrix Multiplication Serverless Call on multiple GPUs

Increase the number of batched matrix multiplication executions in batched multiplication serverless call on GPU (`bmm_gpu`) to 5,000 times. 

Implement function `call_multiGPU` to parallelize the batched matrix multiplication by spliting each batched matrix by half and making one serverless call (`bmm_gpu`) for each half batch, i.e., making 2 serverless calls in parallel, and verify the result's correctness.

For the case in the following table, run your serverless call 5 times and report the time for all the 5 runs.

***Performance of Serverless Batched Matrix Multiplication (Single GPU v.s. Multi GPUs)***

|   `[bs, M, N, K]`  | Bare Metal GPU (s)|  Single GPU (s) |   Multiple GPUs (s) | 
|----------------|----------------|-------------|-------------|
| [200, 1000, 1000, 1000] |  |    |      |   |

Discuss the performance and overhead for each of the three cases.

## What to be included in your submission

- A report includes your experiment settings, e.g., the server and environment you run your experiements on, the CPUs and GPUs you are using, etc., your results, discussions, and answers to all the questions in the previous section.
- Your `batchMM.py` file.
