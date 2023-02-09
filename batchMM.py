import submitit
import numpy as np
import torch
import time

def alloc_tensor(bs, M, N, K):
    tensor1 = torch.randn(bs, M, N)
    tensor2 = torch.randn(bs, N, K)
    return tensor1, tensor2

'''
Batched Matrix Multiplication
serverless call on CPU
'''
def bmm_single(a, b):
    start = time.time()
    # Run batched matrix multiplication on a and b for 1000 times
    for i in range(1000):
        # TODO: batched matrix multiplication on a and b
    end = time.time()
    # Record the bare metal time
    rtime = end - start
    return result, rtime

'''
Matrix Multiplication
serverless call on CPU
'''
def mm_single(a, b):
    # Run one matrix multiplication on a and b for 1000 times
    for i in range(1000):
        # TODO: matrix multiplication on a and b
    return result

'''
Batched Matrix Multiplication
serverless call on GPU
'''
def bmm_gpu(a, b):
    start = time.time()
    # TODO: copy a and b to GPU
    # Run batched matrix multiplication on a and b for 1000 times
    for i in range(1000):
        # TODO: batched matrix multiplication on GPU
    # TODO: copy results back to CPU
    end = time.time()
    # Record the bare metal time
    rtime = end - start
    return result, rtime

'''
Make serverless call on a single CPU
'''
def call_singleCPU(tensor1, tensor2):
    log_folder = "log_test/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=20, slurm_partition="all")
    num_finished = 0
    # Record the serverless call time
    start = time.time()
    job = executor.submit(bmm_single, tensor1, tensor2)
    output, rtime = job.result()
    end = time.time()
    # TODO: verify result correctness
    return end - start, rtime

'''
Make parallel serverless calls on multiple (bs) CPUs
'''
def call_multiCPU(tensor1, tensor2):
    # Submitit for multiple CPUs
    log_folder = "log_test/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=20, slurm_partition="all")
    num_finished = 0
    # Record the serverless call time
    start = time.time()
    # TODO: make bs parallel serverless calls to mm_single on bs CPUs
    output = [job.result() for job in jobs]
    end = time.time()
    # TODO: verify result correctness
    return end - start

'''
Make serverless call on GPU
'''
def call_GPU(tensor1, tensor2):
    log_folder = "log_test/%j"
    # TODO: Define executer and specify parameters
    num_finished = 0
    # Record the serverless call time
    start = time.time()
    # TODO: make serverless call to bmm_gpu on GPU
    output, rtime = job.result()
    end = time.time()
    # TODO: verify result correctness
    return end - start, rtime


def main():
    paramList = [[500, 500, 500],
                [1000, 1000, 1000]
                ]
    for bs in [5,10,15]:
        for param in paramList:
            M = param[0]
            N = param[1]
            K = param[2]
            t1, t2 = alloc_tensor(bs, M, N, K)
            for r in range(5):
                time_singleCPU, rtime_singleCPU = call_singleCPU(t1, t2)
                time_multiCPU = call_multiCPU(t1, t2)
                time_GPU, rtime_GPU = call_GPU(t1, t2)

if __name__ == "__main__":
    main()