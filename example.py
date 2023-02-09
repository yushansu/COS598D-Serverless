import submitit
import numpy as np
import time

def add(a, b):
    return a + b

log_folder = "log_test/%j"
executor = submitit.AutoExecutor(folder=log_folder)
executor.update_parameters(timeout_min=1, slurm_partition="all")
start = time.time()
job = executor.submit(add, 5, 7)
output = job.result()
end = time.time()
print("Serverless call time: %.2f s" % (end - start))

assert output == 12