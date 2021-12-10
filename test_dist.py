# coding=utf8

import torch
import torch.distributed as dist
import nvidia_dlprof_pytorch_nvtx

def test(device, rank, local_rank):
    with torch.autograd.profiler.emit_nvtx():
        _shape = [1280, 1280]
        x1 = torch.randn(_shape)
        x1 = x1.to(device)
        x2 = torch.ones(_shape)
        x2 = x2.to(device)
        dist.all_reduce(x1)
        y = x1 * x2
        return y


if __name__ == '__main__':
    nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
    dist.init_process_group(backend='nccl')
    device_num = torch.cuda.device_count()
    rank = dist.get_rank()
    local_rank = rank % device_num 
    device = torch.device('cuda', local_rank)
    test(device, rank, local_rank)
