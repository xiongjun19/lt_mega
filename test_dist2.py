# coding=utf8

import torch
import torch.distributed as dist
# import nvidia_dlprof_pytorch_nvtx

def test(device, rank, local_rank):
    # with torch.autograd.profiler.emit_nvtx():
        # 和backward-embedding-all-reduce 形状一致
        _shape = [30592, 1024]
        x1 = torch.randn(_shape, dtype=torch.float16)
        x1 = x1.to(device)
        x2 = torch.ones(_shape, dtype=torch.float16)
        x2 = x2.to(device)
        dist.all_reduce(x1)
        y = x1 * x2
        # 和backward-params-all-reduce  
        z = torch.randn([336297858], dtype=torch.float16, device=device) 
        dist.all_reduce(z)
        res = z * z
        
        return res 


if __name__ == '__main__':
    # nvidia_dlprof_pytorch_nvtx.init(enable_function_stack=True)
    dist.init_process_group(backend='nccl')
    device_num = torch.cuda.device_count()
    rank = dist.get_rank()
    local_rank = rank % device_num 
    device = torch.device('cuda', local_rank)
    test(device, rank, local_rank)
