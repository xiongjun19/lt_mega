# coding=utf8

import os
import torch
import torch.distributed as dist


def init_general_dist(pipe_para=1):
    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    device_count = torch.cuda.device_count()

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = rank % device_count
        local_rank = device
        torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    data_group, pipe_group = init_paralel_group(pipe_para, rank, world_size)
    return data_group, pipe_group
    

def init_paralel_group(pipe_para, rank, world_size):
    dp_size = world_size // pipe_para
    data_group = None
    pipe_group = None
    for i in range(pipe_para):
        start_rank = i * dp_size
        end_rank = (i + 1) * dp_size 
        ranks = range(start_rank, end_rank)
        print("d_ranks: ", list(ranks))
        group = dist.new_group(ranks)
        if rank in ranks:
            data_group = group

    for i in range(dp_size):
        start_rank = i 
        end_rank = world_size 
        ranks = range(start_rank, end_rank, dp_size)
        group = dist.new_group(ranks)
        print("p_ranks: ", list(ranks))
        if rank in ranks:
            pipe_group = group
    return data_group, pipe_group


class TestObject(object):
    def __init__(self, args):
        pipe_para = getattr(args, 'pipe_para', 1)
        self.d_group, self.p_group = init_general_dist(pipe_para)
        device_count = torch.cuda.device_count()
        rank = dist.get_rank()
        print(type(rank))
        print(type(device_count))
        local_rank = rank % device_count
        self.device = torch.device('cuda', local_rank)

    def run(self):
        for i in range(10):
            _shape1 = [30592, 1024]
            x1 = torch.randn(_shape1, dtype=torch.float16, device=self.device)
            y1 = torch.ones(_shape1, dtype=torch.float16, device=self.device)
            _shape2 = [336297858 // 2]
            x2 = torch.randn(_shape2, dtype=torch.float16, device=self.device)
            dist.all_reduce(x2, group=self.d_group)
            z2 = x2 * x2
            dist.all_reduce(x1, group=self.p_group)
            z1 = x1 * y1
        return z1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pipe_para',  type=int, default=1)
    args, _= parser.parse_known_args()
    t_obj = TestObject(args)
    t_obj.run()

