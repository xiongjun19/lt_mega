# coding=utf8

'''
this file is designed to join pytorch profile's log with nccl's log
'''

import json
import argparse
from dataclasses import dataclass


@dataclass
class NcclObj: 
    device: int
    op_name: str
    count: int
    dtype: int
    op: int
    stream: str


def load_nccl(f_path):
    res = []
    with open(f_path) as r:
        for line in r:
            if 'NCCL' in line and 'INFO' in line:
                obj = _parse_nccl_obj(line)
                if obj is not None:
                    res.append(obj)
    return res


def _parse_nccl_obj(line):
    line = line.strip()
    line_arr = line.split()
    index = line_arr.index('NCCL')
    if index < 2:
        return None
    line_arr = line_arr[index-2:]
    device = line_arr[1]
    device = int(device.replace('[', '').replace(']', ''))
    op_name = line_arr[4].strip(':')
    count = _parse_next('count', line_arr)
    if count is None:
        return None
    count = int(count)
    dtype = _parse_next('datatype', line_arr)
    if dtype is None:
        return None
    dtype = int(dtype)
    op = _parse_next('op', line_arr)
    if op is None:
        return None
    op = int(op)
    stream = _parse_next('stream', line_arr)
    stream = str(stream)
    return NcclObj(device, op_name, count, dtype, op, stream)


def _parse_next(key, line_arr):
    lg = len(line_arr)
    for i, elem in enumerate(line_arr):
        if key == elem:
            if lg > i+1:
                return line_arr[i+1]
    return None


def _join_coarsely(nccl_event_arr, ncl_obj_arr):
    res = []
    for event in nccl_event_arr:
        ncl_obj_sub_arr = _filter_sub_arr(event, ncl_obj_arr) 
        res.append([event, ncl_obj_sub_arr])
    return res

def _filter_sub_arr(event, ncl_obj_arr):
    res = []
    for obj in ncl_obj_arr:
        if _is_match(obj, event):
            res.append(obj)
    return res

def _is_match(obj, event):
    name = event['name'].lower()
    device = int(event['args']['device'])
    if obj.device != device:
        return False
    if obj.op_name.lower() not in name:
        return False
    if "_f32" in name and obj.dtype!=7:
        return False

    if "_f16" in name and obj.dtype!=6:
        return False

    return True

def calc_cnt_map(nccl_obj_arr):
    res = dict()
    for nccl_obj in nccl_obj_arr:
        res[str(nccl_obj)] = res.get(str(nccl_obj), 0) + 1
    return res

def _filter_map(ncl_obj_cnt, iters):
    res = dict()
    for k, v in ncl_obj_cnt.items():
        if v >= iters:
            res[k] = v
    return res

def parse_and_mine(args):
    ncl_obj_arr = load_nccl(args.nccl)
    ncl_obj_cnt=  calc_cnt_map(ncl_obj_arr)
    filter_obj_cnt = _filter_map(ncl_obj_cnt, args.iters)
    return ncl_obj_arr, filter_obj_cnt

def test(args):
    ncl_obj_arr, ncl_obj_cnt = parse_and_mine(args)
    print("obj_arr is folllowing: ")
    for x in ncl_obj_arr:
        if x.device ==1 and str(x) in ncl_obj_cnt:
            print(x)
    print(f"number of nccl_objs is {len(ncl_obj_arr)}")
    for key, val in ncl_obj_cnt.items():
        print(key)
        print(val)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--f_py_prof', type=str, help="file path to the pytorch profiler")
    parser.add_argument('-n', '--nccl', type=str, help='file path to log with nccl\'s log')
    parser.add_argument('--iters', type=int, help='minimum iterations')
    args = parser.parse_args()
    test(args)


