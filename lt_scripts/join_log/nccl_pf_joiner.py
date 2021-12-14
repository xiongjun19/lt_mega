# coding=utf8

'''
this file is designed to join pytorch profile's log with nccl's log
'''

import json
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import nccl_log_miner as ncl_miner


def _load_py_prof(f_path):
    res = []
    with open(f_path) as r:
        obj = json.load(r)
        evt_obj_arr = obj['traceEvents']
        for evt_obj in evt_obj_arr:
            if 'cat' in evt_obj and 'name' in evt_obj:
                cat = evt_obj['cat']
                name = evt_obj['name'].lower()
                if 'Kernel' == cat:
                    if 'nccl' in name:
                        res.append(evt_obj)
    return res

def _join_coarsely(nccl_event_arr, ncl_obj_arr, ncl_obj_cnt):
    res = []
    for event in nccl_event_arr:
        ncl_obj_sub_arr = _filter_sub_arr(event, ncl_obj_arr) 
        ncl_obj_sub_arr = [x for x in ncl_obj_sub_arr if str(x) in ncl_obj_cnt]
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


def _cons_stream_map(coarse_joined_res):
    # first stream no to ncl_address map
    no_2_addr_map_list = []
    for evt, ncl_obj_subs in coarse_joined_res:
        tmp_dict = dict()
        key = evt['args']['stream']
        tmp_dict[key] = []
        for ncl_obj in ncl_obj_subs:
            val = ncl_obj.stream
            if val not in tmp_dict[key]:
                tmp_dict[key].append(val)
        no_2_addr_map_list.append(tmp_dict)
    # then merge the intersection of the map
    joined_key2map_list = _find_intersection(no_2_addr_map_list)
    # thirdly select out one to  one map if  possible
    res = OrderedDict()
    taken = set()
    for key, val in joined_key2map_list.items():
        if len(val) == 1 and val[0] not in taken:
            res[key] = val
            taken.add(val[0])
    for key, val in joined_key2map_list.items():
        if key not in res:
            new_val = [x for x in val if x not in taken]
            res[key] = new_val
    return res 


def _find_intersection(map_list):
    tmp_key_arr_map = OrderedDict() 
    for tmp_dict in map_list:
        for key, val in tmp_dict.items():
            if key not in tmp_key_arr_map:
                tmp_key_arr_map[key] = []
            tmp_key_arr_map[key].append(val)
    res = OrderedDict()
    for key, val in tmp_key_arr_map.items():
        tmp_list = []
        is_first = True 
        for cur_list in val:
            tmp_list = _list_intersect(tmp_list, cur_list, is_first) 
            is_first = False
        res[key] = tmp_list
    return res


def _list_intersect(list1, list2, is_first=False):
    if is_first:
        return list2
    res = []
    for elem in list1:
        if elem in list2:
            res.append(elem)
    return res


def _filter_with_stream(coarse_joined_res, stream_map):
    res = []
    for evt, sub_arr in coarse_joined_res:
        evt_stream = evt['args']['stream']
        obj_streams = stream_map.get(evt_stream, [])
        tmp_arr = []
        for ncl_obj in sub_arr:
            if ncl_obj not in tmp_arr and ncl_obj.stream in obj_streams:
                tmp_arr.append(ncl_obj)
        res.append([evt, tmp_arr])
    return res

        

def main(args):
    f_py_prof = args.f_py_prof
    nccl_events = _load_py_prof(f_py_prof)
    print(f"number of events is {len(nccl_events)}")
    for evt in nccl_events:
        print("new evt: ")
        print(evt)

    ncl_obj_arr, ncl_obj_cnt = ncl_miner.parse_and_mine(args)
    # step 1 coarsely join
    coarse_joined_res = _join_coarsely(nccl_events, ncl_obj_arr, ncl_obj_cnt) 
    # step 2 construct stream map
    stream_map = _cons_stream_map(coarse_joined_res)
    for key, val in stream_map.items():
        print(key)
        print(val)

    # step3 filter joined_res by stream map
    filter_joined_res = _filter_with_stream(coarse_joined_res, stream_map)
    for evt, sub_arr in filter_joined_res:
        print("*" * 66)
        print(evt)
        print(sub_arr)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--f_py_prof', type=str, help="file path to the pytorch profiler")
    parser.add_argument('-n', '--nccl', type=str, help='file path to log with nccl\'s log')
    parser.add_argument('--iters', type=int, help='minimum iterations log required from nccl')
    args = parser.parse_args()
    main(args)


