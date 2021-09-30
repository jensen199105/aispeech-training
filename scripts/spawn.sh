#!/bin/bash

# A helper script to spawn multiple worker for each gpu in
# CUDA_VISIBLE_DEVICES

OLD_IFS=$IFS
IFS=','
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cuda_list="${CUDA_VISIBLE_DEVICES[@]}"
pids=()
for cur_cuda in $cuda_list;
do
    echo "running on $cur_cuda"
    IFS=$OLD_IFS
    CUDA_VISIBLE_DEVICES=$cur_cuda "$@" &
    pids+=($!);
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

for pid in ${pids[*]}; do
    wait $pid
done
