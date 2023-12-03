#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=$2
echo "Running command : nohup nice -n 1 /opt/uio/modules/rhel8/easybuild/software/Python/3.10.4-GCCcore-11.3.0/bin/python -u $1  &>  out_nohups/out_$1.txt &"
nohup nice -n 1 /opt/uio/modules/rhel8/easybuild/software/Python/3.10.4-GCCcore-11.3.0/bin/python -u "$1"  &> "out_nohups/out_$1.txt" &
