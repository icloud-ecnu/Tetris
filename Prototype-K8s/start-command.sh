#!/bin/bash
nohup python3 -u ./runk8s.py --algo=sxy > ./log/runk8s_sxy.log &
nohup python3 -u ./runk8s.py > ./runk8s-norequest.log &

kubectl get pod -o wide
kubectl top pod | grep tc2 | awk '{print $2}' | tr -cd '[0-9]'
kubectl top pod | grep -P '^tc2' | awk '{print $2}' | tr -cd '[0-9]'

# run runk8s-multi-bench
nohup python3 -u ./runk8s-multi-bench.py --algo=sxy > ./log/runk8s-multi-bench-sxy.log &