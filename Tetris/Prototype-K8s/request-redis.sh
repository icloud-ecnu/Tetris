#!/bin/bash
# set -e
#访问请求
while true
do
        # redis-benchmark -h localhost -p 6379 -c 100 -n 100000
        # localhost:6379的时候top查看资源发现是master节点的资源噌噌噌涨
	redis-benchmark -h localhost -p 31379 -c 100 -n 100000
	echo "#################################### Request send ####################################"
        sleep 1
done
