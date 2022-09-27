#!/bin/bash

for podid in {0..39}
do
    
    podname="tc""$podid"
    nodeid=`expr ${podid} % 10 + 1`
    nodename="k8s-node${nodeid}"
    #echo "podname = ${podname} nodename=${nodename}"
    if !(test -z "$(kubectl get pod -o wide | grep $podname)"); then
	#echo "The result is empty."
    # else
	kubectl delete pod ${podname}
    fi
    sed -i "4c\  name: ${podname}"  /root/tomcat/pod.yaml
    sed -i "8c\  nodeName: ${nodename}"  /root/tomcat/pod.yaml
    kubectl create -f /root/tomcat/pod.yaml
done
