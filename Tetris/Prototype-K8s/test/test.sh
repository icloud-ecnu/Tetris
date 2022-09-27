podname="tc0"

if test -z "$(kubectl get pod -o wide | grep $podname)"; then
	echo "The result is empty."
else
	echo "The result is not empty."
fi
