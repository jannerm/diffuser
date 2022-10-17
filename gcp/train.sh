#/bin/bash

## `qz exec` example training
## usage:
	## qz exec ${VM_INSTANCE} gcp/train.sh


for env in halfcheetah;
do
	for buffer in medium-expert;
	do
		echo $env $buffer
		qz run vm2 \
			python -u /home/code/scripts/train.py \
			--logbase /home/logs/diffuser-singularity/logs \
			--dataset $env-$buffer-v2
	done
done
