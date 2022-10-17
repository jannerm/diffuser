#!/bin/bash

## `qz run` example planning
## usage:
	## ./gcp/run.sh


for env in halfcheetah;
do
	for buffer in medium-expert;
	do
		echo $env $buffer
		qz run vm4 \
			python -u /home/code/scripts/plan_guided.py \
			--loadbase /home/bucket/logs/diffuser-singularity/logs \
			--logbase /home/logs/diffuser-singularity/logs \
			--dataset $env-$buffer-v2 \
			--suffix 0
	done
done
