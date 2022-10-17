#!/bin/bash

## `qz exec` example planning
## usage:
	## qz exec ${VM_INSTANCE} gcp/plan.sh

parallel -j 2 -u \
	python -u /home/code/scripts/plan_guided.py \
		--loadbase /home/bucket/logs/diffuser-singularity/logs \
		--logbase /home/logs/diffuser-singularity/logs \
		--dataset {2}-{3}-v2 \
		--n_guide_steps 2 \
		--scale {4} \
		--t_stopgrad {5} \
		--verbose False \
		--suffix {1} \
		::: {0..19} \
		::: halfcheetah \
		::: medium-expert \
		::: 0.001 \
		::: 0
