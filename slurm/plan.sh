export n_trials=50

for start in 0 50 100;
do
	for env in hopper walker2d halfcheetah;
	do
		for buffer in medium-replay medium medium-expert;
		do
			end=$((start+n_trials-1))
			job=[${start}-${end}]-${env:0:2}-${buffer}
			echo $job
			sbatch -J $job slurm/sbatch.sh \
				parallel -j 2 -u \
					python -u scripts/plan_guided.py \
						--logbase logs/pretrained \
						--dataset $env-$buffer-v2 \
						--prefix plans/reference_var \
						--vis_freq 500 \
						--verbose False \
						--suffix {1} \
					::: $(seq $start $end)
		done
	done
done
