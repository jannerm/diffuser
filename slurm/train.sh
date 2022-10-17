for env in hopper walker2d halfcheetah;
do
	for buffer in medium-replay medium medium-expert;
	do
		job=diff-${env:0:2}-${buffer}
		echo $job
		sbatch -J $job singularity/sbatch.sh \
			python scripts/train.py \
				--dataset $env-$buffer-v2
	done
done
