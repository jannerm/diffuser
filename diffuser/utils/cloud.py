import shlex
import subprocess
import pdb


def sync_logs(logdir, bucket, background=False):
	## remove prefix 'logs' on google cloud
	destination = 'logs' + logdir.split('logs')[-1]
	upload_blob(logdir, destination, bucket, background)

def upload_blob(source, destination, bucket, background):
	command = f'gsutil -m -o GSUtil:parallel_composite_upload_threshold=150M rsync -r {source} {bucket}/{destination}'
	print(f'[ utils/cloud ] Syncing bucket: {command}')
	command = shlex.split(command)

	if background:
		subprocess.Popen(command)
	else:
		subprocess.call(command)
