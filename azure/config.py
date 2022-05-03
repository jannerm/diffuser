import os

def get_docker_username():
   import subprocess
   import shlex
   ps = subprocess.Popen(shlex.split('docker info'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   output = subprocess.check_output(shlex.split("sed '/Username:/!d;s/.* //'"), stdin=ps.stdout)
   username = output.decode('utf-8').replace('\n', '')
   print(f'[ azure/config ] Grabbed username from `docker info`: {username}')
   return username

## /path/to/diffuser/azure
CWD = os.path.dirname(__file__)
## /path/to/diffuser
MODULE_PATH = os.path.dirname(CWD)

CODE_DIRS_TO_MOUNT = [
]
NON_CODE_DIRS_TO_MOUNT = [
   dict(
      local_dir=MODULE_PATH,
      mount_point='/home/code',
   ),
]
REMOTE_DIRS_TO_MOUNT = [
    dict(
        local_dir='/doodad_tmp/',
        mount_point='/doodad_tmp/',
    ),
]
LOCAL_LOG_DIR = '/tmp'

DEFAULT_AZURE_GPU_MODEL = 'nvidia-tesla-t4'
DEFAULT_AZURE_INSTANCE_TYPE = 'Standard_DS1_v2'
DEFAULT_AZURE_REGION = 'eastus'
DEFAULT_AZURE_RESOURCE_GROUP = 'diff'
DEFAULT_AZURE_VM_NAME = 'diff-vm'
DEFAULT_AZURE_VM_PASSWORD = 'Azure1'

DOCKER_USERNAME = os.environ.get('DOCKER_USERNAME', get_docker_username())
DEFAULT_DOCKER = f'docker.io/{DOCKER_USERNAME}/diffuser:latest'

print(f'[ azure/config ] Local dir: {MODULE_PATH}')
print(f'[ azure/config ] Default GPU model: {DEFAULT_AZURE_GPU_MODEL}')
print(f'[ azure/config ] Default Docker image: {DEFAULT_DOCKER}')

AZ_SUB_ID = os.environ['AZURE_SUBSCRIPTION_ID']
AZ_CLIENT_ID = os.environ['AZURE_CLIENT_ID']
AZ_TENANT_ID = os.environ['AZURE_TENANT_ID']
AZ_SECRET = os.environ['AZURE_CLIENT_SECRET']
AZ_CONTAINER = os.environ['AZURE_STORAGE_CONTAINER']
AZ_CONN_STR = os.environ['AZURE_STORAGE_CONNECTION_STRING']
