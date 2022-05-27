docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$PWD/logs,target=/logs/docker \
    --mount type=bind,source=/Ship03/dataset/d4rl,target=/root/.d4rl \
    diffuser 
    # bash -c \
    # "export PYTHONPATH=$PYTHONPATH:/home/code && \
    # python /home/code/scripts/train.py --dataset hopper-medium-expert-v2 --logbase logs/docker"