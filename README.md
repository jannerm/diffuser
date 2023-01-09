## Planning with Diffusion for Flexible Behavior Synthesis, in Pytorch

This is the implementation of the robotics experiments for <a href="https://arxiv.org/abs/2205.09991">Planning with Diffusion for Flexible Behavior Synthesis</a> in Pytorch. 



## Usage

First, install and extract the dataset for training and pretrained models from this <a href="https://www.dropbox.com/s/zofqvtkwpmp4v44/metainfo.tar.gz?dl=0">URL</a> in the root directory of the repo.


To train the unconditional diffusion model on the block stacking task, you can use the following command:

```
python scripts/kuka.py
```

You may evaluate the diffusion model on unconditional stacking with

```
python scripts/unconditional_kuka_planning_eval.py
```

or conditional stacking with

```
python scripts/conditional_kuka_planning_eval.py
```

The rewards are not normalized -- you need to divide numbers by 3.0 to get numbers reported in the paper.

## Citations

```bibtex
@inproceedings{janner2022diffuser,
  title = {Planning with Diffusion for Flexible Behavior Synthesis},
  author = {Michael Janner and Yilun Du and Joshua B. Tenenbaum and Sergey Levine},
  booktitle = {International Conference on Machine Learning},
  year = {2022},
}
```

## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote launcher is based on the [trajectory-transformer](https://github.com/jannerm/trajectory-transformer) repo.

