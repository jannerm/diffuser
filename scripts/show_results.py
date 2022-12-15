import diffuser.utils as utils

loadpath = "/home/admin/matin/diffuser/logs/halfcheetah-medium-expert-v2/diffusion/defaults_H4_T20"

diffusion_experiment = utils.load_diffusion(loadpath)

dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer
model = diffusion_experiment.trainer.ema_model

env = dataset.env
obs = env.reset()

observations = utils.colab.run_diffusion(model, dataset, obs, 4, 'cuda')

sample = observations[-1]
utils.colab.show_sample(renderer, sample)

utils.colab.show_diffusion(renderer, observations[:, :1], substep=1)
