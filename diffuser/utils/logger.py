import os
import json


class Logger:

    def __init__(self, renderer, logpath, vis_freq=10, max_render=8):
        self.renderer = renderer
        self.savepath = logpath
        self.vis_freq = vis_freq
        self.max_render = max_render

    def log(self, t, samples, state, rollout=None):
        if t % self.vis_freq != 0:
            return

        ## render image of plans
        self.renderer.composite(
            os.path.join(self.savepath, f'{t}.png'),
            samples.observations,
        )

        ## render video of plans
        self.renderer.render_plan(
            os.path.join(self.savepath, f'{t}_plan.mp4'),
            samples.actions[:self.max_render],
            samples.observations[:self.max_render],
            state,
        )

        if rollout is not None:
            ## render video of rollout thus far
            self.renderer.render_rollout(
                os.path.join(self.savepath, f'rollout.mp4'),
                rollout,
                fps=80,
            )

    def finish(self, t, score, total_reward, terminal, diffusion_experiment, value_experiment):
        json_path = os.path.join(self.savepath, 'rollout.json')
        json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': diffusion_experiment.epoch, 'epoch_value': value_experiment.epoch}
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
        print(f'[ utils/logger ] Saved log to {json_path}')
