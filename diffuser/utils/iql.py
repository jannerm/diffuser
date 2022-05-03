import os
import numpy as np
import jax
import jax.numpy as jnp
import functools
import pdb

from diffuser.iql.common import Model
from diffuser.iql.value_net import DoubleCritic

def load_q(env, loadpath, hidden_dims=(256, 256), seed=42):
    print(f'[ utils/iql ] Loading Q: {loadpath}')
    observations = env.observation_space.sample()[np.newaxis]
    actions = env.action_space.sample()[np.newaxis]

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)

    critic_def = DoubleCritic(hidden_dims)
    critic = Model.create(critic_def,
                          inputs=[key, observations, actions])

    ## allows for relative paths
    loadpath = os.path.expanduser(loadpath)
    critic = critic.load(loadpath)
    return critic

class JaxWrapper:

    def __init__(self, env, loadpath, *args, **kwargs):
        self.model = load_q(env, loadpath)

    @functools.partial(jax.jit, static_argnames=('self'), device=jax.devices('cpu')[0])
    def forward(self, xs):
        Qs = self.model(*xs)
        Q = jnp.minimum(*Qs)
        return Q

    def __call__(self, *xs):
        Q = self.forward(xs)
        return np.array(Q)
