import gym

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('diffuser.environments.hopper:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('diffuser.environments.walker2d:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()