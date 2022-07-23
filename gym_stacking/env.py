import gym
from gym import spaces, logger
from gym.utils import seeding
from easydict import EasyDict
import random

from .utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, SMALL_BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
    LockRenderer, has_gui, draw_pose, draw_global_system, get_all_links, set_color, get_movable_joints, get_joint_position, set_joint_position, reset_simulation, add_fixed_constraint, remove_fixed_constraint, connect, \
    enable_gravity, get_joint_positions, set_client, get_aabb, aabb_contains_aabb, aabb_intersection

import pybullet as p
import numpy as np
from d4rl import offline_env


def load_world():
    # TODO: store internal world info here to be reloaded
    set_default_camera()
    draw_global_system()
    with HideOutput():
        robot = load_model(DRAKE_IIWA_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = load_model('models/short_floor.urdf', fixed_base=True)

        block_0 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_0, (1, 0, 0, 1))

        block_1 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_1, (0, 1, 0, 1))

        block_2 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_2, (0, 0, 1, 1))

        block_3 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_3, (1, 1, 0, 1))


    body_names = {
        block_0: 'block_0',
        block_1: 'block_1',
        block_2: 'block_2',
        block_3: 'block_3',
    }
    movable_bodies = [block_0, block_1, block_2, block_3]

    construct_blocks = []

    for body in movable_bodies:
        while True:
            pt = np.random.uniform(-1, 1, (2,))

            norm = np.linalg.norm(pt)
            pt = pt / norm * 0.6

            in_collision = False

            set_pose(body, Pose(Point(x=pt[0], y=pt[1], z=stable_z(body, floor))))

            body_aabb = get_aabb(body)

            for intersect_block in construct_blocks:
                intersect_body_aabb = get_aabb(intersect_block)

                if aabb_intersection(body_aabb, intersect_body_aabb) is not None:
                    in_collision = True
                    break

            if not in_collision:
                construct_blocks.append(body)
                break

    return robot, body_names, movable_bodies


def get_env_state(robot, cubes, attachments):
    joints = get_movable_joints(robot)
    joint_pos = get_joint_positions(robot, joints)

    for i, cube in enumerate(cubes):
        pos, rot = get_pose(cube)
        pos, rot = np.array(pos), np.array(rot)

        if attachments[i] > 0.5:
            attach = np.ones(1)
        else:
            attach = np.zeros(1)

        joint_pos = np.concatenate([joint_pos, pos, rot, attach], axis=0)

    return joint_pos



class StackEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -2.4                    2.4
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.209 rad (-12 deg)    0.209 rad (12 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, conditional=False, rearrangment=False, dataset_url=None):
        self.conditional = conditional
        self.rearrangment = rearrangment
        self.action_space = spaces.Box(-1 * np.ones(11), 1 * np.ones(11))
        self.dataset_url = dataset_url
        self.spec = "stacking"

        if self.conditional:
            self.observation_space = spaces.Box(-1 * np.ones(43), 1 * np.ones(43))
        elif self.rearrangment:
            self.observation_space = spaces.Box(-1 * np.ones(43), 1 * np.ones(43))
        else:
            self.observation_space = spaces.Box(-1 * np.ones(39), 1 * np.ones(39))

        near = 0.001
        far = 4.0
        projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

        location = np.array([1.0, 1.0, 2.0])
        end = np.array([0.0, 0.0, 1.0])
        viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

        self.client = connect()
        set_client(self.client)

        self.projectionMatrix = projectionMatrix
        self.viewMatrix = viewMatrix

        self.max_length = 128 * 3

        self.state = None
        self.ref_max_score = 3.0
        self.ref_min_score = 0.0
        self.reset()

    def attach_link(self):
        link = 8

        for i in range(4):
            attach = self.attachments[i]

            if attach > 0.5:
                add_fixed_constraint(self.cubes[i], self.robot, link)
            else:
                remove_fixed_constraint(self.cubes[i], self.robot, link)

    def update_joint_positions(self, joint_delta):
        robot = self.robot
        joints = get_movable_joints(robot)
        joint_pos = get_joint_positions(robot, joints)

        for joint, joint_val, joint_delta in zip(joints, joint_pos, joint_delta):
            set_joint_position(robot, joint, joint_val + joint_delta)

    def get_state(self):
        block = self.goal[self.progress]
        block_id = np.eye(4)[block]
        state = get_env_state(self.robot, self.cubes, self.attachments)

        if self.conditional:
            state = np.concatenate([state, block_id], axis=-1)
        elif self.rearrangment:
            state = np.concatenate([state, block_id], axis=-1)

        self.state = state

        return state

    def compute_value(self, stack_map):
        val = stack_map.reshape((4, 3)).sum(axis=-1).max()
        return val

    def compute_reward(self):
        state = self.state
        reward = 0
        counter = 0

        stack_map = self.stack_map.copy()
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue

                attach_i = state[14+i*8] > 0.5
                attach_j = state[14+i*8] > 0.5
                pos_i = state[7+i*8:10+i*8]
                pos_j = state[7+j*8:10+j*8]

                pos_stack = np.linalg.norm(pos_i[:2] - pos_j[:2], axis=-1) < 0.2
                height_stack = pos_i[..., 2] - pos_j[..., 2] > 0.05

                stack = pos_stack & height_stack & (not attach_i) & (not attach_j)

                if stack and (self.stack_map[counter] == 0):
                    if self.conditional:
                        stack_top = self.goal[self.progress+1]
                        stack_bot = self.goal[self.progress]

                        if i == stack_top and j == stack_bot:
                            self.progress = min(self.progress + 1, 2)
                            self.stack_map[counter] = 1
                            reward = reward + 1
                    elif self.rearrangment:
                        stack_top, stack_bot = self.stacks[self.progress]

                        if i == stack_top and j == stack_bot:
                            self.progress = min(self.progress + 1, len(self.stacks) - 1)
                            self.stack_map[counter] = 1
                            reward = reward + 3 / float(len(self.stacks))
                    else:
                        self.stack_map[counter] = 1
                        reward = reward + 1

                counter = counter + 1

        if (not self.conditional) and (not self.rearrangment):
            reward = self.compute_value(self.stack_map) - self.compute_value(stack_map)

        return reward

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        joint_delta = action[:7]
        deltas = action[7:11]
        self.attachments = np.clip(self.attachments + deltas, 0, 1)

        self.attach_link()
        self.update_joint_positions(joint_delta)

        for i in range(10):
            p.stepSimulation()

        if self.step_counter > self.max_length:
            done = True
        else:
            done = False

        self.state = self.get_state()
        reward = self.compute_reward()

        self.step_counter = self.step_counter + 1

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self, seed=None):
        reset_simulation()
        robot, body, body_names = load_world()
        self.robot = robot
        self.cubes = body_names
        self.attachments  = np.zeros(4)

        enable_gravity()
        self.step_counter = 0
        self.stack_map = np.zeros(12)

        # Goal for conditional stack
        self.goal = np.random.permutation(4)
        self.progress = 0

        # Goal for rearrangment
        self.stacks = []
        top_cubes = list(range(4))
        bottom_cubes = list(range(4))

        while len(top_cubes) > 0:
            # Pick cube to on bottom of stack
            bottom_cube = random.choice(bottom_cubes)

            # Pick cube on top of stack
            rix = random.randint(0, len(top_cubes) - 1)
            top_cube = top_cubes.pop(rix)

            bottom_cubes.remove(bottom_cube)

            if bottom_cube in top_cubes:
                top_cubes.remove(bottom_cube)

            if bottom_cube == top_cube:
                top_cubes.append(top_cube)
                bottom_cubes.append(bottom_cube)
            else:
                self.stacks.append((top_cube, bottom_cube))

        self.state = self.get_state()

        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        projectionMatrix = self.projectionMatrix
        viewMatrix = self.viewMatrix

        _, _, im, _, _ = p.getCameraImage(width=screen_width, height=screen_height, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        im = np.array(im).reshape((256, 256, 4))[:, :, :3]

        return im


class OfflineStackEnv(StackEnv, offline_env.OfflineEnv):
    def __init__(self, conditional=False, rearrangment=False, **kwargs):
        StackEnv.__init__(self, conditional=conditional, rearrangment=rearrangment)
        offline_env.OfflineEnv.__init__(self, **kwargs)

def get_stacking_env(conditional=False, rearrangment=False, dataset_url="", **kwargs):
    env = OfflineStackEnv(conditional=conditional, rearrangment=rearrangment, dataset_url=dataset_url, **kwargs)
    return env
