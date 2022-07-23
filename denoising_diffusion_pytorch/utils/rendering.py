import numpy as np
import imageio
import gym
import mujoco_py as mjc
import pdb
import sys

import pybullet as p
from .pybullet_utils import WorldSaver, connect, dump_body, get_pose, set_pose, Pose, \
    Point, set_default_camera, stable_z, \
    BLOCK_URDF, SMALL_BLOCK_URDF, get_configuration, SINK_URDF, STOVE_URDF, load_model, is_placement, get_body_name, \
    disconnect, DRAKE_IIWA_URDF, get_bodies, HideOutput, wait_for_user, KUKA_IIWA_URDF, add_data_path, load_pybullet, \
    LockRenderer, has_gui, draw_pose, draw_global_system, get_all_links, set_color, get_movable_joints, set_joint_position
from imageio import imwrite, get_writer


def sample_placements(body_surfaces, obstacles=None, min_distances={}):
    if obstacles is None:
        obstacles = [body for body in get_bodies() if body not in body_surfaces]
    obstacles = list(obstacles)
    # TODO: max attempts here
    for body, surface in body_surfaces.items():
        min_distance = min_distances.get(body, 0.01)
        while True:
            pose = sample_placement(body, surface)
            if pose is None:
                return False
            if not any(pairwise_collision(body, obst, max_distance=min_distance)
                       for obst in obstacles if obst not in [body, surface]):
                obstacles.append(body)
                break
    return True


def packed(arm='left', grasp_type='top', num=1):
    # TODO: packing problem where you have to place in one direction
    base_extent = 5.0

    base_limits = (-base_extent/2.*np.ones(2), base_extent/2.*np.ones(2))
    block_width = 0.07
    block_height = 0.1
    #block_height = 2*block_width
    block_area = block_width*block_width

    #plate_width = 2*math.sqrt(num*block_area)
    plate_width = 0.27
    #plate_width = 0.28
    #plate_width = 0.3
    print('Width:', plate_width)
    plate_width = min(plate_width, 0.6)
    plate_height = 0.001

    other_arm = get_other_arm(arm)
    initial_conf = get_carry_conf(arm, grasp_type)

    add_data_path()
    floor = load_pybullet("plane.urdf")
    pr2 = create_pr2()
    set_arm_conf(pr2, arm, initial_conf)
    open_arm(pr2, arm)
    set_arm_conf(pr2, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
    close_arm(pr2, other_arm)
    set_group_conf(pr2, 'base', [-1.0, 0, 0]) # Be careful to not set the pr2's pose

    table = create_table()
    plate = create_box(plate_width, plate_width, plate_height, color=GREEN)
    plate_z = stable_z(plate, table)
    set_point(plate, Point(z=plate_z))
    surfaces = [table, plate]

    blocks = [create_box(block_width, block_width, block_height, color=BLUE) for _ in range(num)]
    initial_surfaces = {block: table for block in blocks}

    min_distances = {block: 0.05 for block in blocks}
    sample_placements(initial_surfaces, min_distances=min_distances)

    return pr2


def load_world():
    # TODO: store internal world info here to be reloaded
    set_default_camera()
    draw_global_system()
    with HideOutput():
        #add_data_path()
        robot = load_model(DRAKE_IIWA_URDF, fixed_base=True) # DRAKE_IIWA_URDF | KUKA_IIWA_URDF
        floor = load_model('models/short_floor.urdf', fixed_base=True)
        # sink = load_model(SINK_URDF, pose=Pose(Point(x=-0.5)), fixed_base=True)
        # stove = load_model(STOVE_URDF, pose=Pose(Point(x=+0.5)), fixed_base=True)

        block_0 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_0, (1, 0, 0, 1))

        block_1 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_1, (0, 1, 0, 1))

        block_2 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_2, (0, 0, 1, 1))

        block_3 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        set_color(block_3, (1, 1, 0, 1))

        # block_4 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        # set_color(block_4, (0, 1, 1, 1))

        # block_5 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        # set_color(block_5, (1, 0, 1, 1))

        # block_6 = load_model('models/drake/objects/simple_cuboid.urdf', fixed_base=False)
        # set_color(block_6, (0.5, 0.5, 0.3, 1))

        #cup = load_model('models/dinnerware/cup/cup_small.urdf',
        # Pose(Point(x=+0.5, y=+0.5, z=0.5)), fixed_base=False)

    # draw_pose(Pose(), parent=robot, parent_link=get_tool_link(robot)) # TODO: not working
    # dump_body(robot)
    # wait_for_user()

    body_names = {
        # sink: 'sink',
        # stove: 'stove',
        block_0: 'block_0',
        block_1: 'block_1',
        block_2: 'block_2',
        block_3: 'block_3',
        # block_4: 'block_4',
        # block_5: 'block_5',
        # block_6: 'block_6',
    }
    movable_bodies = [block_0, block_1, block_2, block_3]#, block_4, block_5, block_6]

    for body in movable_bodies:
        # if random.uniform(0, 1) > 0.5:
        #     x = np.random.uniform(0.3, 0.4)
        # else:
        #     x = np.random.uniform(-0.4, -0.3)

        # if random.uniform(0, 1) > 0.5:
        #     y = np.random.uniform(0.3, 0.4)
        # else:
        #     y = np.random.uniform(-0.4, -0.3)
        pt = np.random.uniform(-1, 1, (2,))

        # pt = np.array([x, y])
        norm = np.linalg.norm(pt)
        pt = pt / norm * 0.6
        set_pose(body, Pose(Point(x=pt[0], y=pt[1], z=stable_z(body, floor))))

    # set_pose(radish, Pose(Point(y=-0.5, z=stable_z(radish, floor))))

    return robot, body_names, movable_bodies

class Renderer:

    def __init__(self, env):
        if type(env) is str:
            self.env = gym.make(env)
        else:
            self.env = env
        self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)

    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None):

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }

        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        if not qvel:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        if type(dim) == int:
            dim = (dim, dim)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def composite(self, savepath, *args, **kwargs):
        sample_images = self.renders(*args, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        if savepath is not None:
            imageio.imsave(savepath, composite)

        return composite

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)


class TampRenderer:

    def __init__(self):
        connect()
        robot = packed(num=5)
        self.robot = robot
        self.cubes = [4, 5, 6, 7, 8]
        near = 0.001
        far = 4.0
        projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

        location = np.array([1.0, 1.0, 2.0])
        end = np.array([0.0, 0.0, 1.0])
        viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

        self.projectionMatrix = projectionMatrix
        self.viewMatrix = viewMatrix


    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None):
        joints = get_movable_joints(self.robot)
        joint_vals = observation[:28]

        for i, cube in enumerate(self.cubes):
            pos = observation[28+i*7:31+i*7]
            rot = observation[31+i*7:35+i*7]
            set_pose(cube, (pos, rot))

        for joint, joint_val in zip(joints, joint_vals):
            set_joint_position(self.robot, joint, joint_val)

        projectionMatrix = self.projectionMatrix
        viewMatrix = self.viewMatrix
        _, _, im, _, seg = p.getCameraImage(width=256, height=256, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        im = np.array(im).reshape((256, 256, 4))[:, :, :3]

        return im

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def composite(self, savepath, *args, **kwargs):
        sample_images = self.renders(*args, **kwargs)
        # writer = get_writer(savepath)
        # composite = np.concatenate(sample_images, axis=1)

        if savepath is not None:
            imageio.imsave(savepath, composite)

        return sample_images

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)


class KukaRenderer:

    def __init__(self):
        connect()
        robot, _, movable = load_world()
        self.robot = robot
        self.cubes = movable
        near = 0.001
        far = 4.0
        projectionMatrix = p.computeProjectionMatrixFOV(60., 1.0, near, far)

        location = np.array([0.1, 0.1, 2.0])
        end = np.array([0.0, 0.0, 1.0])
        viewMatrix = p.computeViewMatrix(location, end, [0, 0, 1])

        self.projectionMatrix = projectionMatrix
        self.viewMatrix = viewMatrix


    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None):
        joints = get_movable_joints(self.robot)
        joint_vals = observation[:7]

        for i, cube in enumerate(self.cubes):
            pos = observation[7+i*8:10+i*8]
            rot = observation[10+i*8:14+i*8]
            rot = rot / np.linalg.norm(rot)
            set_pose(cube, (pos, rot))

        for joint, joint_val in zip(joints, joint_vals):
            set_joint_position(self.robot, joint, joint_val)

        projectionMatrix = self.projectionMatrix
        viewMatrix = self.viewMatrix
        _, _, im, _, seg = p.getCameraImage(width=256, height=256, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
        im = np.array(im).reshape((256, 256, 4))[:, :, :3]

        return im

    def renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def composite(self, savepath, *args, **kwargs):
        sample_images = self.renders(*args, **kwargs)
        # writer = get_writer(savepath)
        # composite = np.concatenate(sample_images, axis=1)

        if savepath is not None:
            imageio.imsave(savepath, composite)

        return sample_images

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    assert state.size == qpos_dim + qvel_dim

    env.set_state(state[:qpos_dim], state[qpos_dim:])

def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask
