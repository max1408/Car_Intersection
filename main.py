import sys, math
import json
import numpy as np
from scipy.spatial import cKDTree
from collections import deque
import time

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
# from gym_car_intersect.envs.car_model import Car
# from car_model import Car
# from gym_car_intersect.envs.new_car import DummyCar
from new_car import DummyCar
from gym.utils import colorize, seeding, EzPickle

import pyglet
from pyglet import gl

import argparse

# Havely Based on work of CarRacing OpenAI gym by Oleg Klimov.
# Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 600 # Changes: 400
WINDOW_W = 600 # Changes: 1200
WINDOW_H = 600 # Changes: 1000

SCALE       = 1       # Track scale
PLAYFIELD   = 60/SCALE # Game over boundary # Changes: 600
FPS         = 50
ZOOM        = 5 # Changes: 2.7        # Camera zoom
ZOOM_FOLLOW = False      # Set to False for fixed view (don't use zoom)

ROAD_WIDTH = 8/SCALE
SIDE_WALK = 4/SCALE

ROAD_COLOR = [0.44, 0.44, 0.44]

# Creating all possible tragectories:
SMALL_TURN = ROAD_WIDTH*0.5
BIG_TURN = ROAD_WIDTH*1.5
START_1, START_2 = (-ROAD_WIDTH, ROAD_WIDTH), (-ROAD_WIDTH, -ROAD_WIDTH)
START_3, START_4 = (ROAD_WIDTH, -ROAD_WIDTH), (ROAD_WIDTH, ROAD_WIDTH)
OUT_DIST = 12 # how far from the view screen to restart new car
TARGET_2, TARGET_4 = (-PLAYFIELD-OUT_DIST, ROAD_WIDTH/2), (-ROAD_WIDTH/2, -PLAYFIELD-OUT_DIST)
TARGET_6, TARGET_8 = (PLAYFIELD+OUT_DIST, -ROAD_WIDTH/2), (ROAD_WIDTH/2, PLAYFIELD+OUT_DIST)
PATH = {
    '34' : [(START_2[0] + math.cos(rad)*SMALL_TURN, START_2[1] + math.sin(rad)*SMALL_TURN)
                for rad in np.linspace(np.pi/2, 0, 10)] + [TARGET_4],
    '36' : [TARGET_6],
    '38' : [(START_1[0] + math.cos(rad)*BIG_TURN, START_1[1] + math.sin(rad)*BIG_TURN)
                for rad in np.linspace(-np.pi/2, 0, 10)] + [TARGET_8],

    '56' : [(START_3[0] + math.cos(rad)*SMALL_TURN, START_3[1] + math.sin(rad)*SMALL_TURN)
                for rad in np.linspace(np.pi, np.pi/2, 10)] + [TARGET_6],
    '58' : [TARGET_8],
    '52' : [(START_2[0] + math.cos(rad)*BIG_TURN, START_2[1] + math.sin(rad)*BIG_TURN)
                for rad in np.linspace(0, np.pi/2, 10)] + [TARGET_2],

    '78' : [(START_4[0] + math.cos(rad)*SMALL_TURN, START_4[1] + math.sin(rad)*SMALL_TURN)
                for rad in np.linspace(-np.pi/2, -np.pi, 10)] + [TARGET_8],
    '72' : [TARGET_2],
    '74' : [(START_3[0] + math.cos(rad)*BIG_TURN, START_3[1] + math.sin(rad)*BIG_TURN)
                for rad in np.linspace(np.pi/2, np.pi, 10)] + [TARGET_4],

    '92' : [(START_1[0] + math.cos(rad)*SMALL_TURN, START_1[1] + math.sin(rad)*SMALL_TURN)
                for rad in np.linspace(0, -np.pi/2, 10)] + [TARGET_2],
    '94' : [TARGET_4],
    '96' : [(START_4[0] + math.cos(rad)*BIG_TURN, START_4[1] + math.sin(rad)*BIG_TURN)
                for rad in np.linspace(-np.pi, -np.pi/2, 10)] + [TARGET_6],
}

ALL_SECTIONS = set(list(PATH.keys()))
INTERSECT = {
    '34' : {'94', '74'},
    '36' : ALL_SECTIONS - {'72', '78', '92'},
    '38' : ALL_SECTIONS - {'56', '92'},

    '56' : {'36', '96'},
    '58' : ALL_SECTIONS - {'34', '94', '92'},
    '52' : ALL_SECTIONS - {'34', '78'},

    '78' : {'58', '38'},
    '72' : ALL_SECTIONS - {'36', '34', '56'},
    '74' : ALL_SECTIONS - {'56', '92'},

    '92' : {'72', '52'},
    '94' : ALL_SECTIONS - {'58', '56', '78'},
    '96' : ALL_SECTIONS - {'78', '34'},
}

PATH_cKDTree = dict()
for key, value in PATH.items():
    PATH_cKDTree[key] = cKDTree(value)

# Road mark crossings:
CROSS_WIDTH = 4/SCALE
template_v = np.array([(0.8, 0), (0.2, 0), (0.2, CROSS_WIDTH), (0.8, CROSS_WIDTH)])
template_h = np.array([(CROSS_WIDTH, 0.2), (0, 0.2), (0, 0.8), (CROSS_WIDTH, 0.8)])

eps = 0/SCALE
crossing_w = [-template_h + np.array([-ROAD_WIDTH-eps, y]) for y in np.arange(-ROAD_WIDTH+1, ROAD_WIDTH+1)]
crossing_e = [template_h + np.array([ROAD_WIDTH+eps, y]) for y in np.arange(-ROAD_WIDTH, ROAD_WIDTH-0)]
crossing_n = [template_v + np.array([y, ROAD_WIDTH+eps]) for y in np.arange(-ROAD_WIDTH, ROAD_WIDTH-0)]
crossing_s = [-template_v + np.array([y, -ROAD_WIDTH-eps]) for y in np.arange(-ROAD_WIDTH+1, ROAD_WIDTH+1)]
crossings = [crossing_w, crossing_e, crossing_n, crossing_s]

cross_line_w = [(-PLAYFIELD, 0), (-ROAD_WIDTH-CROSS_WIDTH-eps*2, 0)]
cross_line_e = [(ROAD_WIDTH+CROSS_WIDTH+eps*2, 0), (PLAYFIELD, 0)]
cross_line_n = [(0, PLAYFIELD), (0, ROAD_WIDTH+CROSS_WIDTH+eps*2)]
cross_line_s = [(0, -PLAYFIELD), (0, -ROAD_WIDTH-CROSS_WIDTH-eps*2)]

cross_line = [cross_line_w, cross_line_e, cross_line_n, cross_line_s]


class MyContactListener(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)

    @staticmethod
    def _priority_check(path1, path2):
        target1, target2 = path1[0], path2[0]

        if target1 == '3' and target2 in {'3', '5', '7'}:
            return True
        elif target1 == '3':
            return False

        if target1 == '5' and target2 in {'5', '7', '9'}:
            return True
        elif target1 == '5':
            return False

        if target1 == '7' and target2 in {'7', '9'}:
            return True
        elif target1 == '7':
            return False

        if target1 == '9' and target2 in {'9', '3'}:
            return True
        elif target1 == '9':
            return False


    def BeginContact(self, contact):
        # Data to define sensor data:
        sensA = contact.fixtureA.sensor
        sensB = contact.fixtureB.sensor

        # Data to define collisions:
        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData

        # Check data we have for fixtures:
        fixA = contact.fixtureA.userData
        fixB = contact.fixtureB.userData

        # Processing Sensoring:
        if sensA and bodyA.name=='car' and bodyB.name=='road':
            if bodyB.road_section in bodyA.penalty_sec:
                bodyA.penalty = True
        if sensB and bodyB.name=='car' and bodyA.name=='road':
            if bodyA.road_section in bodyB.penalty_sec:
                bodyB.penalty = True

        # Behaviour on crossroads:
        if sensA and bodyA.name=='bot_car' and bodyB.name=='road':
            if bodyB.road_section == 1:
                bodyA.cross_time = time.time()
        if sensB and bodyB.name=='bot_car' and bodyA.name=='road':
            if bodyA.road_section == 1:
                bodyB.cross_time = time.time()

        if sensA and bodyA.name=='bot_car' and (bodyB.name in {'car', 'bot_car'}):
            # if self._priority_check(bodyA.path, bodyB.path):
            if fixB == 'body':
                bodyA.stop = True
        if sensB and bodyB.name=='bot_car' and (bodyA.name in {'car', 'bot_car'}):
            # if self._priority_check(bodyB.path, bodyA.path):
            if fixA == 'body':
                bodyB.stop = True

        # Processing Collision:
        if (bodyA.name in {'car', 'wheel'}) and (bodyB.name in {'car', 'bot_car', 'sidewalk'}):
            if fixB != 'sensor':
                bodyA.collision = True
        if (bodyA.name in {'car', 'bot_car', 'sidewalk'}) and (bodyB.name in {'car', 'wheel'}):
            if fixA != 'sensor':
                bodyB.collision = True

    def EndContact(self, contact):
        sensA = contact.fixtureA.sensor
        sensB = contact.fixtureB.sensor

        bodyA = contact.fixtureA.body.userData
        bodyB = contact.fixtureB.body.userData

        fixA = contact.fixtureA.userData
        fixB = contact.fixtureB.userData

        if sensA and bodyA.name=='car' and bodyB.name=='road':
            if bodyB.road_section in bodyA.penalty_sec:
                bodyA.penalty = False
        if sensB and bodyB.name=='car' and bodyA.name=='road':
            if bodyA.road_section in bodyB.penalty_sec:
                bodyB.penalty = False

        # Behaviour on crossroads:
        if sensA and bodyA.name=='bot_car' and bodyB.name=='road':
            if bodyB.road_section == 1:
                bodyA.cross_time = float('inf')
        if sensB and bodyB.name=='bot_car' and bodyA.name=='road':
            if bodyA.road_section == 1:
                bodyB.cross_time = float('inf')

        if sensA and bodyA.name=='bot_car' and (bodyB.name in {'car', 'bot_car'}):
            # if self._priority_check(bodyA.path, bodyB.path):
            if fixB == 'body':
                bodyA.stop = False
        if sensB and bodyB.name=='bot_car' and (bodyA.name in {'car', 'bot_car'}):
            # if self._priority_check(bodyB.path, bodyA.path):
            if fixA == 'body':
                bodyB.stop = False

        # Processing Collision:
        if (bodyA.name in {'car', 'wheel'}) and (bodyB.name in {'car', 'bot_car', 'sidewalk'}):
            if fixB != 'sensor':
                bodyA.collision = False
        if (bodyA.name in {'car', 'bot_car', 'sidewalk'}) and (bodyB.name in {'car', 'wheel'}):
            if fixA != 'sensor':
                bodyB.collision = False


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self, agent = True, num_bots = 4, track_form = 'X', write = False, data_path = 'car_racing_positions.csv'):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = MyContactListener(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.agent = agent
        self.car = None
        self.bot_cars = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.track_form = track_form
        self.data_path = data_path
        self.write = write

        if write:
            car_title = ['car_angle', 'car_pos_x', 'car_pos_y']
            bots_title = []
            for i in range(num_bots):
                bots_title.extend([f'car_bot{i+1}_angle', f'car_bot{i+1}_pos_x', f'car_bot{i+1}_pos_y'])
            with open(data_path, 'w') as f:
                f.write(','.join(car_title + bots_title))
                f.write('\n')

        # check the target position:
        self.moved_distance = deque(maxlen=1000)
        self.target = (0, 0)
        self.num_bots = num_bots

        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]), dtype=np.float32)  # steer, gas, brake
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road: return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()
        for car in self.bot_cars:
            car.destroy()

    def _create_track(self):
        # Creating Road Sections:
        # See notes:
        road_s1 = [(-ROAD_WIDTH, ROAD_WIDTH), (-ROAD_WIDTH, -ROAD_WIDTH), (ROAD_WIDTH, -ROAD_WIDTH), (ROAD_WIDTH, ROAD_WIDTH)]

        road_s2 = [(-PLAYFIELD, ROAD_WIDTH), (-PLAYFIELD, 0), (-ROAD_WIDTH, 0), (-ROAD_WIDTH, ROAD_WIDTH)]
        road_s3 = [(-PLAYFIELD, 0), (-PLAYFIELD, -ROAD_WIDTH), (-ROAD_WIDTH, -ROAD_WIDTH), (-ROAD_WIDTH, 0)]

        road_s4 = [(-ROAD_WIDTH, -ROAD_WIDTH), (-ROAD_WIDTH, -PLAYFIELD), (0, -PLAYFIELD), (0, -ROAD_WIDTH)]
        road_s5 = [(0, -ROAD_WIDTH), (0, -PLAYFIELD), (ROAD_WIDTH, -PLAYFIELD), (ROAD_WIDTH, -ROAD_WIDTH)]

        road_s6 = [(ROAD_WIDTH, 0), (ROAD_WIDTH, -ROAD_WIDTH), (PLAYFIELD, -ROAD_WIDTH), (PLAYFIELD, 0)]
        road_s7 = [(ROAD_WIDTH, ROAD_WIDTH), (ROAD_WIDTH, 0), (PLAYFIELD, 0), (PLAYFIELD, ROAD_WIDTH)]

        road_s8 = [(0, PLAYFIELD), (0, ROAD_WIDTH), (ROAD_WIDTH, ROAD_WIDTH), (ROAD_WIDTH, PLAYFIELD)]
        road_s9 = [(-ROAD_WIDTH, PLAYFIELD), (-ROAD_WIDTH, ROAD_WIDTH), (0, ROAD_WIDTH), (0, PLAYFIELD)]

        # Creting body of roads:
        self.road_poly = road_s1 + road_s2 + road_s3 + road_s4 + road_s5 + road_s6 + road_s7
        if self.track_form == 'X':
            self.road_poly += road_s8 + road_s9

        self.road = []

        # Creating a static object - road with names for specific section + i need to define how to listen to it:
        for i in range(0, len(self.road_poly), 4):
            r = self.world.CreateStaticBody(
                    fixtures=fixtureDef(shape=polygonShape(vertices=self.road_poly[i:i+4]), isSensor=True))
            r.road_section = int(i/4)+1
            r.name = 'road'
            r.userData = r
            self.road.append(r)

        # Creating sidewalks:
        sidewalk_h_nw = [(-PLAYFIELD, ROAD_WIDTH+SIDE_WALK),
                          (-PLAYFIELD, ROAD_WIDTH),
                          (-ROAD_WIDTH-SIDE_WALK*2, ROAD_WIDTH),
                          (-ROAD_WIDTH-SIDE_WALK*2, ROAD_WIDTH+SIDE_WALK)]
        sidewalk_h_sw = [(x, y-2*ROAD_WIDTH-SIDE_WALK) for x, y in sidewalk_h_nw]
        sidewalk_h_ne = [(x+PLAYFIELD+2*SIDE_WALK+ROAD_WIDTH, y) for x, y in sidewalk_h_nw]
        sidewalk_h_se = [(x, y-2*ROAD_WIDTH-SIDE_WALK) for x, y in sidewalk_h_ne]

        sidewalk_v_nw = [(-ROAD_WIDTH-SIDE_WALK, PLAYFIELD),
                         (-ROAD_WIDTH-SIDE_WALK, ROAD_WIDTH+SIDE_WALK*2),
                         (-ROAD_WIDTH, ROAD_WIDTH+SIDE_WALK*2),
                         (-ROAD_WIDTH, PLAYFIELD)]

        sidewalk_v_sw = [(x, y-PLAYFIELD-2*SIDE_WALK-ROAD_WIDTH) for x, y in sidewalk_v_nw]
        sidewalk_v_ne = [(x+2*ROAD_WIDTH+SIDE_WALK, y) for x, y in sidewalk_v_nw]
        sidewalk_v_se = [(x+2*ROAD_WIDTH+SIDE_WALK, y) for x, y in sidewalk_v_sw]

        # sidewalk_c_nw = [(-ROAD_WIDTH-2*SIDE_WALK+np.cos(rad)*1*SIDE_WALK, ROAD_WIDTH+2*SIDE_WALK+np.sin(rad)*1*SIDE_WALK)
        #             for rad in np.linspace(-np.pi/2, 0, 12)]
        sidewalk_c_nw = sidewalk_v_nw[2:0:-1] + sidewalk_h_nw[3:1:-1] + [(-ROAD_WIDTH, ROAD_WIDTH)]
        sidewalk_c_ne = sidewalk_h_ne[1::-1] + sidewalk_v_ne[2:0:-1] + [(ROAD_WIDTH, ROAD_WIDTH)]
        sidewalk_c_sw = sidewalk_h_sw[3:1:-1] + sidewalk_v_sw[::3] + [(-ROAD_WIDTH, -ROAD_WIDTH)]
        sidewalk_c_se = sidewalk_v_se[::3] + sidewalk_h_se[1::-1] + [(ROAD_WIDTH, -ROAD_WIDTH)]

        self.all_sidewalks = [sidewalk_h_nw, sidewalk_h_ne, sidewalk_h_se, sidewalk_h_sw,
                              sidewalk_v_nw, sidewalk_v_ne, sidewalk_v_se, sidewalk_v_sw,
                              sidewalk_c_nw, sidewalk_c_ne, sidewalk_c_se, sidewalk_c_sw]

        #Now let's see the static world:
        sidewalk = self.world.CreateStaticBody(
            fixtures = [fixtureDef(shape=polygonShape(vertices=sw), isSensor=True) for sw in self.all_sidewalks])
        sidewalk.name = 'sidewalk'
        sidewalk.userData = sidewalk

        # remove===============================================================
        # w = self.world.CreateStaticBody(fixtures = fixtureDef(shape=polygonShape(box=(10,10, (20,20), 0))))
        # w.name = 'car'
        # w.userData = w

        return True

    def random_position(self, forward_shift=0, bot=True, exclude=None):
        # creating list to diminish number of trajectories to choose from:
        target_set = set(PATH.keys()) if exclude is None else set(PATH.keys()) - exclude
        if len(target_set) == 0:
            print("No more places where to put car! Consider to decrease the number.")

        target = np.random.choice(list(target_set))
        if exclude is None:
            exclude = {target}
        else:
            exclude.add(target)

        # if some cars on the same trajectory add distance between them
        space = 6*self.bot_targets.count(target[0]) - 3 - forward_shift

        if target[0] == '3':
            new_position = (-np.pi/2, -PLAYFIELD-space, -ROAD_WIDTH/2)
        if target[0] == '5':
            new_position = (0, ROAD_WIDTH/2, -PLAYFIELD-space)
        if target[0] == '7':
            new_position = (np.pi/2, PLAYFIELD+space, ROAD_WIDTH/2)
        if target[0] == '9':
            new_position = (np.pi, -ROAD_WIDTH/2, PLAYFIELD+space)

        if not bot:
            _, x, y = new_position
            if abs(x) > PLAYFIELD-5 or abs(y) > PLAYFIELD-5:
                return self.random_position(forward_shift, bot, exclude=exclude)

        for car in self.bot_cars:
            if car.close_to_target(new_position[1:], dist=3):
                # print(f"car target: {car.hull.path} and new_coord: {target}")
                # print(f"car position: {car.hull.position} and new_coord: {new_position[1:]}")
                # Choose new position:
                return self.random_position(forward_shift, bot, exclude=exclude)

        return target, new_position

    def _to_file(self):
        car_position = [self.car.hull.angle,
                        self.car.hull.position.x,
                        self.car.hull.position.y]

        bots_position = []
        for car in self.bot_cars:
            bots_position.extend([car.hull.angle,
                                  car.hull.position.x,
                                  car.hull.position.y])

        with open(self.data_path, 'a') as fout:
            fout.write(','.join(list(map(str, car_position + bots_position))))
            fout.write('\n')

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False
        self.moved_distance.clear()

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")

        # Generate Bot Cars:
        self.bot_cars = []
        self.bot_targets = []
        init_coord = [(-np.pi/2, -PLAYFIELD+15, -ROAD_WIDTH/2),
                      (0, ROAD_WIDTH/2, -PLAYFIELD+20),
                      (np.pi/2, PLAYFIELD-30, ROAD_WIDTH/2),
                      (np.pi, -ROAD_WIDTH/2, PLAYFIELD-15)]

        # init_colors = [(0.8, 0.4, 1), (1, 0.5, 0.1), (0.1, 1, 0.1), (0.2, 0.8, 1)]
        # trajectory = ['38', '52', '74', '96']
        # self.bot_targets.extend([t[0] for t in trajectory])
        for _ in range(self.num_bots):
            target, new_coord = self.random_position(forward_shift=25)
            self.bot_targets.append(target[0])
            car = DummyCar(self.world, new_coord, color=None, bot=True)
            # j = 2*i+4 if 2*i+4 < 9 else 2
            car.hull.path = target #f"{2*i+3}{j}"
            car.userData = self.car
            self.bot_cars.append(car)

        # for i in range(8):
        #     color = np.random.rand(3)
        #     target, new_coord = self.random_position(forward_shift=15)
        #     self.bot_targets.append(target[0])
        #     car = DummyCar(self.world, new_coord, color=color, bot=True)
        #     car.hull.path = target
        #     car.userData = self.car
        #     self.bot_cars.append(car)

        # Generate Agent:
        if not self.agent:
            init_coord = (0, PLAYFIELD+2, PLAYFIELD)
            target = np.random.choice(list(PATH.keys()))
        else:
            target, init_coord = self.random_position(forward_shift=25, bot=False)
        # target, init_coord = '38', (-np.pi/2, -PLAYFIELD+15, -ROAD_WIDTH/2+2)
        penalty_sections = {2, 3, 4, 5, 6, 7, 8, 9} - set(map(int, target))
        self.car = DummyCar(self.world, init_coord, penalty_sections)
        self.car.hull.path = target
        self.car.userData = self.car
        self.moved_distance.append([self.car.hull.position.x, self.car.hull.position.y])

        if self.write:
            self._to_file()

        return self.step(None)[0]

    def step(self, action):
        # self.car.go_to_target(CarPath)
        if action is not None:
            self.car.steer(action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        prev_stop_values = [] # keep values of bot_cars for stop on cross ans restore them before exiting:
        first_cross = self.bot_cars[np.argmin(list(x.hull.cross_time for x in self.bot_cars))]
        min_cross = np.min(list(x.hull.cross_time for x in self.bot_cars))
        active_path = set(x.hull.path for x in self.bot_cars if x.hull.cross_time != float('inf'))
        for i, car in enumerate(self.bot_cars):
            prev_stop_values.append(car.hull.stop)

            if car.hull.cross_time != float('inf') and car.hull.cross_time > min_cross:
                if len(INTERSECT[car.hull.path] & active_path) != 0:
                    car.hull.stop = True

            if car.hull.stop:
                car.brake(0.8)
            else:
                car.go_to_target(PATH[car.hull.path], PATH_cKDTree[car.hull.path])
                # Check if car is outside of field (close to target)
                # and then change it position
                if car.close_to_target(PATH[car.hull.path][-1]):
                    self.bot_targets[i] = '0'
                    target, new_coord = self.random_position()
                    # new_color = np.random.rand(3) #car.hull.color
                    new_color = car.hull.color
                    new_car = DummyCar(self.world, new_coord, color=new_color, bot=True)
                    new_car.hull.path = target
                    new_car.userData = new_car
                    self.bot_cars[i] = new_car
                    self.bot_targets[i] = target[0]

            car.step(1.0/FPS)

        # Returning previous values of stops:
        for i, car in enumerate(self.bot_cars):
            car.hull.stop = prev_stop_values[i]

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.moved_distance.append([self.car.hull.position.x, self.car.hull.position.y])
        self.t += 1.0/FPS

        if self.write:
            self._to_file()

        self.state = self.render("state_pixels")

        # collision
        # basically i found each bos2d body in self.car and for each put listener in userData.collision:
        # if self.car.hull.collision:
        #     print('Collision')
        #
        # if self.car.hull.penalty:
        #     print('Penalty')

        # Reward:
        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 1
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD+5 or abs(y) > PLAYFIELD+5:
                done = True
                step_reward += -100
            if self.car.close_to_target(PATH[self.car.hull.path][-1], dist=10):
                done = True
                step_reward += 1000
            if self.car.hull.collision:
                done = True
                step_reward += -100
            if np.any([w.collision for w in self.car.wheels]):
                done = True
                step_reward += -100
            if self.car.hull.penalty:
                # done = True
                step_reward += -20
            if np.linalg.norm(self.car.hull.linearVelocity) < 1:
                step_reward += -20
            if len(self.moved_distance) == self.moved_distance.maxlen:
                prev_pos = np.array(self.moved_distance[0])
                curr = np.array(self.moved_distance[-1])
                if np.linalg.norm(prev_pos - curr) < 1:
                    done = True
                    step_reward += -20

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = ZOOM*SCALE #0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = 0 #self.car.hull.position[0] #0
        scroll_y = 0 #self.car.hull.position[1] #-30
        angle = 0 #-self.car.hull.angle #0
        vel = 0 #self.car.hull.linearVelocity #0
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(WINDOW_W/2, WINDOW_H/2)
        # self.transform.set_translation(
        #     WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
        #     WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        # self.transform.set_rotation(angle)

        self.car.draw(self.viewer)
        for car in self.bot_cars:
            car.draw(self.viewer)

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode=="rgb_array" or mode=="state_pixels":
            win.clear()
            t = self.transform
            self.transform.set_translation(0, 0)
            self.transform.set_scale(0.0167, 0.0167)
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = WINDOW_W//2 #STATE_W
                VP_H = WINDOW_H//2 #STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            # self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            # self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        # Painting entire field in white:
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0, 0.5, 0, 1.0) #gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, PLAYFIELD, 0)
        gl.glVertex3f(PLAYFIELD, PLAYFIELD, 0)
        gl.glVertex3f(PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        # Drawing road:
        gl.glColor4f(*ROAD_COLOR, 1)
        for poly in self.road_poly:
            gl.glVertex3f(*poly, 0)
        gl.glEnd()

        # Drawing a sidewalk:
        gl.glColor4f(0.66, 0.66, 0.66, 1)
        for sw in self.all_sidewalks:
            gl.glBegin(gl.GL_POLYGON)
            for v in sw:
                gl.glVertex3f(*v, 0)
            gl.glEnd()

        # Drawing road crossings:
        gl.glColor4f(1, 1, 1, 1)
        for cros in crossings:
            for temp in cros:
                gl.glBegin(gl.GL_QUADS)
                for v in temp:
                    gl.glVertex3f(*v, 0)
                gl.glEnd()
        for line in cross_line:
            gl.glBegin(gl.GL_LINES)
            for v in line:
                gl.glVertex3f(*v, 0)
            gl.glEnd()

        # # Drawing a rock:
        # gl.glBegin(gl.GL_QUADS)
        # gl.glColor4f(0,0,0,1)
        # gl.glVertex3f(10, 30, 0)
        # gl.glVertex3f(10, 10, 0)
        # gl.glVertex3f(30, 10, 0)
        # gl.glVertex3f(30, 30, 0)
        # gl.glEnd()

        # Drawing target destination for car:
        target_vertices = {
            '2': [(-PLAYFIELD, ROAD_WIDTH), (-PLAYFIELD, 0), (-PLAYFIELD+3, 0), (-PLAYFIELD+3, ROAD_WIDTH)],
            '4': [(-ROAD_WIDTH, -PLAYFIELD), (0, -PLAYFIELD), (0, -PLAYFIELD+3), (-ROAD_WIDTH, -PLAYFIELD+3)],
            '6': [(PLAYFIELD, -ROAD_WIDTH), (PLAYFIELD, 0), (PLAYFIELD-3, 0), (PLAYFIELD-3, -ROAD_WIDTH)],
            '8': [(ROAD_WIDTH, PLAYFIELD), (0, PLAYFIELD), (0, PLAYFIELD-3), (ROAD_WIDTH, PLAYFIELD-3)]
        }

        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(*self.car.hull.color, 1)
        for v in target_vertices[self.car.hull.path[1]]:
            gl.glVertex3f(*v, 0)
        gl.glEnd()

        # # Drawing car pathes:
        # gl.glPointSize(5)
        # for car in self.bot_cars:
        #     gl.glBegin(gl.GL_POINTS)
        #     gl.glColor4f(*car.hull.color, 1)
        #     gl.glVertex3f(*car.target, 0)
        #     gl.glEnd()
        #
        #     gl.glBegin(gl.GL_LINES)
        #     for v in PATH[car.hull.path]:
        #         gl.glVertex3f(*v, 0)
        #     gl.glEnd()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots_number", type=int, default=4, help="Number of bot cars in environment.")
    parser.add_argument("--write", default=False, action="store_true", help="Whehter write cars' coord to file.")
    parser.add_argument("--dir", default='car_racing_positions.csv', help="Dir of csv file with car's coord.")
    parser.add_argument("--no_agent", default=True, action="store_false", help="Wehter show an agent or not")
    args = parser.parse_args()

    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )
    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    env = CarRacing(agent=args.no_agent, num_bots=args.bots_number, write=args.write, data_path=args.dir)
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.close()
