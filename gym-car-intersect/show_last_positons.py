from gym_car_intersect.envs.my_env import CarRacing

env = CarRacing(agent = True, num_bots = 1, track_form = 'X',
                write = False, data_path = 'car_racing_positions.csv',
                start_file = False, training_epoch = False)

env.reset()
for _ in range(1000):
    env.training_status()
