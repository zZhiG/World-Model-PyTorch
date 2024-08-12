import os
import shutil
import time
import gym
import numpy as np

from arguments import get_args
from utils.env_util import adjust_action, adjust_obs


class CarRacing_Dataset_Generator:
    def __init__(self, env_name, data_save_path, img_size, sample_epoch, sample_num, render):
        super().__init__()
        self.env_name = env_name
        self.data_save_path = data_save_path
        self.img_size = img_size
        self.sample_epoch = sample_epoch
        self.sample_num = sample_num
        self.render = render

        self.prefix = ''
        filenames = os.path.abspath(__file__).split('\\')
        for f in filenames[:-1]:
            self.prefix += f+'\\'

        self.prefix = os.path.join(self.prefix, data_save_path)

    # Randomize Samples
    def get_random_data(self):
        dataset_save_path = os.path.join(self.prefix, 'random')
        if not os.path.exists(dataset_save_path):
            os.makedirs(dataset_save_path)
        if os.path.exists(dataset_save_path):
            shutil.rmtree(dataset_save_path)
            os.makedirs(dataset_save_path)

        print("Generating data for env {}".format(self.env_name))

        env = gym.make(self.env_name)
        for i in range(self.sample_epoch):
            obs_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []

            observation = env.reset()

            t = 0
            done = False
            while not done and t < self.sample_num:
                action = adjust_action(t, self.sample_num // 2)
                action_sequence.append(action)

                obs = adjust_obs(observation, self.img_size)
                obs_sequence.append(obs)

                observation, r, done, _ = env.step(action)

                if self.render:
                    env.render()  # Visualization
                # time.sleep(1)

                reward_sequence.append(r)
                done_sequence.append(done)

                t += 1

            save_file_name = os.path.join(dataset_save_path, f'epoch_{i+1}.npz')
            np.savez_compressed(save_file_name, obs=obs_sequence, action=action_sequence,
                                reward=reward_sequence, done=done_sequence)

        env.close()


if __name__ == '__main__':
    args = get_args()
    generator = CarRacing_Dataset_Generator(args.env_name, args.data_save_path,
                                            args.img_size, args.sample_epoch, args.sample_num, args.render)
    if args.sample_method == 'random':
        generator.get_random_data()
