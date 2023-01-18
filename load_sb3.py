# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

TIME_STEP = 0.01
LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '011823182712163148'

custom = False
if custom:
    log_dir = "./model_result/"
    log_dir = log_dir + "run6_thibaud_25_11_SAC"

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['motor_control_mode']= "CPG"
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
# env_config['competition_env'] = True

#added
env_config["task_env"] = "LR_COURSE_TASK"       
env_config["observation_space_mode"] = "LR_SPEED"
# env_config["target_speed"] = [0.5]
#env_config["des_vel_x_input"] = 0.5,
#env_config["des_vel_x_max"] = 1,

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

N = 1000

# [TODO] initialize arrays to save data from simulation 
pos_list = np.zeros((N,3))
vel_list = np.zeros([N,3])
power_list = np.zeros([N])
cot_list = np.zeros([N])
ctr_step_dur = 0
start_swing = 0
end_stance = 0
end_swing = 0

for i in range(N):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    p = env.envs[0].env.robot.GetBasePosition()
    pos_list[i,:] = p

    ### task 4 ###
    #record velocity
    vel_list[i,:] = env.envs[0].env.robot.GetBaseLinearVelocity()
    power_list[i] = np.abs(env.envs[0].env.robot.GetMotorTorques().dot(env.envs[0].env.robot.GetMotorVelocities()))
    cot_list[i] = power_list[i]/(sum(env.envs[0].env.robot.GetTotalMassFromURDF())*9.81*np.linalg.norm(vel_list[i,:]))

    #record swing and stange duration
    if np.sin(env.envs[0].env._cpg.get_theta()[0]) <= 0 and ctr_step_dur == 0:
        ctr_step_dur += 1
    if np.sin(env.envs[0].env._cpg.get_theta()[0]) >= 0 and ctr_step_dur == 1:
        start_swing = i
        ctr_step_dur += 1
    if np.sin(env.envs[0].env._cpg.get_theta()[0]) <= 0 and ctr_step_dur == 2:
        end_swing = i
        start_stance = i
        ctr_step_dur += 1
    if np.sin(env.envs[0].env._cpg.get_theta()[0]) >= 0 and ctr_step_dur == 3:
        end_stance = i
        ctr_step_dur += 1
    
# [TODO] make plots:
dic = {0:"x", 1:"y", 2:"z"}
fig, axs = plt.subplots(3,1, figsize = (10,6))
fig.tight_layout(pad = 3.5, w_pad = 1., h_pad=3.0)
for k in range(3):
  # plt.subplot(2,2,k)
  # plt.plot(t,r_list[:,k], label='lenght', color = "r")
  # plt.plot(t,theta_list[:,k], label='theta', color = "b")
  # plt.legend()
  # plt.title("leg", k)
  ax = axs[k]
  ax.plot(pos_list[:,k], label='position', color = "r")
  ax.legend(loc="upper right")
  ax.set_title(f"{dic[k]} - position")
  ax.set_xlabel("Time")
  ax.set_ylabel("Position")

### task 4 ###
avgvel = vel_list[int(0.5/TIME_STEP):, 0].mean()
avgCOT = cot_list.mean()*10
# fig, ax = plt.subplots(1,1, figsize = (8,6))
# ax.plot(t,cot_list)
# fig.show()
# plt.show()
print("################################################")
print("avg vel : ", avgvel)
print("total step duration (duty cycle) : ",(end_stance - start_swing)*TIME_STEP)
print("stance duration : ",(end_stance - end_swing)*TIME_STEP)
print("swing duration : ",(end_swing - start_swing)*TIME_STEP)
print("duty ratio (stance/swing): ", (end_stance - start_stance)/(end_swing - start_swing))
print("avg COT : ", avgCOT)
print("################################################")

fig.suptitle("Position of robot") #u03B8
fig.show()
plt.show()


