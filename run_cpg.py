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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
gait = "WALK"
cpg = HopfNetwork(time_step=TIME_STEP, gait = gait)

TIME_SIMULATION = 3 #was 10s
TEST_STEPS = int(TIME_SIMULATION / (TIME_STEP)) 
t = np.arange(TEST_STEPS)*TIME_STEP

# done[TODO] initialize data structures to save CPG and robot states
#task 1
r_list = np.zeros([TEST_STEPS,4])
theta_list = np.zeros([TEST_STEPS,4])
dr_list = np.zeros([TEST_STEPS,4])
dtheta_list = np.zeros([TEST_STEPS,4])

#task 2
pos_leg_0 = np.zeros([TEST_STEPS,3])
des_pos_leg_0 = np.zeros([TEST_STEPS,3])

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100]) #base : 100 100 100
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3) #base : np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS): #me : on est dans le ref de chanque patte = son point d'attache = 0,0,0
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  # done [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # done [TODO] 
    # Add joint PD contribution to tau for leg i (Equation 4)
    tau +=  kp*(leg_q - q[3*i:3*(i+1)]) + kd*(0 - dq[3*i:3*(i+1)]) # pas sur - [TODO] 

    # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
    # done [TODO]
    J, pos = env.robot.ComputeJacobianAndPosition(i)

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current foot velocity in leg frame (Equation 2)
      v = J@dq[3*i:3*(i+1)]
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T@(kpCartesian@(leg_xyz - pos) + kdCartesian@(0 - v)) #done [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

    #record for task 2
    if i == 0: #record only leg 0
      pos_leg_0[j,:] = pos
      des_pos_leg_0[j,:] = leg_xyz


  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 

  # done [TODO] save any CPG or robot states
  r_list[j,:] = cpg.get_r()
  theta_list[j,:] = cpg.get_theta()
  dr_list[j,:] = cpg.get_dr()
  dtheta_list[j,:] = cpg.get_dtheta()





##################################################### 
# PLOTS
#####################################################
# example

### task 1 ###
# fig, axs = plt.subplots(2,2, figsize = (8,6))
# fig.tight_layout(pad = 3.5, w_pad = 1., h_pad=3.0)
# for k in range(4):
#   # plt.subplot(2,2,k)
#   # plt.plot(t,r_list[:,k], label='lenght', color = "r")
#   # plt.plot(t,theta_list[:,k], label='theta', color = "b")
#   # plt.legend()
#   # plt.title("leg", k)
#   ax = axs[int(np.floor(k/2)), k%2]
#   ax.plot(t,r_list[:,k], label='r', color = "r")
#   ax.plot(t,theta_list[:,k], label=r"$\theta$", color = "b")
#   ax.plot(t,dr_list[:,k], label=r"$\dot{r}$", color = "orange")
#   ax.plot(t,dtheta_list[:,k], label=r"$\dot{\theta}$", color = "g")
#   ax.legend(loc="upper right")
#   ax.set_title(f"leg {k}")
#   ax.set_xlabel("Time")
#   ax.set_ylabel("Amplitude / phase")

# fig.suptitle(r"Parameters r, $\theta$, $\dot{r}$, $\dot{\theta}$ for all four legs in " + gait.lower() + r" mode") #u03B8
# fig.show()
# plt.show()

### task 2 ###
dic = {0:"x", 1:"y", 2:"z"}
fig, axs = plt.subplots(3,1, figsize = (8,6))
fig.tight_layout(pad = 3.5, w_pad = 1., h_pad=3.0)
for k in [0,1,2]:
  # plt.subplot(2,2,k)
  # plt.plot(t,r_list[:,k], label='lenght', color = "r")
  # plt.plot(t,theta_list[:,k], label='theta', color = "b")
  # plt.legend()
  # plt.title("leg", k)
  ax = axs[k%3]
  ax.plot(t,pos_leg_0[:,k], label="real foot position", color = "r")
  ax.plot(t,des_pos_leg_0[:,k], label="desired foot position", color = "b")

  ax.legend(loc="upper right")
  ax.set_title(f"component : {dic[k]}")
  ax.set_xlabel("Time")
  ax.set_ylabel("Position")

  ax.set_xlim([2,3]) #in second

fig.suptitle("Comparison desired vs real foot postion") #u03B8
fig.show()
plt.show()