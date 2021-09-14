# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:37:13 2021

@author: Usuari

5. Watch a Smart Agent!
In the next code cell, you will load the trained weights from file to watch a smart agent!
"""
import envs
from buffer import ReplayBuffer, ReplayBuffer_SummTree
from maddpg import MADDPG
from matd3_bc import MATD3_BC
import torch
import numpy as np
from tensorboardX import SummaryWriter
import os
from utilities import transpose_list, transpose_to_tensor, circle_path, random_levy
import time
import copy
import matplotlib.pyplot as plt

# for saving gif
import imageio

BUFFER_SIZE =   4000 # int(1e6) # Replay buffer size
BATCH_SIZE  =   32 #512      # Mini batch size
GAMMA       =   0.99 #0.95     # Discount factor
TAU         =   0.01     # For soft update of target parameters 
LR_ACTOR    =   1e-3     # Learning rate of the actor
LR_CRITIC   =   1e-3     # Learning rate of the critic
WEIGHT_DECAY =  0 #1e-5     # L2 weight decay
UPDATE_EVERY =  30       # How many steps to take before updating target networks
UPDATE_TIMES =  20       # Number of times we update the networks
SEED = 181299   #198                # Seed for random numbers
BENCHMARK   =   True
EXP_REP_BUF =   False     # Experienced replay buffer activation
PRE_TRAINED =   True    # Use a previouse trained network as imput weights
#Scenario used to train the networks
# SCENARIO    =   "simple_track_ivan" 
SCENARIO    =   "dynamic_track_ivan" 
RENDER = True #in BSC machines the render doesn't work
PROGRESS_BAR = True #if we want to render the progress bar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #To run the pytorch tensors on cuda GPU
HISTORY_LENGTH = 5
# DNN = 'MADDPG'
DNN = 'MATD3_BC'

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    seeding(seed = SEED)
    # number of parallel agents
    parallel_envs = 1
    # number of agents per environment
    num_agents = 1
    # number of landmarks (or targets) per environment
    num_landmarks = 1
    
    # initialize environment
    torch.set_num_threads(parallel_envs)
    env = envs.make_parallel_env(parallel_envs, SCENARIO, seed = SEED, num_agents=num_agents, num_landmarks=num_landmarks, benchmark = BENCHMARK)
       
    # initialize policy and critic
    if DNN == 'MADDPG':
            maddpg = MADDPG(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE)
    elif DNN == 'MATD3_BC':
            maddpg = MATD3_BC(num_agents = num_agents, num_landmarks = num_landmarks, discount_factor=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, weight_decay=WEIGHT_DECAY, device = DEVICE)
    else:
        print('ERROR UNKNOWN DNN ARCHITECTURE')
    agents_reward = []
    for n in range(num_agents):
        agents_reward.append([])
    
    if PRE_TRAINED == True:
     
        # New corrected reward:
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091321_192609\model_dir\episode-200000.pt' #Test 59, MADDPG
        # trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091321_202342\model_dir\episode-50000.pt' #Test 59, TD3_BD.
        trained_checkpoint = r'E:\Ivan\UPC\GitHub\logs\091421_070103\model_dir\episode-200000.pt' #Test 67, TD3_BD.
        
        aux = torch.load(trained_checkpoint)
        for i in range(num_agents):  
            if DNN == 'MADDPG':
                maddpg.maddpg_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                maddpg.maddpg_agent[i].critic.load_state_dict(aux[i]['critic_params'])
            elif DNN == 'MATD3_BC':
                maddpg.matd3_bc_agent[i].actor.load_state_dict(aux[i]['actor_params'])
                maddpg.matd3_bc_agent[i].critic.load_state_dict(aux[i]['critic_params'])
            else:
                break
    
    #Reset the environment
    all_obs = env.reset() 
    # flip the first two indices
    obs_roll = np.rollaxis(all_obs,1)
    obs = transpose_list(obs_roll)
    
    #Initialize history buffer with 0.
    obs_size = obs[0][0].size
    history = copy.deepcopy(obs)
    for n in range(parallel_envs):
        for m in range(num_agents):
            for i in range(HISTORY_LENGTH-1):
                if i == 0:
                    history[n][m] = history[n][m].reshape(1,obs_size)*0.
                aux = obs[n][m].reshape(1,obs_size)*0.
                history[n][m] = np.concatenate((history[n][m],aux),axis=0)
    #Initialize action history buffer with 0.
    history_a = np.zeros([parallel_envs,num_agents,HISTORY_LENGTH,1]) #the last entry is the number of actions, here is 2 (x,y)
    
    scores = 0                
    t = 0
    
    #save gif
    frames = []
    gif_folder = ''
    main_folder = trained_checkpoint.split('\\')
    for i in range(len(main_folder)-2):
        gif_folder += main_folder[i]
        gif_folder += '\\'
    total_rewards = []
    steps = []
    agent_x = []
    agent_y = []
    landmark_x = []
    landmark_y = []
    landmark_p_x = []
    landmark_p_y = []
    range_total = []
    episodes = 0
    episodes_total = []
    while t<200:
        frames.append(env.render('rgb_array'))
        t +=1
        # select an action
        his = []
        for i in range(num_agents):
            his.append(torch.cat((transpose_to_tensor(history)[i],transpose_to_tensor(history_a)[i]), dim=2))
        # actions = maddpg.act(transpose_to_tensor(obs), noise=0.)       
        # actions = maddpg.act(transpose_to_tensor(history), noise=0.) 
        actions = maddpg.act(his,transpose_to_tensor(obs) , noise=0.) 
        
        # print('actions=',actions)
         
        actions_array = torch.stack(actions).detach().numpy()
        actions_for_env = np.rollaxis(actions_array,1)
        
        #cirlce path using my previous functions
        # actions_for_env = circle_path(obs,0.5) #if this value is bigger, the circle radius is smaller 60 => radi = 200m
        print('actions=',actions_for_env)
        
        
        # actions_for_env = np.array([[[-1.]]])
        # if t  > 10:
        #     actions_for_env = np.array([[[0.,0.1]]])
        # if t  > 20:
        #     actions_for_env = np.array([[[0.,0.1]]])
        # if t  > 30:
        #     actions_for_env = np.array([[[0.,0.1]]])
        # if t  > 40:
        #     actions_for_env = np.array([[[1.,0.1]]])
        
        #see a random agent
        # actions_for_env = np.array([[np.random.rand(1)*2-1]])
        beta = 1.99 #must be between 1 and 2
        # actions_for_env = random_levy(beta)
        
        # send all actions to the environment
        next_obs, rewards, dones, info = env.step(actions_for_env)
        
        # Update history buffers
        # Add obs to the history buffer
        for n in range(parallel_envs):
            for m in range(num_agents):
                aux = obs[n][m].reshape(1,obs_size)
                history[n][m] = np.concatenate((history[n][m],aux),axis=0)
                history[n][m] = np.delete(history[n][m],0,0)
        # Add actions to the history buffer
        history_a = np.concatenate((history_a,actions_for_env.reshape(parallel_envs,num_agents,1,1)),axis=2)
        history_a = np.delete(history_a,0,2)
                    
        # update the score (for each agent)
        scores += np.sum(rewards)  
        # Save values to plot later on
        total_rewards.append(np.sum(rewards))
        steps.append(t)          
        for n in range(parallel_envs):
            for m in range(num_agents):
                agent_x.append(obs[n][m][2])
                agent_y.append(obs[n][m][3])
                range_total.append(obs[n][m][6])
            for mm in range(num_landmarks):
                landmark_x.append(info[0]['n'][0][4][0][0])
                landmark_y.append(info[0]['n'][0][4][0][1])
                landmark_p_x.append(obs[n][m][4]+obs[n][m][2])
                landmark_p_y.append(obs[n][m][5]+obs[n][m][3])
                
        # print ('\r\n Rewards at step %i = %.3f'%(t,scores))
        # roll over states to next time step  
        obs = next_obs     

        # print("Score: {}".format(scores))
        episodes += 1
        episodes_total.append(episodes)
        if np.any(dones):
            print('done')
            print('Next:')
            episodes = 0
    
            
    plt.figure(figsize=(5,5))
    plt.plot(steps,total_rewards,'bo-')
    plt.ylabel('Rewards')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.plot(agent_x,agent_y,'bo--',alpha=0.5,label='Agent')
    plt.plot(landmark_p_x,landmark_p_y,'rs--',alpha=0.5,label='Landmark Predicted')
    plt.plot(landmark_x,landmark_y,'k^--',alpha=0.5,label='Landmark Real')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Trained agent (RL)')
    plt.axis('equal')
    # plt.title('Predefined cricumference')
    plt.legend()
    plt.show()
    
    target_error = np.sqrt((np.array(landmark_p_x)-np.array(landmark_x))**2+(np.array(landmark_p_y)-np.array(landmark_y))**2)
    plt.figure(figsize=(5,5))
    plt.plot(steps,target_error,'bo-')
    plt.ylabel('Target prediction error (RMSE)')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.plot(steps,range_total,'bo-')
    plt.ylabel('Range')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.plot(steps,episodes_total,'bo-')
    plt.ylabel('Number of episodes')
    plt.xlabel('Steps')
    plt.title('Trained agent (RL)')
    # plt.title('Predefined cricumference')
    plt.show()
    
    print('MEAN SCORE = ',scores)
    print('TOTAL LAST SCORE = ',np.mean(total_rewards[::-1][:10]))
    
    while True:
        a = 0
    imageio.mimsave(os.path.join(gif_folder, 'seed-{}.gif'.format(SEED)), 
                                frames, duration=.04)
    env.close()
    
if __name__=='__main__':
    main()
    