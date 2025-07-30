import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import time
from app import views
from app.Operator_network.Models.ppo import PPO

device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 100  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 16  # 隐含层神经元个数
env_name = 'CartPole-v1'
return_list = []  # 保存每个回合的return


class MyCallback:
    def __init__(self,num_epochs):
        self.start_time = None
        self.num_epochs = num_epochs  # Total number of epochs for estimate time


    def on_epoch_begin(self):
        self.start_time = time.time()
        views.epoch_g = 0
        views.logs_g = []

    def on_epoch_end(self,epoch,loss):
        views.epoch_g = epoch

        # Update logs globally
        all_log = {'epoch':views.epoch_g,'reward':loss}
        views.logs_g.append(all_log)

        # Calculate estimated remaining time
        end_time = time.time()
        elapsed_time = end_time-self.start_time
        views.estimate_time = elapsed_time*(self.num_epochs-views.epoch_g)/(
            views.epoch_g)  # Estimate time remaining

        print(f'Epoch [{views.epoch_g}/{self.num_epochs}], Reward: {loss:.8f}')
        print(f'Estimated time remaining: {views.estimate_time/60:.2f} minutes')

#epoch=100
def show_PPO(df,seed,branch_layers,trunk_layers,activation,initializer,learning_rate,num_epochs,model_name):
    # ----------------------------------------- #
    # 环境加载
    # ----------------------------------------- #

    env = gym.make(env_name,render_mode="human")
    n_states = env.observation_space.shape[0]  # 状态数 4
    n_actions = env.action_space.n  # 动作数 2

    # ----------------------------------------- #
    # 模型构建
    # ----------------------------------------- #

    agent = PPO(n_states=n_states,  # 状态数
                n_hiddens=n_hiddens,  # 隐含层数
                n_actions=n_actions,  # 动作数
                actor_lr=actor_lr,  # 策略网络学习率
                critic_lr=critic_lr,  # 价值网络学习率
                lmbda=0.95,  # 优势函数的缩放因子
                epochs=10,  # 一组序列训练的轮次
                eps=0.2,  # PPO中截断范围的参数
                gamma=gamma,  # 折扣因子
                device=device
                )
    # ----------------------------------------- #
    # 训练--回合更新 on_policy
    # ----------------------------------------- #
    callback = MyCallback(num_epochs=num_epochs)
    callback.on_epoch_begin()  # Callback for the start of the epoch
    for i in range(num_epochs):
        state = env.reset()[0]  # 环境重置
        done = False  # 任务完成的标记
        episode_return = 0  # 累计每回合的reward

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
            'states':[],
            'actions':[],
            'next_states':[],
            'rewards':[],
            'dones':[],
        }

        while not done:
            action = agent.take_action(state)  # 动作选择
            next_state,reward,done,_,_ = env.step(action)  # 环境更新
            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新状态
            state = next_state
            # 累计回合奖励
            episode_return += reward

        # 模型训练
        agent.learn(transition_dict)

        # 打印回合信息
        if (i+1)%10 == 0:
            # print(f'iter:{i}, return:{np.mean(return_list[-10:])}')
            # 保存每个回合的return
            return_list.append(episode_return)
            callback.on_epoch_end((i+1),episode_return)
            # torch.save(agent.actor.state_dict(),'ql_'+model_name+'actor_model.pkl')
            # torch.save(agent.critic.state_dict(),'ql_'+model_name+'actor_critic.pkl')
    views.epoch_g = views.epoch_g*100
    views.loss_g = return_list
    print('loss数据',return_list)
    env.close()
