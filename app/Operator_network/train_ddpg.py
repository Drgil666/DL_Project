import gym  # 导入 Gym 库，用于创建和管理强化学习环境
import torch
from app import views
import time
from app.Operator_network.Models.ddpg import DDPGAgent

# 绘制学习曲线的方法

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device type: ",device)


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

#epoch=50~60
def show_DDPG(df,seed,branch_layers,trunk_layers,activation,initializer,learning_rate,num_epochs,model_name):
    # 创建环境
    env = gym.make("Pendulum-v1",render_mode="human")
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.shape[0]  # 动作空间维度
    max_action = float(env.action_space.high[0])  # 动作最大值
    max_steps = 200
    # 初始化DDPG智能体
    agent = DDPGAgent(state_dim,action_dim,max_action)
    rewards = []  # 用于存储每个episode的奖励
    callback = MyCallback(num_epochs=num_epochs)
    callback.on_epoch_begin()  # Callback for the start of the epoch
    for episode in range(num_epochs):
        state,_ = env.reset()  # 重置环境，获取初始状态
        episode_reward = 0  # 初始化每轮奖励为0
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            # 执行动作，获取环境反馈
            next_state,reward,done,_,_ = env.step(action)
            # 将样本存入回放池
            agent.add_to_replay_buffer(state,action,reward,next_state,done)

            # 训练智能体
            agent.train()
            # 更新当前状态
            state = next_state
            # 累加奖励
            episode_reward += reward

            if done:  # 如果完成（到达终止状态），结束本轮
                break

        # 记录每轮的累计奖励
        rewards.append(episode_reward)
        # print(f"Episode: {episode+1}, Reward: {episode_reward}")
        callback.on_epoch_end(episode+1,episode_reward)
    views.epoch_g = views.epoch_g*1000
    views.loss_g = rewards
    print('loss数据',rewards)
    env.close()  # 关闭环境
