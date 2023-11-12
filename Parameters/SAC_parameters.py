import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

training_group = parser.add_argument_group('SAC_training')
training_group.add_argument('--env_name',type=str,default='SafetyPointGoal1-v0')
training_group.add_argument('--gamma', type=float, default=0.99)
training_group.add_argument('--cost_gamma', type=float, default=0.99)
training_group.add_argument('--device', type=str, default='cuda')
training_group.add_argument('--seed', type=int, default=0)
training_group.add_argument('--buffer_size',type=int,default=int(1e6))
training_group.add_argument('--hidden_units_actor',type=int,default=256)
training_group.add_argument('--hidden_units_critic',type=int,default=256)
training_group.add_argument('--number_layers',type=int,default=3)
training_group.add_argument('--lr_actor', type=float, default=0.0001)
training_group.add_argument('--lr_critic', type=float, default=0.0001)
training_group.add_argument('--lr_alpha', type=float, default=0.0001)
training_group.add_argument('--cost_limit', type=float, default=25.0)

training_group.add_argument('--SAC_batch_size',type=int,default=4096)
training_group.add_argument('--max_grad_norm', type=float, default=1.0)
training_group.add_argument('--num_training_step',type=int,default=int(1e6))
training_group.add_argument('--eval_interval',type=int,default=int(1e4))
training_group.add_argument('--num_eval_episodes',type=int,default=100)
training_group.add_argument('--max_episode_length',type=int,default=1000)
training_group.add_argument('--reward_factor',type=float,default=1.0)
training_group.add_argument('--tau',type=float,default=1e-2)
training_group.add_argument('--start_steps',type=int,default=int(5e3))
training_group.add_argument('--log_freq',type=int,default=int(1e4))
training_group.add_argument('--weight_path', type=str, default='./weights')

training_group.add_argument('--begin_cpu',type=int,default=0)
training_group.add_argument('--end_cpu',type=int,default=96)
training_group.add_argument('--num_envs',type=int,default=1)
training_group.add_argument('--eval_num_envs',type=int,default=25)

#-------------------------------------------------------------------------------------------------#

# training
args = parser.parse_args()
hidden_units_actor                      = []
hidden_units_critic                     = []
for _ in range(args.number_layers):
    hidden_units_actor.append(args.hidden_units_actor)
    hidden_units_critic.append(args.hidden_units_critic)

args.hidden_units_actor                 = hidden_units_actor
args.hidden_units_critic                = hidden_units_critic
max_value                               = -np.inf

args.weight_path = os.path.join(args.weight_path,args.env_name,'SAC')
args.log_path = f'{args.weight_path}/log_data'
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
args.eval_return = open(f'{args.log_path}/return_{args.seed}.txt','w')
args.eval_cost = open(f'{args.log_path}/cost_{args.seed}.txt','w')

