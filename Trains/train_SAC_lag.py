import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.SAC_lag_parameters import *

#------------------------------------------#
def main():
    import safety_gymnasium
    sample_env = safety_gymnasium.make(args.env_name)
    env = safety_gymnasium.vector.make(env_id=args.env_name, num_envs=args.num_envs)
    if (args.eval_num_envs):
        test_env = safety_gymnasium.vector.make(env_id=args.env_name, num_envs=args.eval_num_envs)
    else:
        test_env = None
    #------------------------------------------#
    from Sources.algo.sac import SAC_lag
    from Sources.utils import create_folder
    from copy import deepcopy
    import threading
    import torch
    import setproctitle
    from torch import nn
    import wandb

    #------------------------------------------#
    def evaluate(algo, env,max_episode_length):
        global max_eval_return
        mean_return = 0.0
        mean_cost = 0.0
        failed_case = []
        cost_sum = [0 for _ in range(args.eval_num_envs)]

        for step in range(args.num_eval_episodes//args.eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            for iter in range(max_episode_length):
                if (iter%100 == 0):
                    print(f'valid {step+1}/{args.num_eval_episodes//args.eval_num_envs}: {iter/max_episode_length*100:.2f}% {iter}/{max_episode_length}', end='\r')
                action = algo.exploit(state)
                state, reward, cost, done, _, _ = env.step(action)
                episode_return += np.sum(reward)
                episode_cost += np.sum(cost)
                for idx in range(args.eval_num_envs):
                    cost_sum[idx] += cost[idx]
            for idx in range(args.eval_num_envs):
                failed_case.append(cost_sum[idx])
                cost_sum[idx] = 0
            mean_return += episode_return 
            mean_cost += episode_cost 

        mean_return = mean_return/args.num_eval_episodes
        mean_cost = mean_cost/args.num_eval_episodes
        tmp_arr = np.asarray(failed_case)

        success_rate = np.sum(tmp_arr<=args.cost_limit)/args.num_eval_episodes
        value = (mean_return * success_rate)/10
        if (value>max_eval_return):
            max_eval_return = value
            # algo.save_models(f'{args.weight_path}/({value:.3f})-({success_rate:.2f})-({mean_return:.2f})-({mean_cost:.2f})')
        else:
            max_eval_return*=0.999
        print(f'[Eval] R: {mean_return:.2f}, C: {mean_cost:.2f}, '+
            f'SR: {success_rate:.2f}, '
            f'V: {value:.2f}, maxV: {max_eval_return:.2f}')

    def train(env,test_env,algo,eval_algo):
        t = [0 for _ in range(args.num_envs)]
        eval_thread = None
        state,_ = env.reset()
        log_cnt = 0
        
        print('start training')
        for step in range(1,args.num_training_step//args.num_envs+1):
            log_info = {}
            if (step%100 == 0):
                print(f'train: {step/(args.num_training_step//args.num_envs)*100:.2f}% {step}/{args.num_training_step//args.num_envs}', end='\r')
            state, t = algo.step(env, state, t)
            if algo.is_update(step*args.num_envs):
                algo.update(log_info)
                
            if (step and step*args.num_envs%args.log_freq==0):
                args.eval_return.write(f'{np.mean(algo.return_reward)}\n')
                args.eval_return.flush()
                args.eval_cost.write(f'{np.mean(algo.return_cost)}\n')
                args.eval_cost.flush()
        
                log_info['update_step']=log_cnt
                try:
                    wandb.log(log_info, step = log_cnt)
                except:
                    print(log_info)
                log_cnt += 1
                    
        algo.save_models(f'{args.weight_path}/s{args.seed}-finish')
            
    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    sample_env.close()

    setproctitle.setproctitle(f'{args.env_name}-SAC-lag-{args.seed}')
    
    algo = SAC_lag(
            state_shape=state_shape, action_shape=action_shape, device=args.device, seed=args.seed, gamma=args.gamma,
                 SAC_batch_size=args.SAC_batch_size, buffer_size=args.buffer_size, lr_actor=args.lr_actor, lr_critic=args.lr_critic, 
                 lr_alpha=args.lr_alpha, hidden_units_actor=args.hidden_units_actor, hidden_units_critic=args.hidden_units_critic, 
                 start_steps=args.start_steps,tau=args.tau,max_episode_length=args.max_episode_length, reward_factor=args.reward_factor,
                 max_grad_norm=args.max_grad_norm,cost_limit=args.cost_limit,
                 penalty=args.penalty)
    
    wandb_logs = True
    wandb_group = f'SAC-lag-{args.penalty}({args.cost_limit})'
    if (wandb_logs):
        print('---------------------using Wandb---------------------')
        wandb.init(project=f'{args.env_name}', settings=wandb.Settings(_disable_stats=True), \
        group='off-policy',job_type=wandb_group, name=f'{args.seed}', entity='hmhuy',config=args)
    else:
        print('----------------------no Wandb-----------------------')
    
    eval_algo = deepcopy(algo)
    create_folder(args.weight_path)
    print(args)
    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

if __name__ == '__main__':
    main()