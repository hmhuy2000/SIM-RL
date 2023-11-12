import sys
sys.path.append('..')
sys.path.append('./')
from Parameters.PPO_parameters import *

#------------------------------------------#
def main():
    import safety_gymnasium
    sample_env = safety_gymnasium.make(env_name)
    env = safety_gymnasium.vector.make(env_id=env_name, num_envs=num_envs)
    if (eval_num_envs):
        test_env = safety_gymnasium.vector.make(env_id=env_name, num_envs=eval_num_envs)
    else:
        test_env = None
    #------------------------------------------#
    from Sources.algo.ppo import PPO_lag
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
        cost_sum = [0 for _ in range(eval_num_envs)]

        for step in range(num_eval_episodes//eval_num_envs):
            state,_ = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            for iter in range(max_episode_length):
                if (iter%100 == 0):
                    print(f'valid {step+1}/{num_eval_episodes//eval_num_envs}: {iter/max_episode_length*100:.2f}% {iter}/{max_episode_length}', end='\r')
                action = algo.exploit(state)
                state, reward, cost, done, _, _ = env.step(action)
                episode_return += np.sum(reward)
                episode_cost += np.sum(cost)
                for idx in range(eval_num_envs):
                    cost_sum[idx] += cost[idx]
            for idx in range(eval_num_envs):
                failed_case.append(cost_sum[idx])
                cost_sum[idx] = 0
            mean_return += episode_return 
            mean_cost += episode_cost 

        mean_return = mean_return/num_eval_episodes
        mean_cost = mean_cost/num_eval_episodes
        tmp_arr = np.asarray(failed_case)

        success_rate = np.sum(tmp_arr<=cost_limit)/num_eval_episodes
        value = (mean_return * success_rate)/10
        if (value>max_eval_return):
            max_eval_return = value
            algo.save_models(f'{weight_path}/({value:.3f})-({success_rate:.2f})-({mean_return:.2f})-({mean_cost:.2f})')
        else:
            max_eval_return*=0.999
        print(f'[Eval] R: {mean_return:.2f}, C: {mean_cost:.2f}, '+
            f'SR: {success_rate:.2f}, '
            f'V: {value:.2f}, maxV: {max_eval_return:.2f}')

    def train(env,test_env,algo,eval_algo):
        t = [0 for _ in range(num_envs)]
        eval_thread = None
        state,_ = env.reset()
        log_cnt = 0

        print('start training')
        for step in range(1,num_training_step//num_envs+1):
            log_info = {}
            if (step%100 == 0):
                print(f'train: {step/(num_training_step//num_envs)*100:.2f}% {step}/{num_training_step//num_envs}', end='\r')
            state, t = algo.step(env, state, t)
            if algo.is_update(step*num_envs):
                    eval_return.write(f'{np.mean(algo.return_reward)}\n')
                    eval_return.flush()
                    eval_cost.write(f'{np.mean(algo.return_cost)}\n')
                    eval_cost.flush()
                    algo.update(log_info)
                    
            if (len(log_info.keys())>0):
                log_info['update_step']=log_cnt
                try:
                    wandb.log(log_info, step = log_cnt)
                except:
                    print(log_info)
                log_cnt += 1                    

            # if step % (eval_interval//num_envs) == 0:
            #     algo.save_models(f'{weight_path}/s{seed}-latest')
            #     if (test_env):
            #         if eval_thread is not None:
            #             eval_thread.join()
            #         eval_algo.copyNetworksFrom(algo)
            #         eval_algo.eval()
            #         eval_thread = threading.Thread(target=evaluate, 
            #         args=(eval_algo,test_env,max_episode_length))
            #         eval_thread.start()
        algo.save_models(f'{weight_path}/s{seed}-finish')

    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    sample_env.close()

    setproctitle.setproctitle(f'{env_name}-PPO-lag-{seed}')
    
    algo = PPO_lag(env_name=env_name,
            state_shape=state_shape, action_shape=action_shape,
            device=device, seed=seed, gamma=gamma,cost_gamma=cost_gamma,buffer_size=buffer_size,
            mix=mix, hidden_units_actor=hidden_units_actor,
            hidden_units_critic=hidden_units_critic,units_clfs=hidden_units_clfs,
            lr_actor=lr_actor,lr_critic=lr_critic,lr_cost_critic=lr_cost_critic,lr_penalty=lr_penalty, epoch_ppo=epoch_ppo,
            epoch_clfs=epoch_clfs,batch_size=batch_size,lr_clfs=lr_clfs,clip_eps=clip_eps, lambd=lambd, coef_ent=coef_ent,
            max_grad_norm=max_grad_norm,reward_factor=reward_factor,max_episode_length=max_episode_length,
            cost_limit=cost_limit,risk_level=risk_level,num_envs=num_envs)
    
    wandb_logs = True
    wandb_group = f'PPO-lag {cost_limit}'
    if (wandb_logs):
        print('---------------------using Wandb---------------------')
        wandb.init(project=f'{args.env_name}', settings=wandb.Settings(_disable_stats=True), \
        group='on-policy',job_type=wandb_group, name=f'{args.seed}', entity='hmhuy',config=args)
    else:
        print('----------------------no Wandb-----------------------')
    
    eval_algo = deepcopy(algo)
    create_folder(weight_path)
    train(env=env,test_env=test_env,algo=algo,eval_algo=eval_algo)

    env.close()
    if (test_env):
        test_env.close()

if __name__ == '__main__':
    main()