python Trains/train_PPO_lag.py \
--env_name='SafetyPointPush1-v0' \
--seed=0 --num_training_step=30000000 \
--gamma=0.99 --cost_gamma=0.99 \
--number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
--coef_ent=0.0001 --reward_factor=1.0 --cost_limit=25.0 \
--lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.01 --clip_eps=0.2 \
--num_eval_episodes=100 --eval_num_envs=25 --max_grad_norm=1.0 --epoch_ppo=160 \
--buffer_size=50000 --eval_interval=500000 --num_envs=25 --max_episode_length=1000 \
--weight_path='./weights/SafetyPointPush1-v0/PPO-lag(25.0)' 