CUDA_VISIBLE_DEVICES=3 python Trains/train_SAC.py \
--env_name='SafetyPointPush1-v0' \
--seed=1 --num_training_step=3000000 \
--gamma=0.99 --cost_gamma=0.99 \
--number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
--reward_factor=1.0 --cost_limit=15.0 \
--lr_actor=0.0001 --lr_critic=0.0001 \
--num_eval_episodes=100 --eval_num_envs=0 --max_grad_norm=1.0 \
--eval_interval=10000 --num_envs=1 --max_episode_length=1000 \
--weight_path='./weights' 