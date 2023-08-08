# CUDA_VISIBLE_DEVICES=0 python Trains/train_PPO_lag.py \
# --env_name='SafetyPointGoal1-v0' \
# --seed=0 --num_training_step=30000000 \
# --gamma=0.99 --cost_gamma=0.99 \
# --number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
# --coef_ent=0.0001 --reward_factor=1.0 --cost_limit=25.0 \
# --lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.03 --clip_eps=0.2 \
# --num_eval_episodes=100 --eval_num_envs=100 --max_grad_norm=1.0 \
# --buffer_size=50000 --eval_interval=50000 --num_envs=10 --max_episode_length=1000 \
# --risk_level=1.0 --batch_size=4096 --epoch_disc=1000 \
# --weight_path='./weights/SafetyPointGoal1-v0/PPO-lag(25)' 

CUDA_VISIBLE_DEVICES=0 python Trains/train_PPO_lag.py \
--env_name='SafetyDoggoCircle1-v0' \
--seed=0 --num_training_step=30000000 \
--gamma=0.99 --cost_gamma=0.99 \
--number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
--coef_ent=0.0001 --reward_factor=1.0 --cost_limit=25.0 \
--lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.03 --clip_eps=0.2 \
--num_eval_episodes=100 --eval_num_envs=100 --max_grad_norm=1.0 \
--buffer_size=50000 --eval_interval=50000 --num_envs=10 --max_episode_length=500 \
--risk_level=1.0 --batch_size=4096 --epoch_disc=1000 \
--weight_path='./weights/SafetyDoggoCircle1-v0/PPO-lag(25)' 