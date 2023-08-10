CUDA_VISIBLE_DEVICES=0 python Trains/train_good_bad.py \
--env_name='SafetyCarGoal1-v0' \
--seed=0 --num_training_step=30000000 \
--gamma=0.99 --cost_gamma=0.99 \
--number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
--coef_ent=0.0001 --reward_factor=1.0 --cost_limit=15.0 \
--lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.01 --clip_eps=0.2 \
--num_eval_episodes=100 --eval_num_envs=100 --max_grad_norm=1.0 \
--buffer_size=50000 --eval_interval=50000 --num_envs=10 --max_episode_length=1000 \
--risk_level=1.0 --batch_size=4096 --epoch_clfs=1000 \
--weight_path='./weights/SafetyCarGoal1-v0/SIM-tanh-dynamic' \
--dynamic_good=True --tanh_conf=True --min_good=25.0 --max_bad=15.0 \
--conf_coef=0.01 --start_bad=300 \
--expert_path='./weights/SafetyCarGoal1-v0/PPO-lag(25)/(1.960)0.72-27.22/actor.pth'


# CUDA_VISIBLE_DEVICES=1 python Trains/train_good_bad.py \
# --env_name='SafetyPointPush1-v0' \
# --seed=3 --num_training_step=60000000 \
# --gamma=0.99 --cost_gamma=0.99 \
# --number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
# --coef_ent=0.0001 --reward_factor=1.0 --cost_limit=15.0 \
# --lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.01 --clip_eps=0.2 \
# --num_eval_episodes=100 --eval_num_envs=100 --max_grad_norm=1.0 \
# --buffer_size=100000 --eval_interval=100000 --num_envs=10 --max_episode_length=1000 \
# --risk_level=1.0 --batch_size=4096 --epoch_clfs=1000 \
# --weight_path='./weights/SafetyPointPush1-v0/SIM-tanh' \
# --dynamic_good=True --tanh_conf=True --min_good=11.0 --max_bad=6.0 --conf_coef=0.01 \
# --expert_path='./weights/SafetyPointPush1-v0/PPO-lag(25)/(0.685)0.64-10.71/actor.pth'

# CUDA_VISIBLE_DEVICES=0 python Trains/train_good_bad.py \
# --env_name='SafetyDoggoCircle1-v0' \
# --seed=4 --num_training_step=60000000 \
# --gamma=0.99 --cost_gamma=0.99 \
# --number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
# --coef_ent=0.0001 --reward_factor=1.0 --cost_limit=15.0 \
# --lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.03 --clip_eps=0.2 \
# --num_eval_episodes=100 --eval_num_envs=100 --max_grad_norm=1.0 \
# --buffer_size=100000 --eval_interval=100000 --num_envs=10 --max_episode_length=500 \
# --risk_level=1.0 --batch_size=4096 --epoch_clfs=1000 \
# --weight_path='./weights/SafetyDoggoCircle1-v0/SIM-sigmoid-fixed' \
# --dynamic_good=False --tanh_conf=False --min_good=18.0 --max_bad=13.0 \
# --conf_coef=0.1 --start_bad=100 \
# --expert_path='./weights/SafetyDoggoCircle1-v0/PPO-lag(25)/(1.569)0.79-19.87/actor.pth'
