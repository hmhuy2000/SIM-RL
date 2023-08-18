<h1>mitate the Good and Avoid the Bad: An incremental approach to Safe
Reinforcement Learning</h1>


<!-- ## Announcements
#### August 18, 2023
- <b>Official code published</b> -->
<!-- ---- -->
## Installation
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

1. Clone this repo
    ```
    git clone https://github.com/hmhuy2000/SIM-RL.git
    cd SIM-RL
    ```

1. Create and activate conda environment
    ```
    conda create -n SIM-RL python=3.10
    conda activate SIM-RL
    ```
1. Install dependencies
    ```
    cd safety-gymnasium
    pip install -e .
    cd ..
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```
## Environment informations
We conduct experiments in 6 different safety-gym environments which can be found in [Safety-gymnaisum](https://www.safety-gymnasium.com/en/latest/). 

## Train Self-IMitation based safe RL
1. In our experiments, we use an relaxed-constraint expert to help our agent collect a number of expert trajectories at the begining for unsafe agent having high return. We want to achieve both high return and safe to strict constraint agent from our work. . Here, we provide some pretrained relaxed agent as well as their threshold  (min_good, max_bad) as bellow:
    ```
    SafetyPointGoal1-v0 (min_good = 25.0, max_bad = 20.0):              weights/Expert_cmax(25)/PointGoal_actor.pth
    SafetyCarGoal1-v0 (min_good = 25.0, max_bad = 20.0):                weights/Expert_cmax(25)/CarGoal_actor.pth
    SafetyPointButton1-v0 (min_good = 12.0, max_bad = 7.0):             weights/Expert_cmax(25)/PointButton_actor.pth
    SafetyCarButton1-v0 (min_good = 15.0, max_bad = 10.0):              weights/Expert_cmax(25)/CarButton_actor.pth
    SafetyPointPush1-v0 (min_good = 11.0, max_bad = 6.0):               weights/Expert_cmax(25)/PointPush_actor.pth
    SafetyCarPush1-v0 (min_good = 8.0, max_bad = 4.0):                  weights/Expert_cmax(25)/CarPush_actor.pth
    ```

1. For example, to run train SIM in SafetyPointPush1-v0:

    ``` bash
    python Trains/train_good_bad.py \
    --env_name='SafetyPointPush1-v0' \
    --seed=0 --num_training_step=30000000 \
    --gamma=0.99 --cost_gamma=0.99 \
    --number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
    --coef_ent=0.0001 --reward_factor=1.0 --cost_limit=15.0 \
    --lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.01 --clip_eps=0.2 \
    --num_eval_episodes=100 --eval_num_envs=0 --max_grad_norm=1.0 --epoch_ppo=160 \
    --buffer_size=50000 --eval_interval=300000 --num_envs=25 --max_episode_length=1000 \
    --risk_level=1.0 --batch_size=4096 --epoch_clfs=100 \
    --weight_path='./weights/SafetyPointPush1-v0/SIM' \
    --dynamic_good=False --tanh_conf=False --min_good=11.0 --max_bad=6.0 \
    --conf_coef=0.03 --start_bad=100 \
    --expert_path='./weights/Expert_cmax(25)/PointPush_actor.pth'

    ```

1. To re-train the relaxed constraint expert from scratch, we train PPO-Lagrangian with a higher cost limit C_max = 25.0. For example, to train relaxed-constraint expert in SafetyPointPush1-v0:
    ```
    python Trains/train_PPO_lag.py \
    --env_name='SafetyCarButton1-v0' \
    --seed=0 --num_training_step=30000000 \
    --gamma=0.99 --cost_gamma=0.99 \
    --number_layers=3 --hidden_units_actor=256 --hidden_units_critic=256 \
    --coef_ent=0.0001 --reward_factor=1.0 --cost_limit=25.0 \
    --lr_actor=0.0001 --lr_critic=0.0001 --lr_penalty=0.01 --clip_eps=0.2 \
    --num_eval_episodes=100 --eval_num_envs=25 --max_grad_norm=1.0 --epoch_ppo=160 \
    --buffer_size=50000 --eval_interval=500000 --num_envs=25 --max_episode_length=1000 \
    --weight_path='./weights/SafetyCarButton1-v0/PPO-lag(25.0)' 
    ```
    More over, we save our weight as following format: "(value)-(satisfaction_rate)-(mean_return)-(mean_cost).pth".
    Select the highest (value) as the relaxed expert for SIM training as well as mean_return fpr min_good.
## Plot Figure

## Directory Tree