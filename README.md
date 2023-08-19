<h1>Imitate the Good and Avoid the Bad: An incremental approach to Safe
Reinforcement Learning</h1>

## Introduction
A popular framework for enforcing safe actions in Reinforcement Learning (RL) is Constrained RL, where trajectory
based constraints on expected cost (or other cost measures)
are employed to enforce safety and more importantly these
constraints are enforced while maximizing expected reward.
Most recent approaches for solving Constrained RL convert
the trajectory based cost constraint into a surrogate problem
that can be solved using minor modifications to RL methods.
A key drawback with such approaches is an over or underestimation of the cost constraint at each state. Therefore, we
provide an approach that does not modify the trajectory based
cost constraint and instead imitates “good” trajectories and
avoids “bad” trajectories generated from incrementally improving policies. We employ an oracle that utilizes a reward
threshold (which is varied with learning) and the overall cost
constraint to label trajectories as “good” or “bad”. A key advantage of our approach is that we are able to work from any
starting policy or set of trajectories and improve on it. In an
exhaustive set of experiments, we demonstrate that our approach is able to outperform top benchmark approaches for
solving Constrained RL problems, with respect to expected
cost, CVaR cost, or even unknown cost constraints.

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
    git clone https://github.com/PKU-Alignment/safety-gymnasium.git
    cd safety-gymnasium
    pip install -e .
    cd ..
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    ```
## Environment informations
We conduct experiments in 6 different safety-gym environments which can be found in [Safety-gymnaisum](https://www.safety-gymnasium.com/en/latest/). 
## Train Self-IMitation based safe RL
1. In our experiments, we use a relaxed-constraint expert to help our agent collect a number of expert trajectories at the beginning for an unsafe agent having a high return. We want to achieve both a high return and safety for the strict-constraint agent through our work. Here, we provide some pre-trained relaxed agents which are located on [drive](https://drive.google.com/drive/folders/17qgFn1Wl_-V6WvmI6liGiawcg7qRct4t?usp=sharing).

1. For example, to run train SIM in SafetyPointPush1-v0:

    ``` bash
    ./Scripts/run_train_good_bad.sh

    ```

1. To re-train the relaxed constraint expert from scratch, we train PPO-Lagrangian with a higher cost limit C_max = 25.0. For example, to train relaxed-constraint expert in SafetyPointPush1-v0:
    ```
    ./Scripts/run_train_PPO_lag.sh
    ```
    More over, we save our weight as following format: "(value)-(satisfaction_rate)-(mean_return)-(mean_cost).pth".
    Select the highest (value) as the relaxed expert for SIM training as well as mean_return fpr min_good.

## Directory Structure
```
├───Parameters                          # Parameter list
├───Sources
│   ├───algo
│   │   ├───base_algo.py
│   │   ├───ppo.py                      # PPO and PPO-lag implementation
│   │   └───sim.py                      # SIM implementation
│   ├───buffer
│   │   └───buffer_PPO.py               # buffer implementation
│   ├───network
│   │   ├───classifier.py               # classifier implementation
│   │   ├───policy.py                   # policy implementation
│   │   └───value.py                    # critic implementation
│   └───utils                           
├───Trains
│   ├───train_good_bad.py               # training file for SIM
│   ├───train_PPO.py                    # training file for PPO
│   └───train_PPO_lag.py                # training file for PPO-lag
├───weights
│   └───Expert_cmax(25)                 # relaxed-constraint expert location
├───Scripts                             # bash script for training
│   ├───run_train_good_bad.sh           
│   ├───run_train_PPO.sh                
│   └───run_train_PPO_lag.sh            
└───Plot_figures                        # Scripts to draw training curves in the paper
    └───log_data                        # collected data for the experiments

```

## Experiments

Our method (SIM) is compared with several existing methods across several different environments:

1. [PPO](https://arxiv.org/pdf/1707.06347.pdf) (John Schulman et al., 2017)
1. [PPO-Lagrangian](https://cdn.openai.com/safexp-short.pdf) (Ray, Achiam,
and Amodei, 2019)
1. [FOCOPS](https://arxiv.org/abs/2002.06506) (Zhang, Vuong, and Ross, 2020)
1. [CUP](https://arxiv.org/abs/2209.07089) (Joshua Achiam et al., 2017)
1. [CPO](https://proceedings.mlr.press/v70/achiam17a) (Yang et al., 2022)

To ensure a fair comparison, we utilize the same hyperparameters across all environments, which can be found in the paper's appendix section. To access and refer to the hyperparameters, please access to the [Parameters](Parameters) and [script bash files](./Scripts/run_train_good_bad.sh). 
All the experiments are listed in [Plot figures](Plot_figures). For additional specifics, kindly refer to that particular folder.

## Conclusion

We introduced a novel framework to solve Constrained RL without relying on cost estimations or cost penalties, as commonly done in prior work. Our new algorithm, based on the idea of learning to mimic the behavior of good demonstrations and avoid bad demonstrations, is non-adversarial and allows learning from demonstration sets to evolve during the training process. Extensive experiments on several challenging benchmark tasks demonstrate that our approach achieves superior performance compared to prior constrained RL algorithms.
Our IL-based framework would open new directions to address safe RL problems without explicitly considering the reward or cost function. Our algorithm relies on sets of good demonstrations generated by a pre-trained policy, so a limitation would be that our algorithm will not work if it is difficult to generate feasible trajectories due to, for instance, strict constraints. A future direction would be to develop new IL-based algorithms to address such issues.

## Contact
...