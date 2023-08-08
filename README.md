# SIM-RL

conda create -n SIM-RL python=3.10
cd safety-gymnasium
pip install -e .
cd ..
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt