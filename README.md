# KMIFQE

This repository is the official implementation of KMIFQE: Kernel Metric Learning for In-Sample Off-Policy Evaluation (presented at ICLR 2024 as a **Spotlight paper**).

### Install Conda Environment
```
conda env create --name envname --file=environment.yml
```

### How to Run

1. Train behavior and target policies: 
```
python train_policy.py \ 
  --env=Hopper-v2
``` 

2. Collect data: 
```
python save_replay_buffer.py \
  --env=Hopper-v2 \
  --policy_idx=[behavior policy file name] \
  --max_timesteps=1000000 \
  --random=0 \ 
  --behav_bias=0 \
  --behav_std=0.3
``` 

3. Train and evaluate KMIFQE: 
```
python main.py \ 
  --env=Hopper-v2 \
  --target_policy_idx=[target policy file name] \
  --behavior_policy_idx=[behavior policy file name] \
  --buffer_size=1000000 \ 
  --behav_bias=0 \
  --behav_std=0.3
``` 

### Bibtex
If you use this code, please cite our paper:
```
@inproceedings{lee2024kmifqe,
  title={Kernel Metric Learning for In-Sample Off-Policy Evaluation of Deterministic {RL} Policies},
  author={Haanvid Lee and Tri Wahyu Guntara and Jongmin Lee and Yung-Kyun Noh and Kee-Eung Kim},
  booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
  year={2024}
}
```