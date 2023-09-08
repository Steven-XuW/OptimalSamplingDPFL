#!/bin/bash
python3 main.py \
--model cnn \
--batch_size 512 \
--num_round 50 \
--clip_size 35 \
--dataset emnist_all_data_3_random_niid \
--sampling_scheme optimal \
--injection_pattern grad \
--read_exist_data True \
--epsilon_seed 0 \
--balance_l 3 \
--clients_per_round 5 \
--gpu