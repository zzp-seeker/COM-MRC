#!/bin/bash

# D2
python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/14res \
--a 5 --b 11 --a_ww 0.5 --b_ww 1 --use_c 1

python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/14lap \
--a 5 --b 11 --a_ww 0.5 --b_ww 1 --use_c 1

python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/15res \
--a 4.5 --b 11.5 --a_ww 0.5 --b_ww 1 --use_c 1

python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/16res \
--a 6 --b 10 --a_ww 1 --b_ww 1 --use_c 1

# D1
python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/14res \
--a 5 --b 11 --a_w 0.5 --b_w 1 --use_c 1

python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/14lap \
--a 6 --b 10 --a_w 1 --b_w 1 --use_c 1

python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/15res \
--a 5 --b 11 --a_w 0.5 --b_w 1 --use_c 1

python main.py --device cuda:0 --seed 2025 --lr 9e-5 --batch_size 15 \
--data_dir ../Dataset/ASTE-Data-V2/16res \
--a 5 --b 11 --a_w 0.5 --b_w 1 --use_c 1









