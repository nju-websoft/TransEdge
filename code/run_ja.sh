#!/usr/bin/env bash

python3 transedge_ea.py --mode 'projection' \
                        --data_dir '../data/DBP15K/ja_en/0_3/' \
                        --ent_norm True \
                        --rel_norm True \
                        --op_is_norm True \
                        --embedding_dim 75 \
                        --pos_margin 0.2 \
                        --neg_margin 2.2 \
                        --neg_param 0.8 \
                        --n_neg_triple 20 \
                        --truncated_epsilon 0.95 \
                        --mlp_layers 1 \
                        --learning_rate 0.01 \
                        --batch_size 2000 \
                        --max_epoch 1000 \
                        --frequency 5