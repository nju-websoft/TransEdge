# TransEdge
Source code for ISWC-2019 paper "[TransEdge: Translating Relation-contextualized Embeddings for Knowledge Graphs](https://link.springer.com/chapter/10.1007/978-3-030-30793-6_35)". 

<p align="center">
  <img width="90%" src="https://github.com/nju-websoft/TransEdge/blob/master/TransEdge_illustration.png" />
</p>

## Dataset
* For entity alignment, we use two datasets DBP15K and DWY100K. DBP15K can be downloaded from [JAPE](https://github.com/nju-websoft/JAPE) and DWY100K is from [BootEA](https://github.com/nju-websoft/BootEA).
* For link prediction, we use two datasets FB15k-237 and WN18RR, which can be downloaded from [ConvE](https://github.com/TimDettmers/ConvE).

## Code
* "transedge_ea.py" is the implementation of TransEdge for entity alignment;
* "transedge_lp.py" is the implementation of TransEdge for link prediction.

### Dependencies
* Python 3
* Tensorflow 1.x 
* Scipy
* Numpy
* Pandas
* Graph-tool (It is recommended to follow the [offical instruction](https://git.skewed.de/count0/graph-tool/wikis/installation-instructions) to install graph-tool.)

### Running

For example, to run TransEdge-CP (w/o semi) on DBP15K ZH-EN, use the following script (supposed that the dataset has been downloaded into the folder '../data/'):
```
python3 transedge_ea.py --mode 'projection' \
                        --data_dir '../data/DBP15K/zh_en/0_3/' \
                        --ent_norm True \
                        --rel_norm True \
                        --op_is_norm True \
                        --embedding_dim 75 \
                        --pos_margin 0.2 \
                        --neg_margin 2.0 \
                        --neg_param 0.8 \
                        --n_neg_triple 20 \
                        --truncated_epsilon 0.95 \
                        --mlp_layers 1 \
                        --learning_rate 0.01 \
                        --batch_size 2000 \
                        --max_epoch 1000 \
                        --frequency 5 \
```

To run TransEdge-CP on DBP15K ZH-EN, use the following script:
```
python3 transedge_ea.py --mode 'projection' \
                        --data_dir '../data/DBP15K/zh_en/0_3/' \
                        --ent_norm True \
                        --rel_norm True \
                        --op_is_norm True \
                        --embedding_dim 75 \
                        --pos_margin 0.2 \
                        --neg_margin 2.0 \
                        --neg_param 0.8 \
                        --n_neg_triple 20 \
                        --truncated_epsilon 0.95 \
                        --mlp_layers 1 \
                        --learning_rate 0.01 \
                        --batch_size 2000 \
                        --max_epoch 1000 \
                        --frequency 5 \
                        --is_bp True \
                        --sim_th 0.7 \
                        --top_k 10 \
                        --op_is_tanh True
```

To run TransEdge-CP on WN18RR, use the following script:
```
python3 transedge_lp.py --mode 'projection' \
                        --data_dir '../data/WN18RR/' \
                        --embedding_dim 500 \
                        --pos_margin 0.2 \
                        --neg_margin 3.5 \
                        --neg_param 0.5 \
                        --n_neg_triple 30 \
                        --truncated_epsilon 1.0 \
                        --truncated_frequency 10 \
                        --mlp_layers 2 \
                        --learning_rate 0.01 \
                        --batch_size 2000 \
                        --max_epoch 500 \
                        --eval_freq 10 \
```

> If you have any difficulty or question in running code and reproducing experimental results, please email to zqsun.nju@gmail.com and whu@nju.edu.cn.

### Disclaimer
The link prediction version of TransEdge is implemented based on the [open-source code of TransE](https://github.com/ZichaoHuang/TransE).

## Citation
If you use our model or code, please kindly cite it as follows:      
```
@inproceedings{TransEdge,
  author    = {Zequn Sun and Jiacheng Huang and Wei Hu and Muhao Chen and Lingbing Guo and Yuzhong Qu},
  title     = {TransEdge: Translating Relation-Contextualized Embeddings for Knowledge Graphs},
  booktitle = {ISWC},
  pages     = {612--629},
  year      = {2019}
}
```
