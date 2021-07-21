# UNICORN

The implementation of _Unified Conversational Recommendation Policy Learning via Graph-based Reinforcement Learning_ (SIGIR 2021). 

The code is partially referred to https://cpr-conv-rec.github.io/. 

## Data Preparation
1. Please download the datasets "SCPR_Data.zip" from https://cpr-conv-rec.github.io/, including lastfm, lastfm_start, and yelp. (If you would like to use your own dataset, please follow the same data format.)
2. Upzip "SCPR_Data.zip" and put "data" folder in the path "unicorn/". 
3. Processing data: `python graph_init.py --data_name <data_name>`
4. Use TransE from [[OpenKE](https://github.com/thunlp/OpenKE)] to pretrain the graph embeddings. And put the pretrained embeddings under "unicorn/tmp/<data_name>/embeds/". Or you can directly download the pretrained TransE embeddings from https://drive.google.com/file/d/1qoZMbYCBi2Y4IsJBdJ8Eg6y30Ap0gsQY/view?usp=sharing.

## Training
`python RL_model.py --data_name <data_name>`

## Evaluation
`python evaluate.py --data_name <data_name> --load_rl_epoch <checkpoint_epoch>`

## Citation
If the code is used in your research, please star this repo and cite our paper as follows:
```
@inproceedings{DBLP:conf/sigir/DengL0DL21,
  author    = {Yang Deng and
               Yaliang Li and
               Fei Sun and
               Bolin Ding and
               Wai Lam},
  title     = {Unified Conversational Recommendation Policy Learning via Graph-based
               Reinforcement Learning},
  booktitle = {{SIGIR} '21: The 44th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Virtual Event, Canada, July
               11-15, 2021},
  pages     = {1431--1441},
  publisher = {{ACM}},
  year      = {2021},
}
```
