#! /usr/bin/env bash
python3 model_1_train.py $1 $2 $3 $4
python3 model_2_train.py $1 $2 $3 $4
python3 model_3_train.py $1 $2 $3 $4
python3 model_no_jieba_train.py $1 $2 $3 $4