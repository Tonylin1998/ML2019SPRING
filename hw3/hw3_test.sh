#! /usr/bin/env bash
wget -O model_1_0.h5 'https://www.dropbox.com/s/mbkrtqx3inzh8ts/model_1_0.h5?dl=1'
wget -O model_1_1.h5 'https://www.dropbox.com/s/q03jcmtfgbf3eh2/model_1_1.h5?dl=1'
wget -O model_1_2.h5 'https://www.dropbox.com/s/xgjx03e24i4qiii/model_1_2.h5?dl=1'
wget -O model_2_0.h5 'https://www.dropbox.com/s/82icbo8j7pw4oz2/model_2_0.h5?dl=1'
wget -O model_2_1.h5 'https://www.dropbox.com/s/qu3if7s1w5jvdi4/model_2_1.h5?dl=1'
wget -O model_2_2.h5 'https://www.dropbox.com/s/646wi3exm514tak/model_2_2.h5?dl=1'
wget -O model_3_0.h5 'https://www.dropbox.com/s/q42kmmf0k9rsvh6/model_3_0.h5?dl=1'
wget -O model_3_1.h5 'https://www.dropbox.com/s/e18uu3vw17756s3/model_3_1.h5?dl=1'
wget -O model_3_2.h5 'https://www.dropbox.com/s/wy3nvih09ziw86w/model_3_2.h5?dl=1'
python3 ensemble.py $1 $2