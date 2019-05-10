#! /usr/bin/env bash
wget -O gensim_1.model 'https://www.dropbox.com/s/b4we1ydzt276lba/gensim_1.model?dl=1'
wget -O gensim_no_jieba.model 'https://www.dropbox.com/s/ho598baur172j39/gensim_no_jieba.model?dl=1'
wget -O model_1_1.h5 'https://www.dropbox.com/s/m93w25txu9tcqbb/model_1_1.h5?dl=1'
wget -O model_1_2.h5 'https://www.dropbox.com/s/and7g3f4zojed97/model_1_2.h5?dl=1'
wget -O model_1_3.h5 'https://www.dropbox.com/s/pa4i9cwes3teeq7/model_1_3.h5?dl=1'
wget -O model_2_1.h5 'https://www.dropbox.com/s/06nwf2fk83fr1ve/model_2_1.h5?dl=1'
wget -O model_2_2.h5 'https://www.dropbox.com/s/07zn26fgk8roatk/model_2_2.h5?dl=1'
wget -O model_2_3.h5 'https://www.dropbox.com/s/n17syo3vyauqyzm/model_2_3.h5?dl=1'
wget -O model_3_1.h5 'https://www.dropbox.com/s/9dp4lzohaw5wq75/model_3_1.h5?dl=1'
wget -O model_no_jieba.h5 'https://www.dropbox.com/s/p9lyvg9wa3ebjne/model_no_jieba.h5?dl=1'
python3 predict.py $1 $2 $3