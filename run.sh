# # model_id=5
dataset=pems08

python -u train.py -d $dataset -m 0.25 

python -u train.py -d $dataset -m 0.5  

python -u train.py -d $dataset -m 0.75 --emb_size 32

python -u train.py -d $dataset -m 0.9 --emb_size 8

dataset=pems04

python -u train.py -d $dataset -m 0.25 

python -u train.py -d $dataset -m 0.5  

python -u train.py -d $dataset -m 0.75 

python -u train.py -d $dataset -m 0.9  

dataset=metr-la

python -u train.py -d $dataset -m 0.25 

python -u train.py -d $dataset -m 0.5  

python -u train.py -d $dataset -m 0.75 

python -u train.py -d $dataset -m 0.9  


dataset=PEMS-BAY

python -u train.py -d $dataset -m 0.25 

python -u train.py -d $dataset -m 0.5  

python -u train.py -d $dataset -m 0.75 

python -u train.py -d $dataset -m 0.9  