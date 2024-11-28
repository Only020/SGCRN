
dataset=pems08

python -m train -d $dataset -m 0.25 --emb_size 32

python -m train -d $dataset -m 0.5  --emb_size 32

python -m train -d $dataset -m 0.75 --emb_size 32

python -m train -d $dataset -m 0.9 --emb_size 16


dataset=metr-la

python -m train -d $dataset -m 0.25 

python -m train -d $dataset -m 0.5  

python -m train -d $dataset -m 0.75 --emb_size 32

python -m train -d $dataset -m 0.9  --emb_size 32

dataset=pems04

python -m train -d $dataset -m 0.25 

python -m train -d $dataset -m 0.5  

python -m train -d $dataset -m 0.75 

python -m train -d $dataset -m 0.9  

dataset=PEMS-BAY

python -m train -d $dataset -m 0.25 

python -m train -d $dataset -m 0.5  --emb_size 32

python -m train -d $dataset -m 0.75 

python -m train -d $dataset -m 0.9   --emb_size 32
