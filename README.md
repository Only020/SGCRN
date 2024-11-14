# Simple Graph Convolutional Recurrent Network for Missing Node Traffic Flow Prediction

All datasets can be found in this link: https://github.com/ChengqingYu/MTS_dataset

Node missing generation code is available at https://github.com/GestaltCogTeam/GinAR.git

The following is the meaning of the core hyperparameter:
- input_len: The length of historical observation 
- num_id: The number of Nodes
- out_len: The length of forecasting steps 
- in_size:  The number of input features (Details you can refer to: https://github.com/zezhishao/BasicTS)
- emb_size: Embedding size
- grap_size: Node embedding size
- layer_num: The number of SGCRN layer
- dropout: dropout
- adj_mx: Adjacency matrix. (Details you can refer to: https://github.com/zezhishao/BasicTS)
- if_spatial: Spatial embedding
- if_day_in_week: Day Embedding
- if_time_in_day: Weekly Embedding

