import argparse
import copy
import datetime
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from metric.mask_metric import masked_mae,masked_mape,masked_rmse,masked_mse
from model.arch import SGCRN
from data_solve import load_adj
import time

plt.rcParams['font.sans-serif']=['SimHei'] 
plt.rcParams['axes.unicode_minus']=False 

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def Inverse_normalization(x,max,min):
    return x * (max - min) + min

def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()

parser = argparse.ArgumentParser()
parser.add_argument("-d",'--data', type=str, default='PEMS08', help='data name')
parser.add_argument("-g","--gpu", type=int, default=0, help="gpu id")
parser.add_argument('-m','--mask', type=float, default=0.25, help='mask rate')
parser.add_argument("--epoch", type=int, default=100, help="epoch")
parser.add_argument("--emb_size", type=int, default=16, help="emb_size")
parser.add_argument("--grap_size", type=int, default=8, help="grap_size")
parser.add_argument("--if_spatial", type=int, default=16, help="if_spatial")
parser.add_argument("-day","--if_time_in_day", type=int, default=16, help="if_time_in_day")
parser.add_argument("-week","--if_day_in_week", type=int, default=16, help="if_day_in_week")
args = parser.parse_args()

### PEMS-BAY、METR-LA、PeMS04、PeMS08
data_name = args.data.upper()
data_file = "data/" + data_name + "/data.npz"
raw_data = np.load(data_file,allow_pickle=True)

### graph
adj_mx, _ = load_adj("MTS_data/" + data_name + "/adj_"  + data_name + ".pkl", "doubletransition")

print(raw_data.files)
batch_size = 64
epoch = args.epoch
IF_mask = args.mask
lr_rate = 0.006

### Hyperparameter
input_len= 12
if data_name == "PEMS-BAY":
    num_id = 325
elif data_name == "METR-LA":
    num_id = 207
elif data_name == "PEMS04":
    num_id = 307
elif data_name == "PEMS08":
    num_id= 170
out_len=12
in_size=3
emb_size=args.emb_size
grap_size = args.grap_size
layer_num = 2
dropout = 0.15
adj_mx = [torch.tensor(i).float() for i in adj_mx]
max_norm = 5      #Gradient pruning
max_num =  100

###learning rate
num_lr = 5
gamme = 0.5
milestone = [1,15,40,70,90]


now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


### Train_data
if IF_mask == 0.25:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_25"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_50"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
elif IF_mask == 0.75:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_75"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
elif IF_mask == 0.9:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_90"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
else:
    train_data = torch.cat([torch.tensor(raw_data["train_x_raw"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)

train_data = DataLoader(train_data,batch_size=batch_size,shuffle=True)


### Valid_data
if IF_mask == 0.25:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_25"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_50"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_75"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.9:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_90"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
else:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_raw"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)

valid_data = DataLoader(valid_data,batch_size=batch_size,shuffle=False)

### test_data
if IF_mask == 0.25:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_25"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_50"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_75"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.9:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_90"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
else:
    test_data = torch.cat([torch.tensor(raw_data["test_x_raw"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)

test_data = DataLoader(test_data,batch_size=batch_size,shuffle=False)

max_min = raw_data['max_min']
max_data, min_data = max_min[0],max_min[1]

###CPU and GPU
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


### model
model = SGCRN(input_len, num_id, out_len, in_size, emb_size, grap_size, layer_num, dropout, adj_mx,args.if_spatial,args.if_time_in_day,args.if_day_in_week)
model = model.to(device)
optimizer = optim.Adam(params=model.parameters(),lr=lr_rate)
num_vail = 0
min_vaild_loss = float("inf")
wait = 0
max_wait = 20

log_path = f"./study/logs/{data_name}/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
log = os.path.join(log_path, f"{data_name}-{int(IF_mask*100)}-{now}.log")
log = open(log, "a")
log.seek(0)
log.truncate()

save_path = f"./study/saved_models/{data_name}/{int(IF_mask*100)}_{now}/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
save = os.path.join(save_path, f"{data_name}-best.pt")

print_log('data_name:',data_name,'num_id:',num_id,'mask:',
          IF_mask,'layer_num:',layer_num,'epoch:',epoch,
          'emb_size:',emb_size,'grap_size:',grap_size,'if_spatial:',args.if_spatial,
            'if_time_in_day:',args.if_time_in_day,'if_day_in_week:',args.if_day_in_week,log=log)

### train
for i in range(epoch):
    num = 0
    loss_out = 0.0
    model.train()
    start = time.time()
    for data in train_data:
        model.zero_grad()

        train_feature = data[:, :, :,0:in_size].to(device)
        train_target = data[:, :, :,-1].to(device)
        train_pre = model(train_feature)
        loss_data = masked_mae(train_pre,train_target,0.0)

        num += 1
        loss_data.backward()

        if max_norm > 0 and i < max_num:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        else:
            pass
        num += 1

        optimizer.step()
        loss_out += loss_data
    loss_out = loss_out/num
    end = time.time()


    num_va = 0
    loss_vaild = 0.0
    model.eval()
    with torch.no_grad():
        for data in valid_data:

            valid_x = data[:, :, :,0:in_size].to(device)
            valid_y = data[:, :, :,-1].to(device)
            valid_pre = model(valid_x)
            loss_data = masked_mae(valid_pre, valid_y,0.0)

            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va


    if loss_vaild < min_vaild_loss:
        min_vaild_loss = loss_vaild
        best_state_dict = copy.deepcopy(model.state_dict())

        if i>=20:
            epoch_path = os.path.join(save_path, f"{data_name}_{i+1}.pt")
            torch.save(best_state_dict, epoch_path)

    if (i + 1) in milestone:
        for params in optimizer.param_groups:
            params['lr'] *= gamme
            params["weight_decay"] *= gamme

    print_log('Loss of the {} epoch of the training set: {:02.4f}, Loss of the validation set Loss:{:02.4f}, training time: {:02.4f}:'.format(i+1,loss_out,loss_vaild,end - start),log=log)

torch.save(best_state_dict, save)
model.load_state_dict(best_state_dict)
model = model.to(device)
model.eval()

with torch.no_grad():
    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in test_data:
        test_feature = data[:, :, :,0:in_size].to(device)
        test_target = data[:, :, :,-1].to(device)
        test_pre = model(test_feature)
        if num == 0:
            all_pre = test_pre
            all_true = test_target
        else:
            all_pre = torch.cat([all_pre, test_pre], dim=0)
            all_true = torch.cat([all_true, test_target], dim=0)
        num += 1

final_pred = Inverse_normalization(all_pre, max_data, min_data)
final_target = Inverse_normalization(all_true, max_data, min_data)


mae,mape,rmse = masked_mae(final_pred, final_target,0.0),\
                masked_mape(final_pred, final_target,0.0)*100,masked_rmse(final_pred, final_target,0.0)
mae,mape,rmse = mae.cpu().numpy(),mape.cpu().numpy(),rmse.cpu().numpy()
out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
    rmse,
    mae,
    mape,
)

print_log(out_str, log=log)

log.close()
final_pred = final_pred.cpu().numpy()
final_target = final_target.cpu().numpy()

np.savez_compressed(
    os.path.join(save_path, f"{data_name}-pred.npz"),
    pred=final_pred,
    true=final_target,
)


