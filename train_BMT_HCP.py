import torch
import pandas as pd
import os
import numpy as np
import torch.nn.functional as F
import sklearn
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import svm
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset
from model.main_model import Brain_Mapping_Transformer


def seed_torch(seed):
#     random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    
    
def TF_train(x_train,y_train,x_test,y_test,index):
    batch_size = 256
    val_pre=0.2
    x_train,x_val,y_train,y_val =train_test_split(x_train,y_train,test_size=val_pre,stratify=y_train)
    size=np.shape(x_test)
    print(size)
    seed_torch(77)
    y_train=torch.LongTensor(y_train)
    y_train=F.one_hot(y_train)
    x_train=torch.FloatTensor(x_train)
    x_test=torch.FloatTensor(x_test)
    x_val=torch.FloatTensor(x_val)
    y_val=torch.LongTensor(y_val)
    y_test=torch.LongTensor(y_test)
    y_train_true=y_train.detach().numpy()
    y_train_true=np.argmax(y_train_true,axis=1)
    
    
    kernel=np.zeros((4,116,116))
    kernel[0]=np.ones((116,116))
    kernel[1,:40,:40]=np.ones((40,40))
    kernel[2,40:80,40:80]=np.ones((40,40))
    kernel[3,80:116,80:116]=np.ones((36,36))
    kernel=torch.FloatTensor(kernel)
     
    # ##################################################################3
    BMT=Brain_Mapping_Transformer(1200, 116, 4, 2, 0.05, 'Part')

    
    optimizer=torch.optim.Adam(BMT.parameters(),0.0002)
    loss_fn_1=torch.nn.CrossEntropyLoss()
    maxacc_val=0
    r1=0.01
    r2=0.2
    r3=0.1
    BMT = BMT.cuda()
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    kernel=kernel.cuda()
    for i in range(200):
        BMT.train()
        epoch_loss = 0
        bsize=0
        for x_batch, y_batch in train_loader:  # 使用小批次数据
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            y_predict = BMT(x_batch)
            loss=loss_fn_1(y_predict,y_batch.float())+r1*BMT.weight_loss(x_batch)+r2*BMT.relibility_loss(x_batch, kernel)+r3*BMT.sim_loss(x_batch)
            epoch_loss += loss.item()
            bsize+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if i%10==0:
            print(epoch_loss/bsize)
        BMT.eval()
        val_predictions = []
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch = x_val_batch.cuda()
                v_predict = BMT(x_val_batch)
                v_predict = v_predict.cpu().detach().numpy()
                v_predict = np.argmax(v_predict, axis=1)
                val_predictions.extend(v_predict)
    
        acc_val = accuracy_score(val_predictions, y_val.numpy())
        if acc_val>=maxacc_val:
            maxacc_val=acc_val
            torch.save(BMT,'/model/HCP_BMT_fold_'+str(index)+'.pkl')
    print('BMT FOLD '+str(index))
    BMT=torch.load('/model/HCP_BMT_fold_'+str(index)+'.pkl')
    BMT.eval()
    train_predictions = []
    val_predictions = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.cuda()
            train_predict = BMT(x_batch)
            train_predict = train_predict.cpu().detach().numpy()
            train_predict = np.argmax(train_predict, axis=1)
            train_predictions.extend(train_predict)
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch = x_val_batch.cuda()
            v_predict = BMT(x_val_batch)
            v_predict = v_predict.cpu().detach().numpy()
            v_predict = np.argmax(v_predict, axis=1)
            val_predictions.extend(v_predict)
    tr_acc=accuracy_score(y_train_true,train_predictions)
    val_acc=accuracy_score(y_val.cpu().detach().numpy(),val_predictions)
    del x_val
    del x_batch
    torch.cuda.empty_cache()

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_predictions=[]
    with torch.no_grad():
        for x_test_batch, y_test_batch in test_loader:
            x_test_batch = x_test_batch.cuda()
            test_predict = BMT(x_test_batch)
            test_predict = test_predict.cpu().detach().numpy()
            test_predict = np.argmax(test_predict, axis=1)
            test_predictions.extend(test_predict)
    acc=accuracy_score(y_test.cpu().detach().numpy(), test_predictions)
    print('train_acc='+str(tr_acc))
    print('val_acc='+str(val_acc))
    print('test_acc='+str(acc))
    recall=recall_score(y_test.cpu().detach().numpy(), test_predictions)
    prec=precision_score(y_test.cpu().detach().numpy(), test_predictions)
    try:
        auc=roc_auc_score(y_test.cpu().detach().numpy(), test_predictions)
    except:
        auc=0.5
    print('recall='+str(recall))
    print('precision='+str(prec))
    print('Auc='+str(auc))
    del x_test
    
    torch.cuda.empty_cache()

    return acc,recall,prec,auc



name_list=pd.read_csv('/data/hcp_data/clean_hcp.csv')
x_input=np.load('/data/hcp_data/aal_timeseries_norm.npy')
x_input=np.nan_to_num(x_input)
print(np.shape(x_input))

y_label=name_list['Gender']
y=np.zeros((len(y_label),))
 
for i in range(len(y_label)):
    if y_label[i]=='M':
        y[i]=0
    else:
        y[i]=1
seed_torch(67)
shix=np.random.permutation(np.arange(len(y_label)))
x_input=x_input[shix]
y=y[shix]
acc=[]
rec=[]
prec=[]
auc=[]
y_split = np.array_split(y, 10)
x_split = np.array_split(x_input, 10)
for i in range(10):
    x_test=x_split[i]
    y_test=y_split[i]
    x_train=[]
    y_train=[]
    for j in range(10):
        if j!=i:
            if len(x_train) == 0:
                x_train=x_split[j]
                y_train=y_split[j]
            else:
                x_train=np.concatenate((x_train,x_split[j]),axis=0)
                y_train=np.concatenate((y_train,y_split[j]),axis=0)
    a,r,p,au=TF_train(x_train,y_train,x_test,y_test,i)
    acc.append(a)
    rec.append(r)
    prec.append(p)
    auc.append(au)
print(np.mean(acc))
print(np.std(acc))
print(np.mean(rec))
print(np.std(rec))
print(np.mean(prec))
print(np.std(prec))
print(np.mean(auc))
print(np.std(auc))