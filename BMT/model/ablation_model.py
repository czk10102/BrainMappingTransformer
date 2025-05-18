import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from omegaconf import DictConfig
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from bnt import BrainNetworkTransformer
import longformer

class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor) -> torch.tensor:
        pass


class E2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)


class BrainNetCNN(BaseModel):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.in_planes = 1
        self.d = config.dataset.node_sz

        self.e2econv1 = E2EBlock(1, 32, config.dataset.node_sz, bias=True)
        self.e2econv2 = E2EBlock(32, 64, config.dataset.node_sz, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.d))
        self.N2G = torch.nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = torch.nn.Linear(256, 128)
        self.dense2 = torch.nn.Linear(128, 30)
        self.dense3 = torch.nn.Linear(30, 2)

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):
        node_feature = node_feature.unsqueeze(dim=1)
        out = F.leaky_relu(self.e2econv1(node_feature), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(
            self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.leaky_relu(
            self.dense1(out), negative_slope=0.33), p=0.5)
        out = F.dropout(F.leaky_relu(
            self.dense2(out), negative_slope=0.33), p=0.5)
        out = F.leaky_relu(self.dense3(out), negative_slope=0.33)
#         out = F.log_softmax(out,dim=1)

        return out


class TSM_single(Module):
    def __init__(self, n_time, n_ROI, dropout):
        super(TSM_single, self).__init__()
        self.kernel_1=nn.TransformerEncoderLayer(d_model=n_ROI, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True)        
        self.kernel_2=nn.TransformerEncoderLayer(d_model=n_time, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True)
        self.l0=nn.Linear(2*n_ROI,1)
    def forward(self, input):
#         [batch_size,num_node,num_time]=input.shape
        time_pointer_1=self.kernel_1(torch.transpose(input, 1, 2))
        time_pointer_2=self.kernel_2(input)
        time_pointer=torch.cat((time_pointer_1, torch.transpose(time_pointer_2, 1, 2)), dim=2)
        time_pointer=self.l0(time_pointer)
        time_pointer=torch.tanh(time_pointer)
        time_pointer=F.relu(time_pointer)
        time_pointer=torch.squeeze(time_pointer)
        sum=torch.sum(time_pointer,dim=1)
        w_in=torch.einsum('bit,bt->bit',input,time_pointer)        
        c=torch.einsum('bit,bjt->bij',input,w_in)
        c=c*torch.reciprocal(sum).unsqueeze(1).unsqueeze(1)   
        return c
     
    def get_timelabel(self, input):
        time_pointer_1=self.kernel_1(torch.transpose(input, 1, 2))
        time_pointer_2=self.kernel_2(input)
        time_pointer=torch.cat((time_pointer_1, torch.transpose(time_pointer_2, 1, 2)), dim=2)
        time_pointer=self.l0(time_pointer)
        time_pointer=torch.tanh(time_pointer)
        time_pointer=F.relu(time_pointer)
        time_pointer=torch.squeeze(time_pointer)
         
        return time_pointer
     
    def weight_loss(self, input):
        kw=self.get_timelabel(input)
        mk=kw-0.5 
        return -torch.norm(mk, p=1)/torch.numel(mk)
    def relibility_loss(self, input):
        input_1=input[:,:,::2]
        input_2=input[:,:,1::2]
 
        time_pointer=self.get_timelabel(input)
        tp1=time_pointer[:,::2]
        tp2=time_pointer[:,1::2]
 
        sum1=torch.sum(tp1,dim=1)
        w_in_1=torch.einsum('bit,bt->bit',input_1,tp1)        
        c1=torch.einsum('bit,bjt->bij',input_1,w_in_1)
        c1=c1*torch.reciprocal(sum1).unsqueeze(1).unsqueeze(1)        
        sum2=torch.sum(tp2,dim=1)
        w_in_2=torch.einsum('bit,bt->bit',input_2,tp2)        
        c2=torch.einsum('bit,bjt->bij',input_2,w_in_2)
        c2=c2*torch.reciprocal(sum2).unsqueeze(1).unsqueeze(1)     
        cc=c1-c2
        return torch.norm(cc, p=1)/cc.shape[1]/cc.shape[0]/cc.shape[2]
    
class TSM_multiKH(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False):
        super(TSM_multiKH, self).__init__()
        self.kernel_1=nn.TransformerEncoderLayer(d_model=n_ROI, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True)        
        self.kernel_2=nn.TransformerEncoderLayer(d_model=n_time, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True)
        self.l0=nn.Linear(2*n_ROI,n_kernel*n_head)
        self.n_head=n_head
    def forward(self, input):
#         [batch_size,num_node,num_time]=input.shape
        time_pointer_1=self.kernel_1(torch.transpose(input, 1, 2))
        time_pointer_2=self.kernel_2(input)
        time_pointer=torch.cat((time_pointer_1, torch.transpose(time_pointer_2, 1, 2)), dim=2)
        time_pointer=self.l0(time_pointer)
        time_pointer=torch.tanh(time_pointer)
        time_pointer=F.relu(time_pointer)
        time_pointer=torch.squeeze(time_pointer)
        sum=torch.sum(time_pointer,dim=1)
        w_in=torch.einsum('bit,btk->bkit',input,time_pointer)        
        c=torch.einsum('bit,bkjt->bkij',input,w_in)
        c=c*torch.reciprocal(sum).unsqueeze(2).unsqueeze(2)   
        return c
     
    def get_timelabel(self, input):
        time_pointer_1=self.kernel_1(torch.transpose(input, 1, 2))
        time_pointer_2=self.kernel_2(input)
        time_pointer=torch.cat((time_pointer_1, torch.transpose(time_pointer_2, 1, 2)), dim=2)
        time_pointer=self.l0(time_pointer)
        time_pointer=torch.tanh(time_pointer)
        time_pointer=F.relu(time_pointer)
        time_pointer=torch.squeeze(time_pointer)
         
        return time_pointer
     
    def weight_loss(self, input):
        kw=self.get_timelabel(input)
        mk=kw-0.5 
        return -torch.norm(mk, p=1)/torch.numel(mk)
    def relibility_loss(self, input, kernel):
        input_1=input[:,:,::2]
        input_2=input[:,:,1::2]
 
        time_pointer=self.get_timelabel(input)
        tp1=time_pointer[:,::2]
        tp2=time_pointer[:,1::2]
 
        sum1=torch.sum(tp1,dim=1)
        w_in_1=torch.einsum('bit,btk->bkit',input_1,tp1)        
        c1=torch.einsum('bit,bkjt->bkij',input_1,w_in_1)
        c1=c1*torch.reciprocal(sum1).unsqueeze(2).unsqueeze(2)
         
        sum2=torch.sum(tp2,dim=1)
        w_in_2=torch.einsum('bit,btk->bkit',input_2,tp2)        
        c2=torch.einsum('bit,bkjt->bkij',input_2,w_in_2)
        c2=c2*torch.reciprocal(sum2).unsqueeze(2).unsqueeze(2)
         
        cc=c1-c2
        kk=kernel.repeat(self.n_head,1,1)
        cc=cc*kk
        return torch.norm(cc, p=1)/torch.norm(kk, p=1)/cc.shape[0]


# class TSM_multiKH(Module):
#     def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False):
#         super(TSM_multiKH, self).__init__()
#         self.kernel_Conv=nn.TransformerEncoderLayer(d_model=n_ROI, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True)        
# #         self.kernel_2=nn.TransformerEncoderLayer(d_model=n_time, nhead=2, dim_feedforward=128, dropout=dropout, batch_first=True, norm_first=True)
#         self.l0=nn.Linear(n_ROI,n_kernel*n_head)
#         self.n_head=n_head
#     def forward(self, input):
# #         [batch_size,num_node,num_time]=input.shape
#         time_pointer=self.kernel_Conv(torch.transpose(input, 1, 2))
#         time_pointer=self.l0(time_pointer)
#         time_pointer=torch.tanh(time_pointer)
#         time_pointer=F.relu(time_pointer)
#         time_pointer=torch.squeeze(time_pointer)
#         sum=torch.sum(time_pointer,dim=1)
#         w_in=torch.einsum('bit,btk->bkit',input,time_pointer)        
#         c=torch.einsum('bit,bkjt->bkij',input,w_in)
#         c=c*torch.reciprocal(sum).unsqueeze(2).unsqueeze(2)   
#         return c
#     
#     def get_timelabel(self, input):
#         time_pointer=self.kernel_Conv(torch.transpose(input, 1, 2))
# 
# 
#         time_pointer=self.l0(time_pointer)
#         time_pointer=torch.tanh(time_pointer)
#         time_pointer=F.relu(time_pointer)
#         time_pointer=torch.squeeze(time_pointer)
#         
#         return time_pointer
#     
#     def weight_loss(self, input):
#         kw=self.get_timelabel(input)
#         mk=kw-0.5 
#         return -torch.norm(mk, p=1)/torch.numel(mk)
#     def relibility_loss(self, input, kernel):
#         input_1=input[:,:,::2]
#         input_2=input[:,:,1::2]
# 
#         time_pointer=self.get_timelabel(input)
#         tp1=time_pointer[:,::2]
#         tp2=time_pointer[:,1::2]
# 
#         sum1=torch.sum(tp1,dim=1)
#         w_in_1=torch.einsum('bit,btk->bkit',input_1,tp1)        
#         c1=torch.einsum('bit,bkjt->bkij',input_1,w_in_1)
#         c1=c1*torch.reciprocal(sum1).unsqueeze(2).unsqueeze(2)
#         
#         sum2=torch.sum(tp2,dim=1)
#         w_in_2=torch.einsum('bit,btk->bkit',input_2,tp2)        
#         c2=torch.einsum('bit,bkjt->bkij',input_2,w_in_2)
#         c2=c2*torch.reciprocal(sum2).unsqueeze(2).unsqueeze(2)
#         
#         cc=c1-c2
#         kk=kernel.repeat(self.n_head,1,1)
#         cc=cc*kk
#         return torch.norm(cc, p=1)/torch.norm(kk, p=1)/cc.shape[0]    

class TSM_multiKH_L(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False):
        super(TSM_multiKH_L, self).__init__()
        config_1 = DictConfig({
            'hidden_size': n_time,
            'node_size':n_ROI,
            'num_attention_heads':n_head,
            'attention_probs_dropout_prob':dropout,
            'attention_window':[2,2],
            'attention_dilation':[1,1],
            'attention_mode':'sliding_chunks',
            'autoregressive':False
        })
        config_2 = DictConfig({
            'hidden_size': n_ROI,
            'node_size':n_time,
            'num_attention_heads':n_head,
            'attention_probs_dropout_prob':dropout,
            'attention_window':[2,2],
            'attention_dilation':[1,1],
            'attention_mode':'sliding_chunks',
            'autoregressive':False
        })
#         self.kernel_1=Longformer.longformer.LongformerSelfAttention(config_1,1)
#         self.kernel_2=Longformer.longformer.LongformerSelfAttention(config_2,2)
        self.kernel_1=longformer.LongformerSelfAttention(config_1,0)
        self.kernel_2=longformer.LongformerSelfAttention(config_2,1)
        self.l0=nn.Linear(2*n_ROI,n_kernel*n_head)
        self.n_head=n_head
    def forward(self, input):
#         [batch_size,num_node,num_time]=input.shape
        time_pointer_1=self.kernel_1(input)
        time_pointer_2=self.kernel_2(torch.transpose(input, 1, 2))
#         time_pointer_1=self.kernel_2(input)
#         time_pointer_2=self.kernel_1(torch.transpose(input, 1, 2))
        time_pointer=torch.cat((time_pointer_2[0], torch.transpose(time_pointer_1[0], 1, 2)), dim=2)
        time_pointer=self.l0(time_pointer)
        time_pointer=torch.tanh(time_pointer)
        time_pointer=F.relu(time_pointer)
        time_pointer=torch.squeeze(time_pointer)
        sum=torch.sum(time_pointer,dim=1)
        w_in=torch.einsum('bit,btk->bkit',input,time_pointer)        
        c=torch.einsum('bit,bkjt->bkij',input,w_in)
        c=c*torch.reciprocal(sum).unsqueeze(2).unsqueeze(2)   
        return c
     
    def get_timelabel(self, input):
        time_pointer_1=self.kernel_1(input)
        time_pointer_2=self.kernel_2(torch.transpose(input, 1, 2))
        time_pointer=torch.cat((time_pointer_2[0], torch.transpose(time_pointer_1[0], 1, 2)), dim=2)
        time_pointer=self.l0(time_pointer)
        time_pointer=torch.tanh(time_pointer)
        time_pointer=F.relu(time_pointer)
        time_pointer=torch.squeeze(time_pointer)
         
        return time_pointer
     
    def weight_loss(self, input):
        kw=self.get_timelabel(input)
        mk=kw-0.5 
        return -torch.norm(mk, p=1)/torch.numel(mk)
    def relibility_loss(self, input, kernel):
        input_1=input[:,:,::2]
        input_2=input[:,:,1::2]
 
        time_pointer=self.get_timelabel(input)
        tp1=time_pointer[:,::2]
        tp2=time_pointer[:,1::2]
 
        sum1=torch.sum(tp1,dim=1)
        w_in_1=torch.einsum('bit,btk->bkit',input_1,tp1)        
        c1=torch.einsum('bit,bkjt->bkij',input_1,w_in_1)
        c1=c1*torch.reciprocal(sum1).unsqueeze(2).unsqueeze(2)
         
        sum2=torch.sum(tp2,dim=1)
        w_in_2=torch.einsum('bit,btk->bkit',input_2,tp2)        
        c2=torch.einsum('bit,bkjt->bkij',input_2,w_in_2)
        c2=c2*torch.reciprocal(sum2).unsqueeze(2).unsqueeze(2)
         
        cc=c1-c2
        kk=kernel.repeat(self.n_head,1,1)
        cc=cc*kk
        return torch.norm(cc, p=1)/torch.norm(kk, p=1)/cc.shape[0]
 

    
class KAM(Module):
    def __init__(self, n_ROI, n_slice, AttenMeth='All'):
        super(KAM, self).__init__()
        if AttenMeth=='All':
            self.atten_w=Parameter(torch.zeros((n_slice)))
        elif AttenMeth=='Part':
            self.atten_w=Parameter(torch.zeros((n_slice, n_ROI, n_ROI)))
        else:
            print('Error')
        self.atm=AttenMeth
        self.w=Parameter(torch.FloatTensor(n_slice, n_ROI, n_ROI))
        self.reset_parameters()
    def forward(self, input):
        output=torch.einsum('bsxy,syz->bsxz',input,self.w)
        output=output+input
        if self.atm=='All':
            k=torch.sigmoid(self.atten_w)
            k=F.softmax(k,dim=0)
            output=torch.einsum('bsxy,s->bxy',output,k)
        elif self.atm=='Part':
            k=torch.sigmoid(self.atten_w)
            k=F.softmax(k,dim=0)
            output=torch.einsum('bsxy,sxy->bxy',output,k)
        return output
    def reset_parameters(self):
        stdv = np.sqrt(3.0/(self.w.size(1)))
#         stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
    def get_kernel_weight(self):
        k=torch.sigmoid(self.atten_w)
        k=F.softmax(k,dim=0)
        return k
    def sim_loss(self, input):
        output=torch.einsum('bsxy,syz->bsxz',input,self.w)
        output=output+input
        stand=output.std(dim=1)
        return stand.mean()

class TSR_BrainNet(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False, AttenMeth='All'):
        super(TSR_BrainNet, self).__init__()
        if m_kernel and m_head:
            self.encoder=TSM_multiKH(n_time, n_ROI, n_kernel, n_head, dropout, m_kernel, m_head)
            self.atten=KAM(n_ROI, n_kernel*n_head, AttenMeth=AttenMeth)
        config = DictConfig({
            'dataset': {
                'name': 'MyDataset',
                'batch_size': 64,
                'shuffle': True,
                'node_sz':n_ROI
            }
        })
        self.decoder=BrainNetCNN(config)
    def forward(self, input):
        c=self.encoder(input)
        c=self.atten(c)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def sim_loss(self,input):
        c=self.encoder(input)
        return self.atten.sim_loss(c)
    def get_kernel_weight(self):
        return self.atten.get_kernel_weight()
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input, kernel):
        return self.encoder.relibility_loss(input, kernel)
    def get_low_level(self,input):
        return self.encoder(input)
    def get_high_level(self,input):
        c=self.encoder(input)
        return self.atten(c)
    
    
    
class TSR_BrainNet_S_K_F(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False, AttenMeth='All'):
        super(TSR_BrainNet_S_K_F, self).__init__()
        if m_kernel and m_head:
            self.encoder=TSM_multiKH(n_time, n_ROI, n_kernel, n_head, dropout, m_kernel, m_head)
            self.atten=KAM(n_ROI, n_kernel*n_head, AttenMeth=AttenMeth)
        config = DictConfig({
            'model': {
                'name': 'MyModel',
                'layers': [128, 64, 32],
                'learning_rate': 0.001,
#                 'pos_encoding': 'identity',
                'pos_encoding': 'none',
                'sizes':[100,n_ROI,n_ROI],
                'pooling': [True, False, True],
                'orthogonal': True,
                'freeze_center':False,
                'project_assignment':True,
                'pos_embed_dim':4
        },
            'dataset': {
                'name': 'MyDataset',
                'batch_size': 64,
                'shuffle': True,
                'node_sz':n_ROI
        }
    })
        self.decoder=BrainNetworkTransformer(config)
    def forward(self, input):
        c=self.encoder(input)
        c=self.atten(c)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def sim_loss(self,input):
        c=self.encoder(input)
        return self.atten.sim_loss(c)
    def get_kernel_weight(self):
        return self.atten.get_kernel_weight()
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input, kernel):
        return self.encoder.relibility_loss(input, kernel)
    def get_low_level(self,input):
        return self.encoder(input)
    def get_high_level(self,input):
        c=self.encoder(input)
        return self.atten(c)
    
    
    
class TSR_BrainNet_L_K_F(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False, AttenMeth='All'):
        super(TSR_BrainNet_L_K_F, self).__init__()
        if m_kernel and m_head:
            self.encoder=TSM_multiKH_L(n_time, n_ROI, n_kernel, n_head, dropout, m_kernel, m_head)
            self.atten=KAM(n_ROI, n_kernel*n_head, AttenMeth=AttenMeth)
        config = DictConfig({
            'model': {
                'name': 'MyModel',
                'layers': [128, 64, 32],
                'learning_rate': 0.001,
#                 'pos_encoding': 'identity',
                'pos_encoding': 'none',
                'sizes':[100,n_ROI,n_ROI],
                'pooling': [True, False, True],
                'orthogonal': True,
                'freeze_center':False,
                'project_assignment':True,
                'pos_embed_dim':4
        },
            'dataset': {
                'name': 'MyDataset',
                'batch_size': 64,
                'shuffle': True,
                'node_sz':n_ROI
        }
    })
        self.decoder=BrainNetworkTransformer(config)
    def forward(self, input):
        c=self.encoder(input)
        c=self.atten(c)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def sim_loss(self,input):
        c=self.encoder(input)
        return self.atten.sim_loss(c)
    def get_kernel_weight(self):
        return self.atten.get_kernel_weight()
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input, kernel):
        return self.encoder.relibility_loss(input, kernel)
    def get_low_level(self,input):
        return self.encoder(input)
    def get_high_level(self,input):
        c=self.encoder(input)
        return self.atten(c)


class TSR_BrainNet_L_K_Fp(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False, AttenMeth='All'):
        super(TSR_BrainNet_L_K_Fp, self).__init__()
        if m_kernel and m_head:
            self.encoder=TSM_multiKH_L(n_time, n_ROI, n_kernel, n_head, dropout, m_kernel, m_head)
            self.atten=KAM(n_ROI, n_kernel*n_head, AttenMeth=AttenMeth)

        self.decoder=torch.load('/mnt/sdb/czk/model/ASD_BNT_pretrain.pkl')
    def forward(self, input):
        c=self.encoder(input)
        c=self.atten(c)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def sim_loss(self,input):
        c=self.encoder(input)
        return self.atten.sim_loss(c)
    def get_kernel_weight(self):
        return self.atten.get_kernel_weight()
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input, kernel):
        return self.encoder.relibility_loss(input, kernel)
    def get_low_level(self,input):
        return self.encoder(input)
    def get_high_level(self,input):
        c=self.encoder(input)
        return self.atten(c)


class TSR_BrainNet_S_K_Fp(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False, AttenMeth='All'):
        super(TSR_BrainNet_S_K_Fp, self).__init__()
        if m_kernel and m_head:
            self.encoder=TSM_multiKH(n_time, n_ROI, n_kernel, n_head, dropout, m_kernel, m_head)
            self.atten=KAM(n_ROI, n_kernel*n_head, AttenMeth=AttenMeth)
        
        self.decoder=torch.load('/mnt/sdb/czk/model/ASD_BNT_pretrain.pkl')
    def forward(self, input):
        c=self.encoder(input)
        c=self.atten(c)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def sim_loss(self,input):
        c=self.encoder(input)
        return self.atten.sim_loss(c)
    def get_kernel_weight(self):
        return self.atten.get_kernel_weight()
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input, kernel):
        return self.encoder.relibility_loss(input, kernel)
    def get_low_level(self,input):
        return self.encoder(input)
    def get_high_level(self,input):
        c=self.encoder(input)
        return self.atten(c)

class TSR_BrainNet_S_F(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False):
        super(TSR_BrainNet_S_F, self).__init__()
        if m_kernel and m_head:
            self.encoder=TSM_multiKH(n_time, n_ROI, n_kernel, n_head, dropout, m_kernel, m_head)
        config = DictConfig({
            'model': {
                'name': 'MyModel',
                'layers': [128, 64, 32],
                'learning_rate': 0.001,
#                 'pos_encoding': 'identity',
                'pos_encoding': 'none',
                'sizes':[100,n_ROI,n_ROI],
                'pooling': [True, False, True],
                'orthogonal': True,
                'freeze_center':False,
                'project_assignment':True,
                'pos_embed_dim':4
        },
            'dataset': {
                'name': 'MyDataset',
                'batch_size': 64,
                'shuffle': True,
                'node_sz':n_ROI
        }
    })
        self.decoder=BrainNetworkTransformer(config)
    def forward(self, input):
        c=self.encoder(input)
        c=torch.mean(c, dim=1, keepdim=False)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input, kernel):
        return self.encoder.relibility_loss(input, kernel)
    def get_low_level(self,input):
        return self.encoder(input)
    def get_high_level(self,input):
        c=self.encoder(input)
        return torch.mean(c, dim=1, keepdim=False)
        
class TSR_BrainNet_1_F(Module):
    def __init__(self, n_time, n_ROI, dropout):
        super(TSR_BrainNet_1_F, self).__init__()
        self.encoder=TSM_single(n_time, n_ROI, dropout)
        config = DictConfig({
            'model': {
                'name': 'MyModel',
                'layers': [128, 64, 32],
                'learning_rate': 0.001,
#                 'pos_encoding': 'identity',
                'pos_encoding': 'none',
                'sizes':[100,n_ROI,n_ROI],
                'pooling': [True, False, True],
                'orthogonal': True,
                'freeze_center':False,
                'project_assignment':True,
                'pos_embed_dim':4
        },
            'dataset': {
                'name': 'MyDataset',
                'batch_size': 64,
                'shuffle': True,
                'node_sz':n_ROI
        }
    })
        self.decoder=BrainNetworkTransformer(config)
    def forward(self, input):
        c=self.encoder(input)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input):
        return self.encoder.relibility_loss(input)
    def get_low_level(self,input):
        return self.encoder(input)

class TS_Corr(Module):
    def __init__(self, n_time, n_ROI, Win_Size, Win_Step,dropout):
        super(TS_Corr, self).__init__()
        
        self.kernel_size=Win_Size
        self.n_win=int((n_time-Win_Size)/Win_Step+1)
        self.kernel_Conv=torch.nn.Conv1d(n_ROI, 1, Win_Size,padding=int(Win_Size/2-0.5))
    def forward(self, input):
        [batch_size,num_node,num_time]=input.shape
        kernel_w=self.kernel_Conv(input)
        kernel_w=torch.mean(kernel_w,dim=1)
        kernel_w=torch.tanh(kernel_w)
        kernel_w=F.relu(kernel_w)
        kernel_w=torch.squeeze(kernel_w)
        sum=torch.sum(kernel_w,dim=1)

        w_in=torch.einsum('bit,bt->bit',input,kernel_w)
        
        c=torch.einsum('bit,bjt->bij',input,w_in)
        c=c*torch.reciprocal(sum).unsqueeze(1).unsqueeze(1)
       
        return c
    
    def get_timelabel(self, input):
        kernel_w=self.kernel_Conv(input)
        kernel_w=torch.mean(kernel_w,dim=1)
        kernel_w=torch.tanh(kernel_w)
        kernel_w=F.relu(kernel_w)
        kernel_w=torch.squeeze(kernel_w)
        
        return kernel_w
    
    def weight_loss(self, input):
        kw=self.get_timelabel(input)
        mk=kw-0.5
        
        return -torch.norm(mk, p=1)/torch.numel(mk)
    def relibility_loss(self, input):
        input_1=input[:,:,::2]
        input_2=input[:,:,1::2]
 
        time_pointer=self.get_timelabel(input)
        tp1=time_pointer[:,::2]
        tp2=time_pointer[:,1::2]
 
        sum1=torch.sum(tp1,dim=1)
        w_in_1=torch.einsum('bit,bt->bit',input_1,tp1)        
        c1=torch.einsum('bit,bjt->bij',input_1,w_in_1)
        c1=c1*torch.reciprocal(sum1).unsqueeze(1).unsqueeze(1)        
        sum2=torch.sum(tp2,dim=1)
        w_in_2=torch.einsum('bit,bt->bit',input_2,tp2)        
        c2=torch.einsum('bit,bjt->bij',input_2,w_in_2)
        c2=c2*torch.reciprocal(sum2).unsqueeze(1).unsqueeze(1)     
        cc=c1-c2
        return torch.norm(cc, p=1)/cc.shape[1]/cc.shape[0]/cc.shape[2]


class TSR_BrainNet_C_F(Module):
    def __init__(self, n_time, n_ROI, Win_Size, Win_Step, dropout):
        super(TSR_BrainNet_C_F, self).__init__()
        self.encoder=TS_Corr(n_time, n_ROI, Win_Size, Win_Step, dropout)
        config = DictConfig({
            'model': {
                'name': 'MyModel',
                'layers': [128, 64, 32],
                'learning_rate': 0.001,
#                 'pos_encoding': 'identity',
                'pos_encoding': 'none',
                'sizes':[100,n_ROI,n_ROI],
                'pooling': [True, False, True],
                'orthogonal': True,
                'freeze_center':False,
                'project_assignment':True,
                'pos_embed_dim':4
        },
            'dataset': {
                'name': 'MyDataset',
                'batch_size': 64,
                'shuffle': True,
                'node_sz':n_ROI
        }
    })
        self.decoder=BrainNetworkTransformer(config)
    def forward(self, input):
        c=self.encoder(input)
        c=self.decoder([],c)
        return c
    def get_timelabel(self,input):
        return self.encoder.get_timelabel(input)
    def weight_loss(self, input):
        return self.encoder.weight_loss(input)
    def relibility_loss(self, input):
        return self.encoder.relibility_loss(input)
    def get_low_level(self,input):
        return self.encoder(input)   