import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from omegaconf import DictConfig
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from model.bnt import BrainNetworkTransformer


"Correlation Encoder"   
 
class TSM_multiKH(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout):
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
    
    
class Brain_Mapping_Transformer(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, AttenMeth='All'):
        super(Brain_Mapping_Transformer, self).__init__()
        self.encoder=TSM_multiKH(n_time, n_ROI, n_kernel, n_head, dropout)
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



class Brain_Mapping_Transformer_pretrain(Module):
    def __init__(self, n_time, n_ROI, n_kernel, n_head, dropout, m_kernel=False, m_head=False, AttenMeth='All'):
        super(Brain_Mapping_Transformer_pretrain, self).__init__()
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