import torch
#A2KA Module
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")
torch.manual_seed(1)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(1)

    
import math   
class Attention(nn.Module):

    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        self.ll =    nn.Embedding(500,hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, embding):

        rate_matrix = self.atten_Matrix(embding)
        rate_matrix = self.relu(rate_matrix)
        att_rate = F.softmax(rate_matrix,dim=1)
        lll= rate_matrix.size()[1]
        sum_ = (embding*att_rate).sum(1)/math.sqrt(lll)
        sum_ = self.layer_norm(sum_)
        return sum_,att_rate
torch.manual_seed(1)



class FeedForward_Norm(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        super(FeedForward_Norm, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Feedforward neural network with residual connection and layer normalization
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm(src)
        return src 


class A2KA(nn.Module):

    def __init__(self, hidden_dim,config):
        super(A2KA, self).__init__()
        self.Att_config = config
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim

        real_dim = hidden_dim
  
        Att_li = []
        for fig in self.Att_config:
            sub_li = []
            for k in range(fig):
                sub_li.append(Attention(real_dim))
            Att_li.append(nn.ModuleList(sub_li))
        self.Att_li = nn.ModuleList(Att_li)    
        self.AAt = Attention(real_dim)
        
     
        pro_li = []
        for fig in self.Att_config:
            sub_li = []
            to_dim = real_dim/fig
#             print(to_dim)
            for k in range(fig):
                sub_li.append(nn.Linear(real_dim,int(to_dim)))
            pro_li.append(nn.ModuleList(sub_li))
        self.pro_li = nn.ModuleList(pro_li)    
        
 
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(1,real_dim))
        self.project_li = nn.ModuleList(project_li)
        
                

        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(real_dim,real_dim))
        self.FI_li = nn.ModuleList(project_li)

  
        set_dim = int(real_dim/4)
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(real_dim,set_dim))
        self.Set_li = nn.ModuleList(project_li)
       

        length = (len(self.Att_config))*set_dim+real_dim
        self.hidden2p = nn.Linear(length,1)

  
        #the norm function is variable, we provide two kinds of example for A2KA
        self.norm_li = nn.ModuleList([FeedForward_Norm(hidden_dim) for _ in range(len(self.Att_config))])
        # self.norm_li = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(len(self.Att_config))])
    def forward(self, embding):

        batch_size = (embding.size()[0])
        vec_store = []

        t_emb = embding
        attention_dis = []
        for i,fig in enumerate(self.Att_config):
            vec_s = []
            att_s = []
            for k in range(fig):
                vec,att_ = self.Att_li[i][k](t_emb)
                vec = self.pro_li[i][k](vec)
                vec = F.relu(vec)
                vec_s.append(vec)
                att_s.append(att_)
            att_s = torch.stack(att_s).squeeze(3)

        
            sum_att = att_s.sum(0).unsqueeze(2)
            attention_dis.append(att_s.sum(0).unsqueeze(2))
            sum_att = self.project_li[i](sum_att)

            t_emb = (t_emb*sum_att)+t_emb
            t_emb = self.norm_li[i](t_emb)

            vec_s =  torch.stack(vec_s)
            vec_s = vec_s.transpose(0,1)            

            z = vec_s
            z = z.permute(0,2,1)
            batch = z.size()[0]
            output = z.reshape(batch,-1)
     
            output = F.relu(self.FI_li[i](output))+output
            output = F.relu(self.Set_li[i](output))
            vec_store.append(output)
            
        
        ott = torch.stack(vec_store).transpose(0,1).reshape(batch_size,-1)
        
        sum_,_ = self.AAt(t_emb)
        um_ = torch.cat((sum_,ott),1)       
        um_ = self.dropout(um_)    
        P = torch.sigmoid(self.hidden2p(um_))
        return P,attention_dis
    
# example:
# hidden_dimention = 512
# config = [6,12,12,5]
# model =A2KA( hidden_dimention,config)