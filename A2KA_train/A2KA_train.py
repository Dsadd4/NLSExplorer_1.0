import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_data, generate_representation, getBatch, get_padding

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")
torch.cuda.set_device(1)

torch.manual_seed(1)



BaseDir = './deeploc_dataset/'
object = 'Swissprot_Train_Validation_dataset.csv'
Path = BaseDir +object
data = pd.read_csv(Path)
namelist=[]
for i in range(0,11):
    namelist.append(data.columns[4:-1].values[i])


from utils import save_mydict,load_mydict


chosen = 5
import pandas as pd
BaseDir = './deeploc_dataset/'
object = 'Swissprot_Train_Validation_dataset.csv'
Path = BaseDir +object
data = pd.read_csv(Path)
name2seq = {}
name2label = {}
name_list = []
for item in zip(data['ACC'],data['Sequence']):
    name2seq[item[0]]=item[1]
    name_list.append(item[0])

namelist=[]
for i in range(0,11):
    namelist.append(data.columns[4:-1].values[i])

print(namelist)

print('-----------')


# print(f'we have chosen {namelist[chosen]}')

for item in zip(data['ACC'],data[namelist[chosen]]):
    name2label[item[0]]=int(item[1])

trainning_f = []
for item in name_list:
    trainning_f.append((name2label[item],name2seq[item]))






import numpy as np
def train(model, criterion, optimizer, vector, tags):
    # print(vector)
    vector = list(vector)+[torch.randn(1022,1280)]
    # print(vector)
    vector = pad_sequence(vector,batch_first=True).cuda().float()[:-1,:,:]
    
    # vector = vector.cuda().float()

    tags = torch.tensor(tags).cuda()

    model.zero_grad()
    tag_scores,_ = model(vector)
    print('---------*********--------')
    print(tags[0:1])
    print(tag_scores[0:1])


    loss = criterion(tag_scores.reshape(-1), tags.float())
   
    loss.backward()
    optimizer.step()

    return model, loss

from torch.nn.utils.rnn import pad_sequence,pack_sequence
def begin_train_dur(mul_y, model2, optimizer, train_total,Batchsize,stages):
  
    import random

    torch.cuda.empty_cache()
    total = train_total
    # total = get_padding(train_total)

    train_data = zip(total, mul_y)
    trainning_data = []
    for item in train_data:
        trainning_data.append(item)
    random.shuffle(trainning_data)

    criterion = nn.BCELoss()


    for item in getBatch(Batchsize, trainning_data):
        vector, tags = zip(*item)
        model2, loss = train(model2, criterion, optimizer, vector, tags)
        print(f"epochs:{stages}")
        print(f'the loss is {loss}')
        del vector, tags
    return model2




    
import math   
class Attention(nn.Module):

    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
class A2KA(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(A2KA, self).__init__()
        self.Att_config = [64]*16
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,2,bidirectional=True,batch_first = True)
        
        # The linear layer that maps from hidden state space to tag space
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

        norm_li = []
        for i in range(len(self.Att_config)):
            
            norm_li.append(nn.LayerNorm(real_dim))
        self.norm_li = nn.ModuleList(norm_li)
        
    def forward(self, embding):

        batch_size = (embding.size()[0])
        vec_store = []
#         for_sum = torch.cat((embding,embding),2)
        
        
        t_emb = embding
 
        origin_emb = t_emb
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


from tqdm import tqdm
EMB_DIM=1280
hidden_dim = EMB_DIM
model2 = A2KA( EMB_DIM, 1280).cuda()
learning_rate = 0.00001


mul_y = []
datts=[]
for item in trainning_f:
    mul_y.append(item[0])
    datts.append(item[1])

mul_datt=[]
for item in datts:
    if len(item)>=1023:
        item = item[0:1022]
    mul_datt.append(item)
    
nucleus =  load_mydict('./Dataset/swiss_all_loc_in_nucleus')
nucleus_not =  load_mydict('./Dataset/swiss_all_not_loc_in_nucleus')    

_,nu_se = zip(*nucleus)
_,notnu_se = zip(*nucleus_not)

def norm_seqli(datt):
    temp = []
    for item in datt:
        if len(item)>=1023:
            item = item[0:1022]
        temp.append(item)
    return temp
nu_se = norm_seqli(nu_se)
notnu_se = norm_seqli(notnu_se)


def generate_train(nu_se,notnu_se):
    import random
    random.shuffle(notnu_se)
    random.shuffle(nu_se)
    pos_seq = nu_se[0:15000]
    neg_seq = notnu_se[:len(pos_seq)]
    y1=[1]*len(pos_seq)
    y2 = [0]*len(pos_seq)
    pos_train = generate_representation([1]*len(pos_seq),pos_seq)
    neg_train = generate_representation([1]*len(neg_seq),neg_seq)
    train_total=pos_train+neg_train
    y = y1+y2
    ss_train = list(zip(train_total,y))
    random.shuffle(ss_train)
    train_total,y = zip(*ss_train)
    return train_total,y

train_total,y = generate_train(nu_se,notnu_se)

augumentation = "tes"
Batchsize=24
optimizer = optim.Adam(model2.parameters(), lr=learning_rate)
for stages in range(120):
            
    model2 = begin_train_dur(y,model2,optimizer,train_total,Batchsize,stages)
    if stages%5 == 0:
        train_total,y = generate_train(nu_se,notnu_se)
    if stages%15==0:  
        torch.save(model2.state_dict(), f'./NLS_loc_model_{stages}{augumentation}')
        
torch.save(model2.state_dict(), f'./NLS_loc_model{augumentation}')
