import torch
# python 单个序列推荐_小模型.py 0.15 4 f
#  python 多个序列推荐_小模型.py 0.15 5 n s
#0.15表示抓取的片段比例 4代表最小的片段长度 f表示要不要过滤掉单个位点
#绷不住了 大模型没有小模型效果好 被直接秒杀

#这个综合了切片功能

#python 多个序列推荐_大模型_频繁片段分析.py 0.30 4 n s


setdivice = 1
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
# device=torch.device(f"cuda:{setdivice}" if torch.cuda.is_available() else "cpu")
# device2 = torch.device("cpu")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from utils import get_data,generate_representation,getBatch,get_padding
from model.attention.SelfAttention import ScaledDotProductAttention
import torch
#测试   定义第二类神经网络 无多头注意力版
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np




import pandas as pd
import sys
parameter = sys.argv
print(parameter)




#正式模型部分
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
#测试   定义第二类神经网络 无多头注意力版
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)
class Attention(nn.Module):

    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        #最后投影到概率上面去
            
    def forward(self, embding):
        
        rate_matrix = self.atten_Matrix(embding)
        rate_matrix = self.relu(rate_matrix)
        att_rate = F.softmax(rate_matrix,dim=1)
        sum_ = (embding*att_rate).sum(1)
#         print(sum.size())
      
        return sum_,att_rate
    
import math   
class Attention(nn.Module):

    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        #最后投影到概率上面去
        self.ll =    nn.Embedding(500,hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, embding):

        rate_matrix = self.atten_Matrix(embding)
        rate_matrix = self.relu(rate_matrix)
        att_rate = F.softmax(rate_matrix,dim=1)
        lll= rate_matrix.size()[1]
        sum_ = (embding*att_rate).sum(1)/math.sqrt(lll)
        # len_emb = self.ll(torch.tensor(lenttth).cuda()).unsqueeze(0)
        # sum_ = len_emb*sum_
#         print(sum.size())
        sum_ = self.layer_norm(sum_)
        return sum_,att_rate

#测试   定义第二类神经网络 无多头注意力版
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
torch.manual_seed(1)

#下面这个是结合了LSTM与各个位置注意力机制的网络
#下面这个是结合了LSTM与各个位置注意力机制的网络
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
        self.Att_config = [64]*16
        self.dropout = nn.Dropout(p=0.1)
        self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,2,bidirectional=True,batch_first = True)
        
        # The linear layer that maps from hidden state space to tag space
        real_dim = hidden_dim


        
        #储存attention 层的部分
        Att_li = []
        for fig in self.Att_config:
            sub_li = []
            for k in range(fig):
                sub_li.append(Attention(real_dim))
            Att_li.append(nn.ModuleList(sub_li))
        self.Att_li = nn.ModuleList(Att_li)    
        self.AAt = Attention(real_dim)
        
        #压缩得到的vec维度
        pro_li = []
        for fig in self.Att_config:
            sub_li = []
            to_dim = real_dim/fig
#             print(to_dim)
            for k in range(fig):
                sub_li.append(nn.Linear(real_dim,int(to_dim)))
            pro_li.append(nn.ModuleList(sub_li))
        self.pro_li = nn.ModuleList(pro_li)    
        
        
        #用于将综合好的embedding投影到高维以方便产生乘积
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(1,real_dim))
        self.project_li = nn.ModuleList(project_li)
        
                
        #线性变换填充维度
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(real_dim,real_dim))
        self.FI_li = nn.ModuleList(project_li)

        #压缩一下避免爆炸
        set_dim = int(real_dim/4)
        project_li = []
        for i in range(len(self.Att_config)):
            project_li.append(nn.Linear(real_dim,set_dim))
        self.Set_li = nn.ModuleList(project_li)
       
        #最后投影到概率上面去
        length = (len(self.Att_config))*set_dim+real_dim
        self.hidden2p = nn.Linear(length,1)

        #norm 层
        norm_li = []
        for i in range(len(self.Att_config)):
            
            norm_li.append(nn.LayerNorm(real_dim))
        self.norm_li = nn.ModuleList(norm_li)
        
    def forward(self, embding):
#         lstm_out, _ = self.lstm(embding)
#         lstm_out = self.dropout(lstm_out)
        batch_size = (embding.size()[0])
        vec_store = []
#         for_sum = torch.cat((embding,embding),2)
        
        
        t_emb = embding
        #储存一开始的emb_
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
#             print(att_s.size())
            #综合每个query生成attention
            sum_att = att_s.sum(0).unsqueeze(2)
            attention_dis.append(att_s.sum(0).unsqueeze(2))
            sum_att = self.project_li[i](sum_att)
#             print(sum_att.size())
           
            #生成下一层的embedding,并且使用layernorm
#             t_emb = (t_emb*sum_att)+t_emb+origin_emb
            t_emb = (t_emb*sum_att)+t_emb
            t_emb = self.norm_li[i](t_emb)
            
 #生成每个query 生成的vector
            vec_s =  torch.stack(vec_s)
            vec_s = vec_s.transpose(0,1)            
#             print(t_emb.size())
#             t_emb = self.encod[i](t_emb)
#             下面是生成每个层的total_embedding方法
            z = vec_s
            z = z.permute(0,2,1)
            batch = z.size()[0]
            output = z.reshape(batch,-1)
            #线性变换后填充维度
            output = F.relu(self.FI_li[i](output))+output
            output = F.relu(self.Set_li[i](output))
            vec_store.append(output)
            
        
        ott = torch.stack(vec_store).transpose(0,1).reshape(batch_size,-1)
        
        sum_,_ = self.AAt(t_emb)
        um_ = torch.cat((sum_,ott),1)       
        um_ = self.dropout(um_)    
        P = torch.sigmoid(self.hidden2p(um_))
      
        return P,attention_dis
        

from utils import save_mydict,load_mydict
#在这里选择和调整要计算的序列
# print(seq_li)





#加载模型
EMB_DIM=1280
model2 = LSTMTagger( EMB_DIM, 1280).cuda()
# Path_ = "./4_enhance60_one_last"
Path_ = "./Peptide_recomendation/NLS_loc_modeltes"

model2.load_state_dict(torch.load(Path_,map_location={'cuda:0':'cuda:1'}))


# print(len(record.seq))
# print("*"*300)
#选出长度小于1024的序列 存在字典和列表里面

def get_embedding(seq_li):
    mul_datt=[]


    for seq in seq_li:
        if len(seq)>=1023:
            seq = seq[0:1022]
        mul_datt.append(seq)
    

    for item in mul_datt:
        if len(item)<1:
            print(item)

    #运行这一段，重点是seq是我们要的seq
    to_gene = mul_datt
    train_total=generate_representation([1]*len(to_gene),to_gene)
    return train_total[0]





from utils import getBatch
from torch.nn.utils.rnn import pad_sequence,pack_sequence


def get_distribu(model2,to_cal):
    att_dis = []
    with torch.no_grad():
        model2.eval()
        embedding = to_cal.unsqueeze(0)
        # tag_scores = model2(pad_sequence(embedding,batch_first=True).cuda())
        tag_scores = model2(embedding.cuda())
        pre = torch.stack(tag_scores[1]).sum(0).cpu().detach().numpy()
        for rep in pre:
            att_dis.append(rep)
    return att_dis


# print(len(mul_datt))


# thred = 0.3

import sys



def random_streth(segg,set_len,max_len):
    import random as rd
    fis = segg[0]
    then = segg[1]
    while int(then-fis)!=set_len:
        flag = rd.choice([0,1])
        if flag == 0 and fis>1:
            fis=fis - 1
        elif then<max_len:
            then+=1
    return (fis,then)
    
def filter_single(pre_seg):
    pre_seg_without_single = []
    for th in pre_seg:
        # short_li= []
        if th[0]!=th[1]:
            # short_li.append(th)
        # if short_li != []:
            pre_seg_without_single.append(th)
    return pre_seg_without_single
            
            
    
# from matplotlib import pyplot as plt
def seg_generate(pre_seg_att,cofactor,recom_complete,f_c,seqlen,stret_choose):
    import numpy as np
    pre_seg = []
    for fig in pre_seg_att:
        y= F.softmax(torch.tensor(fig).reshape(-1)).cpu().numpy()
    #     print(y)
        p_y = np.log(300*y)
        c = F.relu(torch.tensor(p_y))
        import numpy as np
        length = min(seqlen,1022)
        find_way = int(cofactor*length)
        segment = np.sort(np.argsort(-np.array(y))[0:find_way])
        begin = segment[0]
        seg_d = []
        merg_l = 2
        for i,index in enumerate(segment):
        #     print(segment[i])
            if i+1<len(segment) and np.absolute(segment[i]-segment[i+1])>merg_l:
    #             print(segment[i])
                
                seg_d.append((begin,segment[i]))
                begin = segment[i+1]
        seg_d.append((begin,segment[-1]))
        #选择是否过滤掉单个位点
        # print(seg_d)
        if f_c == "f":
            seg_d = filter_single(seg_d)
        
        # print(seg_d)
        ts_seg_d = []
        #选择是否有补齐机制
        if recom_complete>2:
            #在最短长度的基础上根据采样系数随机延长
            import random as rd
            add_len = rd.choices([0,1,2],[0.5,0.3,0.2])[0]
            recom_complete = recom_complete + add_len
            for segg in seg_d:
                if segg[1]-segg[0]<recom_complete:
                    ts_segg = random_streth(segg,recom_complete,seqlen)
                    ts_seg_d.append(ts_segg)
                else:
                    ts_seg_d.append(segg)
        
        
        if stret_choose == "s":
            st_li = []
            import random as rd
            
            
            for segg in ts_seg_d:
                add_len = rd.choices([3,4,5,6,7],[0.3,0.2,0.2,0.2,0.1])[0]
                recom_complete = segg[1]-segg[0] + add_len
                ts_segg = random_streth(segg,recom_complete,seqlen)
                st_li.append(ts_segg)
            return [st_li]
        
        pre_seg.append(ts_seg_d)
    return pre_seg


#     break
    

    
#我们现在需要不考虑单个位点信息的
# 单个位点信息去除函数



def get_embedding(seq_li):
    
    train_total=generate_representation([1]*len(seq_li),seq_li)
    return train_total









def ju_cover(seg,label):
    if seg[0]<=label[1] and label[0]<=seg[0]:
        return True
    if seg[1]<=label[1] and label[0]<=seg[1]:
        return True
    if seg[0]<=label[0] and seg[1]>=label[1]:
        return True
    if seg[0]>=label[0] and seg[1]<=label[1]:
        return True
    return False

def get_seq_li(total_gp):
    seq_li = []
    for data in total_gp:
        if len(data[0])>1022:
            seq = data[0][:1022]
        else:
            seq = data[0]
        seq_li.append(seq)
    return seq_li



def get_seq_seg(seq,preseg):
    seg = preseg[0]
    seq_li = []
    for se in seg:
        seg_s = seq[se[0]:se[1]]
        seq_li.append(str(seg_s))
    return seq_li
        

# print(names_d)
# 此函数包含了切片程序，把长度超过1022的按照1022切片成块
def get_store_data(nd,slice_len):
    data_store = []
    seq_li = []
    for record in SeqIO.parse(path+'/'+nd, "fasta"):
        # print(record.seq)
        # print(record.id)
        ID = record.id
        if ID[:2]=='sp':
            AC = ID.split('|')[1]
        else:
            AC = ID
        # print(AC)
        
        seq = record.seq
        
        if len(seq)>1022:
            # data_store.append([seq[:1022],AC])
            # seq_li.append(str(seq[:1022]))
            num_o_seg = int(len(seq)/slice_len)
            
            for sl in range(num_o_seg+1):
                if sl<num_o_seg:
                    slice_seg =seq[sl*slice_len:(sl+1)*slice_len]
                    
                else:
                    slice_seg =seq[sl*slice_len:]
                    
                if len(slice_seg)<50:
                    continue
            
                data_store.append([slice_seg,AC+f'-{sl+1}'])
                seq_li.append(str(slice_seg))
            
            
            
        
        else:
            data_store.append([seq,AC])
            seq_li.append(str(seq))
        
        
        
    return data_store,seq_li



def get_recommend_result(data_store,embeding_gp,sa_name):
     
    total_num = 0
    call_num = 0  
    pre_seg_att_li = []
    
    data_s = []
    for data,to_cal in zip(data_store,embeding_gp):

        seq = str(data[0])
        if len(seq)>1022:
            seq = seq[0:1022]
        ac = data[1]
        #这个label其实没啥用
        label = (1,3)
        
        # print(to_cal)
        pre_seg_att = get_distribu(model2,to_cal)
        print('yys')
        #方便下面统计数据用
        pre_seg_att_li.append(pre_seg_att)
        cofactor = float(parameter[1])
        recom_complete = int(parameter[2])
        stret_choose = parameter[4]
        seqlen = len(data[0])
        f_c = parameter[3]
        print('fd')
        pre_seg = seg_generate(pre_seg_att,cofactor,recom_complete,f_c,seqlen,stret_choose)
        print(pre_seg)
        print('fd2')
        data_for_rec = [seq,label,ac,pre_seg[0]]
        data_s.append(data_for_rec)
        print(pre_seg )
        print( get_seq_seg(data[0],pre_seg))
        print('-------------')
        
    path = './progres/recomenda_trans'
    print('saving')
    save_mydict(data_s,path+f'/{sa_name}')



import os

from Bio import SeqIO
import os
path = './Peptide_recomendation/single_input'
names_d = os.listdir(path)

for file_ in names_d:
    if 'pdb' in file_:
        continue
    print(file_)
    print(file_[:-6])
    print('---------------------s-------------------')
    data_store,seq_li = get_store_data(file_,1022)
    # seq_li = get_seq_li(data_store)
    
    
    embeding_gp = get_embedding(seq_li)
    get_recommend_result(data_store,embeding_gp,file_[:-6])