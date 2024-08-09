import os
import sys
import torch
import numpy as np
import pandas as pd
import csv
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_data, generate_representation, getBatch, get_padding
from model.attention.SelfAttention import ScaledDotProductAttention

device = "cuda:1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

torch.manual_seed(1)

parameter = sys.argv
print(parameter)


    
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

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim):
        super(LSTMTagger, self).__init__()
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

        
        
        t_emb = embding
    
        origin_emb = t_emb
        attention_dis = []

     
        total_attention_store = []
        for i,fig in enumerate(self.Att_config):
            vec_s = []
            att_s = []
            for k in range(fig):
                vec,att_ = self.Att_li[i][k](t_emb)
                vec = self.pro_li[i][k](vec)
                vec = F.relu(vec)
                vec_s.append(vec)
                att_s.append(att_)
            total_attention_store.append(att_s)

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
        
        sum_,final_att = self.AAt(t_emb)
        um_ = torch.cat((sum_,ott),1)       
        um_ = self.dropout(um_)    
        P = torch.sigmoid(self.hidden2p(um_))
      
        return P,attention_dis,total_attention_store,final_att
        
    

    


from utils import save_mydict,load_mydict



EMB_DIM=1280
model2 = LSTMTagger( EMB_DIM, 1280).cuda()
Path_ = "./NLS_loc_modeltes"
model2.load_state_dict(torch.load(Path_,map_location=device))



import sys



def random_streth(segg,set_len,max_len):
    import random as rd
    fis = segg[0]
    then = segg[1]
    while int(then-fis)!=set_len:
        flag = rd.choice([0,1])
        if flag == 0 and fis>=1:
            fis=fis - 1
        elif then<max_len:
            then+=1
        if then>= max_len and fis==0:
            break
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
            


# print(names_d)
def get_embedding(seq_li):
    
    train_total=generate_representation([1]*len(seq_li),seq_li)
    return train_total


def get_distribu(model2,to_cal):
    att_dis = []
    with torch.no_grad():
        model2.eval()
        embedding = to_cal[0].unsqueeze(0)
        # tag_scores = model2(pad_sequence(embedding,batch_first=True).cuda())
        tag_scores = model2(embedding.cuda())
        pre = torch.stack(tag_scores[1]).sum(0).cpu().detach().numpy()
        for rep in pre:
            att_dis.append(rep)
    return att_dis,tag_scores[2],tag_scores[3]
 



def seg_generate(pre_seg_att,cofactor,recom_complete,f_c,seqlen,stret_choose):
    import numpy as np
    pre_seg = []
    for fig in pre_seg_att:
        y= F.softmax(torch.tensor(fig).reshape(-1)).cpu().numpy()
 
        p_y = np.log(300*y)
        c = F.relu(torch.tensor(p_y))
        import numpy as np
        length = min(seqlen,1022)
        find_way = int(cofactor*length)
        if find_way==0 or find_way==1:
            find_way = 2
        segment = np.sort(np.argsort(-np.array(y))[0:find_way])
      
        begin = segment[0]
        seg_d = []
        
        merg_l = 2
        for i,index in enumerate(segment):
  
            if i+1<len(segment) and np.absolute(segment[i]-segment[i+1])>merg_l:
    #             print(segment[i])
                
                seg_d.append((begin,segment[i]))
                begin = segment[i+1]
        seg_d.append((begin,segment[-1]))
        
      
       
        if f_c == "f":
            seg_d = filter_single(seg_d)
        
        
        ts_seg_d = []

        if recom_complete>2:
           
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




import csv
import time
import os

def get_calculatetime_result(seq, embeding, cofactor):
    recom_complete = 3
    stret_choose = 'n'
    f_c = 'n'

    seqlen = len(seq)

    pre_seg_att,total_attention,final_att = get_distribu(model2, embeding)

    pre_seg = seg_generate(pre_seg_att, cofactor, recom_complete, f_c, seqlen, stret_choose)
   


    return pre_seg, total_attention

def cover_cal(pre_result, ac_label):
    num_p = 0
    p_num = 0
    for ac in pre_result:
        pre_s = pre_result[ac]
        num_p+=len(pre_s)
        if ac not in ac_label:
            continue
        NLS_LOC = ac_label[ac]
        # print(pre_s)
        # print(NLS_LOC)
        
        for sgs in pre_s:
            for item in NLS_LOC:
                if ju_cover(sgs,item):
                    p_num+=1
                    break
                    
    return num_p,p_num

def overlap_length(segment1, segment2):
    start1, end1 = segment1
    start2, end2 = segment2

    # overlap length
    overlap = max(0, min(end1, end2) - max(start1, start2))

    # not overlap length
    if overlap!=0 :
        non_overlap = end1-start1+1+end2-start2+1 -2* (overlap+1)
    else :
        return 0,0

    return overlap, non_overlap



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





 


def check_and_return(EXNES):
    Ok_data=[]
    for item in EXNES:
        seq = item[0]
        la = item[1]
        if la[0]>=len(seq):
            continue
        else:
            Ok_data.append(item)
    return Ok_data

import os
path = './Multi_data'

das_for_a = load_mydict('./Multi_data/Swiss_2024_EXP_NLS_data')
# das_for_a = []
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000255')
# das_for_a += total_gp 
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000256')
# das_for_a += total_gp 
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000250')
# das_for_a += total_gp 
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000303')
# das_for_a += total_gp 
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000305')
# das_for_a += total_gp 
# print(das_for_a[0])

das_for_a = check_and_return(das_for_a )
   


import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_target_cumulative = pd.DataFrame(columns=['Layer', 'Count'])
data_apc_total = pd.DataFrame()

seq_store = []
for singel_d in das_for_a:
    seq = singel_d[0]
    
    if len(seq)>1022:
        print(singel_d[1])
        seq_store.append(seq[0:1022])
    else:
        seq_store.append(seq)
embeding_t= get_embedding(seq_store)



import pandas as pd


def pos_li2seg(seq,seg):
    res = []
    # print(seg)
    for sgs in seg[0]:
        
        res.append(seq[sgs[0]:sgs[1]])
    return res

def select_generate(total_attention,ac,labels,seqlen,seq,cofactor,select_li):
    recom_complete = 3
    stret_choose = 'n'
    f_c = 'n'

    num_layers = len(total_attention)
    num_heads = len(total_attention[0])
    #
    all_layer_vis = {}
    for i, layer_att in enumerate(total_attention):
        layer_vis_a = []
        for j in range(num_heads):
            attention_matrix = layer_att[j].cpu().numpy().reshape((len(layer_att[j]), -1))
            layer_vis_a.append(attention_matrix)
        # Sum the attention matrices for the current layer
        all_layer_vis[i+1] = layer_vis_a

    
    
    Total_apc = []
    Total_target = []
    Total_frequent_seg = []

    total_select = None

    for layer_index in all_layer_vis:
        Lay_att = all_layer_vis[layer_index]

        Layer_apc = []
        Layer_target = []
        Layer_frequent_seg = []

        for bau_index,bau_att in enumerate(Lay_att):
            
            if select_li[layer_index-1][bau_index]==1:
                if  total_select is None:
                    total_select  = bau_att
                else:

                    total_select += bau_att



    if total_select is not None:
        pre_seg = seg_generate(total_select, cofactor, recom_complete, f_c, seqlen, stret_choose)
    else:
        pre_seg = [[]]
    label = labels
    Hit = 0
    apcli = []
    #计算是否命中
    if label[0]!=label[1]:
        Flag = False
        for seg in pre_seg[0]:
            if ju_cover(seg,label)==True:
                Flag = True
                break
        if Flag == True:
            Hit = 1
        else:
            Hit = 0
        
        for seg in pre_seg[0]:
            if ju_cover(seg,label)==True:
                overlap,nooverlap = overlap_length(seg, label)
                if (overlap+nooverlap)!=0:
                    apc_s = overlap/(overlap+nooverlap)
                    apcli.append(apc_s)
    else:
        for seg in pre_seg[0]:
            if label[0]<=seg[1] and label[0]>=seg[0] :
                call_num+=1
                apcli.append(1)
                break 
    
    if apcli!=[]:
        T_aPC = max(apcli)
    else:
        T_aPC = 0       

          


    return Hit,T_aPC


index = 0

T_APC = []
T_TARGET = []
T_FREQUENT = []



from utils import load_mydict
statis_target = load_mydict('statis_target')
statis_apc=load_mydict('statis_apc')



def sele_liss_generate(t1,t2):
    total_select = []
    number_ac = 0
    for layer_apc,layer_target in zip(statis_apc, statis_target):
        layer_select = []
        for bau_apc,bau_target in zip(layer_apc,layer_target):
            if bau_apc>=t1 and bau_target>=t2:
                layer_select.append(1)
                number_ac+=1
            else:
                layer_select.append(0)
        total_select.append(layer_select)
    return total_select
# print(f'You have selected {number_ac} BAUs as attention generator')


max_t1 = 0.442
max_t2 = 295

# select_li= load_mydict('select_li')
t1 = max_t1*0.8
t2 = max_t2*0.8
select_li= sele_liss_generate(t1,t2)
save_mydict(select_li,'per80_select_li')
all_hit = [1 for i in range(64)]
full_li = [all_hit for _ in range(16)]


Total_attt_storeg = []


all_l_hit = []
all_l_apc = []

sel_l_hit= []
sel_l_apc= []

ac_values = []
for singel_d in das_for_a:
    seq = singel_d[0]
    labels = singel_d[1]
    ac = singel_d[2]
    ac_values.append(ac)
    seqlen = len(seq)
    embeding = embeding_t[index]
    index+=1

    cofactor = 0.10

   

    preseg, total_attention = get_calculatetime_result(seq, [embeding], cofactor)

    Total_attt_storeg.append(total_attention)

    alls_Hit,alls_apc = select_generate(total_attention,ac,labels,seqlen,seq,cofactor,full_li)
    all_l_hit.append(alls_Hit)
    all_l_apc.append( alls_apc)

    Hit,T_aPC = select_generate(total_attention,ac,labels,seqlen,seq,cofactor,select_li)
    sel_l_hit.append(Hit)
    sel_l_apc.append(T_aPC)

    print(f'The {index}th record')
    print('Selectall:')
    print(f' apc is {alls_apc}')

    print('Selectstrategy:')
    print(f' apc is {T_aPC}')




# select_li= load_mydict('select_li')
t1 = 0.35
t2 = 250
select_li= sele_liss_generate(t1,t2)

all_hit = [1 for i in range(64)]
full_li = [all_hit for _ in range(16)]




import csv
import numpy as np


max_t1 = 0.442
max_t2 = 295


results_tot_hit = {}
results_avg_apc = {}

for perc in range(0, 100, 10):
    perc = perc / 100
    print(f'------now percent is {perc}------')
    t1 = max_t1 * perc
    t2 = max_t2 * perc

    select_li = sele_liss_generate(t1, t2)
    
  
    results_tot_hit[perc*100] = {}
    results_avg_apc[perc*100] = {}

    for cofactor in range(3, 100):
        
        cofactor = cofactor / 100
        print(f'cofactor vary at {cofactor}')
        sel_l_hit = []
        sel_l_apc = []
        for i, singel_d in enumerate(das_for_a):

            seq = singel_d[0]
            labels = singel_d[1]
            ac = singel_d[2]

            seqlen = len(seq)
            embeding = embeding_t[i]
            index += 1

            Hit, T_aPC = select_generate(Total_attt_storeg[i], ac, labels, seqlen, seq, cofactor, select_li)

            sel_l_hit.append(Hit)
            sel_l_apc.append(T_aPC)

        tot_hit = np.sum(sel_l_hit)
        avg_apc = np.mean(sel_l_apc)
        
        # 保存结果
        results_tot_hit[perc*100][cofactor] = tot_hit
        results_avg_apc[perc*100][cofactor] = avg_apc


# with open('exexp2-tot_hit_results.csv', 'w', newline='') as csvfile:
#     fieldnames = ['Cofactor'] + [f'Perc_{perc}' for perc in range(0, 100, 10)]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for cofactor in range(3, 100):
#         cofactor = cofactor / 100
#         row = {'Cofactor': cofactor}
#         row.update({f'Perc_{perc}': results_tot_hit[perc][cofactor] for perc in range(0, 100, 10)})
#         writer.writerow(row)


# with open('exexp2-avg_apc_results.csv', 'w', newline='') as csvfile:
#     fieldnames = ['Cofactor'] + [f'Perc_{perc}' for perc in range(0, 100, 10)]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     for cofactor in range(3, 100):
#         cofactor = cofactor / 100
#         row = {'Cofactor': cofactor}
#         row.update({f'Perc_{perc}': results_avg_apc[perc][cofactor] for perc in range(0, 100, 10)})
#         writer.writerow(row)


