import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_mydict
from mamba import Mamba, MambaConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"




import math
class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
       
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, embding):
        lenttth = embding.size()[1]
        rate_matrix = self.atten_Matrix(embding)
        rate_matrix = self.relu(rate_matrix)
        att_rate = F.softmax(rate_matrix,dim=1)
        lll= rate_matrix.size()[1]
        sum_ = (embding*att_rate).sum(1)/math.sqrt(lll)
        sum_ = self.layer_norm(sum_)
        return sum_,att_rate
   
config = MambaConfig(d_model=16, n_layers=2)
model = Mamba(config)   
def threshold_ret(pre,threshold):
    recomd = []
    # print(pre)
    for idx,number in enumerate(pre[0]):
        # print(number)
        if number>= threshold:
            recomd.append(idx)
    return recomd

acc_li = []
f = open('./record6.txt','w+')
# r_model = model

  
        
def corr_r(seg_li,max_len):
    r_li = []
    for item in seg_li:
        if int(item[0])>=max_len-2:
            continue
        
        r_li.append(item)
    return r_li


#最新推荐出来的数据

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

def top_cout(number,ins_lab,recomd,pre_seg,topn):
    for item in ins_lab:
        for p_i in recomd[0][0:topn]:
            if ju_cover(item,pre_seg[p_i]):
                number+=1
                break
    return number 


class gnn_classifier(nn.Module):
    def __init__(self):
        super(gnn_classifier, self).__init__()
        hidden_dim=1280
        lstm_dim = int(hidden_dim/4)
        cls_dim = lstm_dim *2
        conv_dim = 256
        # self.cls = nn.Linear(cls_dim +conv_dim, 1)
        self.cls = nn.Linear(cls_dim , 1)
        self.lstm = nn.LSTM(int(hidden_dim), lstm_dim,1,bidirectional=True,batch_first = True)
        self.sigmoi = nn.Sigmoid()
        self.att_fu = Attention(cls_dim )
        config = MambaConfig(d_model= int(hidden_dim/2), n_layers=12)
        self.mamb = Mamba(config)
    def forward(self, embding,segment):
        embding = embding.cuda()
        max_len =embding.size()[1]
        pre = 'begjin'
        embding = embding.unsqueeze(0)
        # print(embding.size())
        for item in segment:
            if item[1]+1<=max_len:
                posi_emb = embding[:,item[0]:item[1]+1,:]
            else:
                posi_emb = embding[:,item[0]:max_len,:]
            
            te,_= self.lstm(posi_emb)
            # print(te.size())     
            te = self.mamb(te)+te
            sum_,_ = self.att_fu(te)
            # sum_ = torch.concat((sum_,tc),1)
            pre_di = self.sigmoi(self.cls(F.relu(sum_)))
            if pre == 'begjin':
                pre = pre_di 
            else:
                pre =torch.concat((pre,pre_di),1)
        return pre   

import os
def show_acc(X_te,embding_te,model,epoch):
    
    with torch.no_grad():

        ac_label = {}
        thres_f1_pre_acc = [{} for threshold in range(5,100)]
        thres = []
        f1 = []
        for j,ba_data in enumerate(X_te):
            
            ins_ac = ba_data[0]
            print(ba_data[1])
            ins_lab = [(ba_data[1][0][0]-1,ba_data[1][0][1]-1)]
            
                
            ac_label = add2dict(ac_label,ins_ac,ins_lab.copy())
            
            pre_seg = ba_data[2]
            
            pre = model(embding_te[j],pre_seg )
            recomd = torch.argsort(-1*pre,1)
            
            # print('-------------g----------------')
            index = 0
            for threshold in range(5,100):
                threshold = threshold /100
                if ins_ac not in thres_f1_pre_acc[index]:
                    for item in recomd[0]:
                        if  pre[0][item]>=threshold:
                            thres_f1_pre_acc[index] = add2dict( thres_f1_pre_acc[index],ins_ac,[pre_seg[item]].copy())
                index+=1
            
          
        
        index = 0
        for threshold in range(5,100):
            threshold = threshold /100
            # print(f'the epoch is{epoch}, threshold is{threshold}')
            thr_res = get_recallandpre(thres_f1_pre_acc[index], ac_label)
            
            if thr_res!=0:
                thres.append(threshold) 
                f1.append(thr_res[2])
            thr_apc = str(cal_apc(thres_f1_pre_acc[index], ac_label))
            print(f'the epoch is{epoch}, threshold is{threshold}, result is {thr_res}, apc is {thr_apc}')
            filename = f"./阈值导向训练/_{epoch}result.csv"
            filename = f"./阈值导向训练/hybrid_{epoch}result.csv"
            if not os.path.isfile(filename):
                with open(filename, 'w') as f:
                    f.write('threshold,recall,precision,f1,apc' + '\n')
                
            else:
                with open(filename, 'a') as f:
                  
                    if thr_res!=0:
                        f.write(f'{threshold},{thr_res[0]},{thr_res[1]},{thr_res[2]},{thr_apc}' + '\n')
            index+=1
      
                
        
        
        return f1,thres

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

def cal_apc(pre_result,ac_label):
    atp = 0
    afs = 0
    for ac in pre_result:
        pre_s = pre_result[ac]
        NLS_LOC = ac_label[ac]
        for sgs in pre_s:
            flag = 0
            for item in NLS_LOC:
                if ju_cover(sgs,item):
                    overlap,nooverlap = overlap_length(sgs,item)
                    flag = 1
                    break
            if flag == 0:
                overlap= 0
                nooverlap = sgs[1] - sgs [0]+1
                
                NLNO = []
                for item in NLS_LOC:
                    nlsno = item[1] - item [0]
                    NLNO.append(nlsno)
                # print(NLNO)
                nooverlap += min(NLNO)
                
            atp+= overlap
            afs+= nooverlap
    # print(atp)
    # print(afs)
    # print('------------')
    if (atp+afs) == 0:
        return 0
    return atp/(atp+afs)

def add2dict(label_dict, key, value):

    if key in label_dict:
        # 如果键已存在，则将值添加到对应列表中
        label_dict[key].extend(value)  # 使用 extend() 方法将值添加到列表中
    else:
        # 如果键不存在，则创建一个新的键值对
        label_dict[key] = value.copy()  # 使用 copy() 方法创建新的列表，以避免副作用
    return label_dict


def get_recallandpre(top_re,ac_label):
    num_p,p_num = cover_cal(top_re, ac_label)  
    if num_p==0 or p_num==0:
        return 0
    pre = p_num/num_p
    
    
    num_p,p_num = cover_cal(ac_label,top_re)    
   
    recall = p_num/num_p
    
    f1 = pre*recall*2/(pre+recall)
    return [recall,pre,f1]

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

import numpy as np
import random

import random


def get_result(data_store,tot_emb,data_store2,tot_emb2):
    for k in range(1):
        model = gnn_classifier().cuda() 
        model_path = ''
        try:

            model.load_state_dict(torch.load(model_path )) 
        except:
            print('Model load error, please specify your model path')
        epoch = 000
        f1,thress = show_acc(data_store2,tot_emb2,model,epoch)
        print(f1)       
    return 0


def get_data(insp_train):
    seq_store = []
    data_store = []
    label_store = []
    for st_d in insp_train:
        ins_seq = st_d[0]
        ins_label = st_d[1]
        ins_ac = st_d[2]
        ins_pre = st_d[3]
        data_store.append([ins_ac,[ins_label],ins_pre])
        seq_store.append(ins_seq)
    print('***************uuuuuuuuuuuuuuuu*****************')
    from utils import generate_representation
    emb_seq = generate_representation([1]*len(seq_store),seq_store)
    return data_store,emb_seq
    

insp_train = load_mydict('./for_recom/insp_train_0.6')
data_store,tot_emb = get_data(insp_train)
# insp_train = load_mydict(f'./for_recom/yeast_0.3')
insp_train = load_mydict('./for_recom/hybrd2_0.3')
data_store2,tot_emb2 = get_data(insp_train)
# print(len(data_store2))
# get the test result.
get_result(data_store,tot_emb,data_store2,tot_emb2)
