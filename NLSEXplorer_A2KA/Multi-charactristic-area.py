import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_data, generate_representation, getBatch, get_padding, save_mydict, load_mydict


device = "cuda:1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

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
#         lstm_out, _ = self.lstm(embding)
#         lstm_out = self.dropout(lstm_out)
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
#             print(att_s.size())
           
            sum_att = att_s.sum(0).unsqueeze(2)
            attention_dis.append(att_s.sum(0).unsqueeze(2))
            sum_att = self.project_li[i](sum_att)
#             print(sum_att.size())
           
            
#             t_emb = (t_emb*sum_att)+t_emb+origin_emb
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
        


EMB_DIM=1280
model2 = LSTMTagger( EMB_DIM, 1280).cuda()
Path_ = "./NLS_loc_modeltes"
model2.load_state_dict(torch.load(Path_,map_location=device))




def get_embedding(seq_li):
    mul_datt=[]


    for seq in seq_li:
        if len(seq)>=1023:
            seq = seq[0:1022]
        mul_datt.append(seq)
    

    for item in mul_datt:
        if len(item)<1:
            print(item)

    to_gene = mul_datt
    train_total=generate_representation([1]*len(to_gene),to_gene)
    return train_total[0]






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


def get_seq_li(total_gp):
    seq_li = []
    for data in total_gp:
        if len(data[0])>1022:
            seq = data[0][:1022]
        else:
            seq = data[0]
        seq_li.append(seq)
    return seq_li
#     break
    


    




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
       
        label_dict[key].extend(value)  
    else:
        
        label_dict[key] = value.copy()  
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
 
def get_recall_result(total_gp,embeding_gp,parameter):
    total_num = 0
    call_num = 0  
    pre_seg_att_li = []
    new_store = []
    for data,to_cal in zip(total_gp,embeding_gp):

        c_st = data

        pre_seg_att = get_distribu(model2,to_cal)

        pre_seg_att_li.append(pre_seg_att)
        cofactor = 0.3
        recom_complete = 3
        stret_choose = 'n'
        seqlen = len(data[0])
        f_c = 'n'
        pre_seg = seg_generate(pre_seg_att,cofactor,recom_complete,f_c,seqlen,stret_choose)

        # label = (int(data[1][0])-1,int(data[1][1])-1)
        label = (int(data[1][0]),int(data[1][1]))

        total_num+=1
        for seg in pre_seg[0]:
            if ju_cover(seg,label)==True:
                call_num+=1
                break
        print(ju_cover(seg,label))

        c_st.append(pre_seg[0])
        new_store.append(c_st)
        print('-------------')
    print(new_store)

    print(call_num)
    print(total_num)
    print(call_num/total_num)
    return pre_seg_att_li,new_store





          
    
# from matplotlib import pyplot as plt
def seg_generate(pre_seg_att,cofactor,recom_complete,f_c,seqlen,stret_choose):
    import numpy as np
    pre_seg = []
    for fig in pre_seg_att:
        y= F.softmax(torch.tensor(fig).reshape(-1)).cpu().numpy()
        # print(y)
    
        p_y = np.log(300*y)
        c = F.relu(torch.tensor(p_y))
        import numpy as np
        length = min(seqlen,1022)
        find_way = int(cofactor*length)
        if find_way==0 or find_way==1:
            find_way = 2
        segment = np.sort(np.argsort(-np.array(y))[0:find_way])
        # print(segment)
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
        
    
        #采样延展机制
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

def get_dist_figure(total_gp,pre_seg_att_li,cofactor,recom_complete,stret_choose,f_c):
    total_num = 0
    call_num = 0  
    one_shot_a = 0
    T_aPC = []
    for data,pre_seg_att in zip(total_gp,pre_seg_att_li): 
        seqlen = len(data[0])
        pre_seg = seg_generate(pre_seg_att,cofactor,recom_complete,f_c,seqlen,stret_choose)
        # label = (int(data[1][0])-1,int(data[1][1])-1)
        label = (int(data[1][0]),int(data[1][1]))
        #one-shot_apc
        one_shotapc = (label[1]-label[0]+1)/seqlen
        one_shot_a+=one_shotapc 
        
        total_num+=1
        #计算apc
        apcli = []
        #计算是否命中
        if label[0]!=label[1]:
            for seg in pre_seg[0]:
                if ju_cover(seg,label)==True:
                    call_num+=1
                    break
            
            for seg in pre_seg[0]:
                if ju_cover(seg,label)==True:
                    overlap,nooverlap = overlap_length(seg, label)
                    if (overlap+nooverlap)!=0:
                        apc_s = overlap/(overlap+nooverlap)
                        apcli.append(apc_s )
        else:
            for seg in pre_seg[0]:
                if label[0]<=seg[1] and label[0]>=seg[0] :
                    call_num+=1
                    apcli.append(1)
                    break 
            

        print(apcli)
        if apcli!=[]:
            T_aPC.append(max(apcli))      
        # else:
        #     T_aPC.append(0) 
        # print('-------------')
        
    
    recccll = call_num/total_num
    if T_aPC!=[]:

        Apc = sum(T_aPC)/len(T_aPC)
    else:
        Apc = 0
   
        
    ave_on_shot = one_shot_a/total_num
    print(f'in a parameter of cofactor:{cofactor},recom_complete:{recom_complete},stret_choose:{stret_choose},f_c:{f_c},the recall is {recccll},Apc is {Apc},oneshot apc is {ave_on_shot} ')
    
    return (cofactor,recom_complete,stret_choose,f_c,recccll,Apc,ave_on_shot)


import os
path = './Multi_data'



# total_gp = load_mydict(path+'/insp_train')
# tp1 = load_mydict(path+'/full_forhybrid')
# tp2 = load_mydict(path+'/yeast')
# total_gp += tp1
# total_gp += tp2

# total_gp = load_mydict(path+'/Swiss_2024_EXP_NLS_data')
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000255')
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000256')
total_gp = load_mydict(path+'/C_NLS_2024_uni_0000250')
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000303')
# total_gp = load_mydict(path+'/C_NLS_2024_uni_0000305')

# total_gp = load_mydict(path+'/SWISS_2024_DNA_BIND')
# total_gp = load_mydict(path+'/SWISS_2024_NES_nuclear_eport_signal')
# total_gp = load_mydict(path+'/SWISS_2024_RNA_CAP_BIND')
# total_gp = load_mydict(path+'/SWISS_2024_trna_interaction')
# total_gp = load_mydict(path+'/SWISS_2024_HTH_LA')



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

total_gp = check_and_return(total_gp)
print(len(total_gp))
seq_li = get_seq_li(total_gp)
embeding_gp = get_embedding(seq_li)
pre_seg_att_li,new_store = get_recall_result(total_gp,embeding_gp,parameter)
   
 
savename = ['insp_train_','yeast_','hybrd2_','fulld_']
# sa = 'tOTAL_INSP'
# sa = 'Swiss_2024_EXP_NLS_data'
# sa = 'C_NLS_2024_uni_0000255'
# sa = 'C_NLS_2024_uni_0000256'
# sa = 'C_NLS_2024_uni_0000250'
# sa = 'C_NLS_2024_uni_0000303'
# sa = 'C_NLS_2024_uni_0000305'

sa = 'SWISS_2024_DNA_BIND'
# sa = 'SWISS_2024_NES_nuclear_eport_signal'
# sa = 'SWISS_2024_RNA_CAP_BIND'
# sa = 'SWISS_2024_trna_interaction'
# sa = 'SWISS_2024_HTH_LA'

print(sa)

recom_complete = 3



fi_li = ['n','f']
st_li = ['n','s']
for stret_choose in st_li:
    for f_c in fi_li:
        cofa= []
        recall_ = []
        apcs = []
        for cofactor in range(3,100):
            cofactor = cofactor/100
            cofactor,recom_complete,stret_choose,f_c,recall,Apc,ave_on_shot = get_dist_figure(total_gp,pre_seg_att_li
                                                                                        ,cofactor,recom_complete,stret_choose,f_c)
            cofa.append(cofactor)
            recall_.append(recall)
            apcs.append(Apc)
        print(list(zip(cofa,recall_,apcs)))

        #保存结果
        ex = list(zip(cofa,recall_,apcs))
        import csv
        with open(f'./hitapc_{sa}_{stret_choose}_{f_c}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            file.write(f'avg-oneshot apc,{ave_on_shot}\n')
            file.write('cofactor,recall,aPC\n')
            for row in ex:
                writer.writerow(row)
 

