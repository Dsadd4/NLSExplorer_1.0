from sklearn import svm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from sklearn.model_selection import KFold
import numpy as np
clf = svm.SVC(kernel="linear", gamma=2)
# print(dir(clf))

from sklearn import svm

import torch
# print(torch.__version__)  #注意是双下划线
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from utils import load_mydict


#去除单个位点的预测结果
base_dir = '193_important_segment_without_single'
have_struc = load_mydict('NLS_protein_with_alphafold')
pre_data = load_mydict(base_dir+'/'+'196_protein_import_segment_without_single200')

#这里通过对pre_data的调整可以加入新的NLS训练数据
# print(pre_data[0])
# print('-----------------')
# print(have_struc )


# pre_data = load_mydict(base_dir+'/'+'196_protein_import_segment_without_single200')
# print(pre_data[0])

def generate_dict(pre_data,have_struc):
    proteinws = {}
    proteinwlable = {}
    proteinseq = {}
    l = 0
    num = 0

    for item in pre_data:
        if item [2] in have_struc:
            # print()
            proteinws[item [2]] = item [3]
            proteinwlable[item [2]] = item[1]
            proteinseq[item [2]] = item[0]
            # print(item[0])
            l+=len(item[0])
            num+=1

    # print(proteinws)

    proteinw_fs = {}
    for name in proteinws:
        pre_seg = proteinws[name]
        seq_len = len(proteinseq[name])
        f_seg = []
        for th in pre_seg:
            if th[0]<=seq_len:
                if th[1]>seq_len:
                    f_seg.append((th[0],seq_len))
                
                    
                elif th[0]==th[1]:
                    f_seg.append((th[0],th[0]))
                    
                elif th[0]<th[1]:
                    f_seg.append((th[0],th[1]))
        proteinw_fs[name]=f_seg
    # print('----------------')
    # print(proteinw_fs)
    return proteinws,proteinwlable,proteinseq,proteinw_fs
    
proteinws,proteinwlable,proteinseq,proteinw_fs = generate_dict(pre_data,have_struc)



import random
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
#----------------------开始造数据----------------
#用于训练的数据
#消融实验在这里直接定义为空即可
def generate_single_training(name,label,pre_seg):
    import os
    import progres as pg
    positive_for_train = []
    negative_for_train = []
    # print(label)
    # print(pre_seg)
    for th in pre_seg:
        if ju_cover(th,label)==True:
            positive_for_train.append(th)
        else:
            negative_for_train.append(th)
 
    q = f"{name}.pdb"
    dir = './183_alphafold_pdb/pdb_full_gain2/'+q
    # dir = './loc_in_nucleus_with_pdb/'+q
    #--------------------------------------------------------------------------------------------------------
    #在这里判断是否有对应文件
    
    if os.path.exists(dir )==False:
        return [[],[],[],[],positive_for_train,negative_for_train]
        
    coords = pg.read_coords(dir)
    if coords==[]:
        return [[],[],[],[],positive_for_train,negative_for_train]
    
    # print(coords)
    
    extract_feat,extract_adj_mat,extract_mask = pg.get_ecgnn_feature(coords)

    data_for_train  = [coords,extract_feat,extract_adj_mat,extract_mask,positive_for_train,negative_for_train]
    # 消融实验，添加这行即可
    data_for_train  = [[],[],[],[],positive_for_train,negative_for_train]
    return data_for_train


    

    


# data_store
# tot_emb   
from egnn_pytorch import EGNN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Attention(nn.Module):

    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        #最后投影到概率上面去
        self.ll =    nn.Linear(hidden_dim+1,hidden_dim)
    def forward(self, embding):
        lenttth = embding.size()[1]
        rate_matrix = self.atten_Matrix(embding)
        rate_matrix = self.relu(rate_matrix)
        att_rate = F.softmax(rate_matrix,dim=1)
        sum_ = (embding*att_rate).sum(1)
        sum_ = torch.concat((sum_,torch.tensor([lenttth]).cuda().unsqueeze(0)),1)
        sum_ = self.ll(sum_)
#         print(sum.size())
      
        return sum_,att_rate
    
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

      
        return sum_,att_rate
import math
# n = int(input('数字:'))
# x = math.sqrt(n)
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
        lenttth = embding.size()[1]
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
   
import torch.nn as nn
import torch.nn.functional as F
class oDCNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(oDCNN, self).__init__()

        # 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

        # Activation function
        self.act = nn.GELU()
        
               

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.act(self.conv1(x))
        x = F.max_pool1d(x,x.size(2))
#         x = x.view
        x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = self.fc(x)
        return x

import torch.nn as nn
import torch.nn.functional as F
class oDCNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(oDCNN, self).__init__()

        # 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=in_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=2)
        # Activation function
        self.act = nn.GELU()
        
               

    def forward(self, x):
        x = x.permute(0,2,1)
        res_x = x
        x = self.conv1(x)
       
        x = self.conv2(x)
        x = self.act(x+res_x)
        
        # print(x.size())
        x = F.gelu(self.conv3(x))
        
        x = F.max_pool1d(x,x.size(2))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        return x



import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(CNN, self).__init__()

        # 1D convolution layer
        self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=in_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=2)
        # Activation function
        self.act = nn.GELU()
        
               

    def forward(self, x):
        x = x.permute(0,2,1)
        res_x = x
        x = self.conv1(x)
       
        x = self.conv2(x)
        x = self.act(x+res_x)
        
        # print(x.size())
        x = F.gelu(self.conv3(x))
        
        x = x.permute(0,2,1)  # Flatten the tensor
        return x



class gnn_classifier(nn.Module):

    def __init__(self):
        super(gnn_classifier, self).__init__()
        hidden_dim=1408
        hidden_edge_dim = 128
        self.edge_emb = nn.Embedding(2, hidden_edge_dim )
        n_layers = 1
        
        edge_dim= int(hidden_dim/15)
        
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(EGNN(dim = edge_dim, edge_dim =  hidden_edge_dim))


       
        lstm_dim = int(hidden_dim/4)
        cls_dim = lstm_dim *2
        conv_dim = 256
        # self.cls = nn.Linear(cls_dim +conv_dim, 1)
        self.cls = nn.Linear(cls_dim , 1)
        self.lstm = nn.LSTM(int(hidden_dim), lstm_dim,1,bidirectional=True,batch_first = True)
        self.sigmoi = nn.Sigmoid()
        self.att_fu = Attention(cls_dim )
        
        
        self.conv = oDCNN(hidden_dim,conv_dim)
        
        
        self.re_dim = nn.Linear(int(hidden_dim), edge_dim)
        # self.inc_dim = nn.Linear(edge_dim, int(hidden_dim))
        self.inc_dim = nn.Linear(edge_dim, int(hidden_dim))
        
        
        self.att_fu2 = Attention(int(hidden_dim/5)*2)
        # self.cls2 = nn.Linear(int(hidden_dim/5)*2+conv_dim, 1)
        self.cls2 = nn.Linear(int(hidden_dim/5)*2, 1)
        
        self.lstm2 = nn.LSTM(int(hidden_dim), int(hidden_dim/5),1,bidirectional=True,batch_first = True)
        self.sigmoi2 = nn.Sigmoid()
        
        conv_dim = 256
        #仅仅为了改变适应
        self.conv2 = CNN(hidden_dim,hidden_dim)
        
        # mid_dim = int(hidden_dim/5)
        # self.re_dim2 = nn.Linear(int(hidden_dim), mid_dim )
        # self.inc_dim2 = nn.Linear(mid_dim , int(hidden_dim))
        
    def forward(self, embding,coord,adj_matrix,positive_for_train,negative_for_train):
        #变化邻接矩阵，使其变成边特征嵌入,positive_for_train,negative_for_train

        embding = embding.cuda()
        choose = 'include-struc'
        # choose = 'no'
        if coord!= []:
            adj_matrix = adj_matrix.long().unsqueeze(3).cuda()
            edge_emb = self.edge_emb(adj_matrix)
            edge_emb = edge_emb.squeeze(3)
        #在这里引入结构信息模块
        if coord!= [] and choose == 'include-struc' and edge_emb.size()[1]==embding.size()[1]:
            # adj_matrix = adj_matrix.long().unsqueeze(3).cuda()
            # # print(adj_matrix.size())
            # # print("-------")
            # edge_emb = self.edge_emb(adj_matrix)
            # edge_emb = edge_emb.squeeze(3)
            
            coord = torch.tensor(coord).unsqueeze(0).cuda()
            # print(edge_emb.size())
            # print(coord.size())
            # print(embding.size())
            if edge_emb.size()[1]!=embding.size()[1]:
                edge_emb = edge_emb[:,:embding.size()[1],:embding.size()[1],:]
                coord = coord[:,:embding.size()[1],:]
            
            store = embding
            
            embding =self.re_dim(embding)
            for layer in self.layers:
                embding, coord = layer(embding, coord, edge_emb)
            embding =self.inc_dim(embding)
            embding =F.relu(embding)
            
            # embding = store
            # print(edge_emb.size())
            # print(embding.size())
            # embding = torch.concat((store,embding),2)
            embding = F.gelu(embding +store)
            
        
        
            max_len =embding.size()[1]
            pre = 'begjin'
            if len(positive_for_train)!=0:
                for item in positive_for_train:
                    # print('--------')
                    # print(item)
                    # print(max_len)
                    if item[0]>max_len:
                        continue
                    if item[1]+1<=max_len:
                        posi_emb = embding[:,item[0]:item[1]+1,:]
                    else:
                        posi_emb = embding[:,item[0]:max_len,:]
                        
                    # print(posi_emb.size())
                    # print('-------------') 
                    #一维卷积网络引入
                    # tc = self.conv(posi_emb)
                    
                    te,_= self.lstm(posi_emb)
                    sum_,_ = self.att_fu(te)
                    
                    # sum_ = torch.concat((sum_,tc),1)
                    
                    pre_di = self.sigmoi(self.cls(F.relu(sum_)))
                    
                    if pre == 'begjin':
                        pre = pre_di 
                    else:
                        pre =torch.concat((pre,pre_di),1)
                        
            if len(negative_for_train)!=0:
                for item in negative_for_train:
                    # if item[0]>max_len:
                    #     continue
                    nega_emb = embding[:,item[0]:item[1]+1,:]
                    
                    # print(item)
                    
                    # tc = self.conv(nega_emb)
                    
                    te,_= self.lstm(nega_emb)
                    sum_,_ = self.att_fu(te)
                    
                    # sum_ = torch.concat((sum_,tc),1)
                    # pre_di = self.sigmoi(self.cls(sum_))
                    pre_di = self.sigmoi(self.cls(F.relu(sum_)))
                    
                    if pre == 'begjin':
                        pre = pre_di 
                    else:
                        pre =torch.concat((pre,pre_di),1)

            return pre  
        
        else:
            # store = embding
            # # embding =self.re_dim2(embding)
            # # embding =self.inc_dim2(embding)
            # # embding = F.relu(embding +store)
            
            max_len =embding.size()[1]
            pre = 'begjin'
            if len(positive_for_train)!=0:
                for item in positive_for_train:
                    # print('--------')
                    # print(item)
                    # print(max_len)
                    # if item[0]>max_len:
                    #     continue
                    if item[1]+1<=max_len:
                        posi_emb = embding[:,item[0]:item[1]+1,:]
                    else:
                        posi_emb = embding[:,item[0]:max_len,:]
                        
                    # tc = self.conv2(posi_emb)
                    
                    # posi_emb = self.conv2(posi_emb)
                    
                    # te,_= self.lstm2(posi_emb)
                    # sum_,_ = self.att_fu2(te)
                    
                    # # sum_ = torch.concat((sum_,tc),1)
                    
                    # pre_di = self.sigmoi2(self.cls2(F.relu(sum_)))
                    
                    te,_= self.lstm(posi_emb)
                    sum_,_ = self.att_fu(te)
                    
                    # sum_ = torch.concat((sum_,tc),1)
                    
                    pre_di = self.sigmoi(self.cls(F.relu(sum_)))
                    
                    
                    if pre == 'begjin':
                        pre = pre_di 
                    else:
                        pre =torch.concat((pre,pre_di),1)
                        
            if len(negative_for_train)!=0:
                for item in negative_for_train:
                    # if item[0]>max_len:
                    #     continue
                    # print(item)
                    # print(max_len)
                    if item[1]+1<=max_len:
                        nega_emb = embding[:,item[0]:item[1]+1,:]
                    else:
                        nega_emb = embding[:,item[0]:max_len,:]
                    
                    # tc = self.conv2(nega_emb)
                    # nega_emb = self.conv2(nega_emb)
                    
                    # te,_= self.lstm2(nega_emb)
                    # sum_,_ = self.att_fu2(te)
                    # # sum_ = torch.concat((sum_,tc),1)
                    # # pre_di = self.sigmoi(self.cls(sum_))
                    # pre_di = self.sigmoi2(self.cls2(F.relu(sum_)))
                    
                
                    
                    
                    te,_= self.lstm(nega_emb)
                    sum_,_ = self.att_fu(te)
                    # sum_ = torch.concat((sum_,tc),1)
                    # pre_di = self.sigmoi(self.cls(sum_))
                    pre_di = self.sigmoi(self.cls(F.relu(sum_)))
                    
                    if pre == 'begjin':
                        pre = pre_di 
                    else:
                        pre =torch.concat((pre,pre_di),1)

            return pre      



 
        




def show_acc(X_te,embding_te,model):
    #默认一次数据只有一条nls
    import torch
    with torch.no_grad():
        num_top3 = 0
        num_top2 = 0
        num_top1 = 0
        
        total_num = 0
        for j,ba_data in enumerate(X_te):
            
            negg = []
            for ne in ba_data[-1]:
                if ne[0]!=ne[1]:
                    negg.append(ne)
            
            poo = []
            # po_r = 0
            for po in ba_data[-2]:
                if po[0]!=po[1]:
                    poo.append(po)
                    
            # if len(po_for_train)==0:
            #     continue
            
            # #预测结果   
            # print('*'*60)
            max_len = embding_te[j].size(1)
            poo = corr_r(poo ,max_len)
            negg =  corr_r(negg ,max_len) 
            # print(max_len)
            # print(poo)
            # print(negg)   
            # print(ba_data)
            if poo==[]:
                continue     
            pre = model(embding_te[j],ba_data[0],ba_data[2],poo,negg )
            # print(pre)
            
            po_la =[]
            if len(poo)!=0:
                po_la = [1.0]*len(poo)     
            ne_la = []
            if len(ba_data[-1])!=0:
                ne_la = [0.0]*len(ba_data[-1]) 
            fin_la = []
            for item in po_la:
                fin_la.append(item)
            for item in ne_la :
                fin_la.append(item)
                
         
            
            
            #这是找出正确的序列编号，命名为right_li
            right_li = []
            for iddx,thi in enumerate(fin_la):
                if thi==1:
                    right_li.append(iddx)
            
            # print(pre)
            recomd = torch.argsort(-1*pre,1)
            top = 3
            print(right_li)
            print(recomd[0])
            print('-------------g----------------')
            
            
            
            for p_i in recomd[0][0:top]:
                if int(p_i) in right_li:
                    # print('nice!')
                    num_top3+=1
                    break
            
            for p_i in recomd[0][0:top-1]:
                if int(p_i) in right_li:
                    # print('nice!')
                    num_top2+=1
                    break
            
            for p_i in recomd[0][0:top-2]:
                if int(p_i) in right_li:
                    # print('nice!')
                    num_top1+=1
                    break
            # total_num+=1
            if len(ba_data[-2])!=0:
                total_num+=1
                
                

        print('p'*50)    
        print(f"the total_num is {total_num}")
 
        return num_top3/total_num,num_top2/total_num,num_top1/total_num


acc_li = []
f = open('./record6.txt','w+')
# r_model = model

def generate_r(seg,l,max_len):
    import random
    beg =random.choice(list(range(l)))
    b_ar = random.choice([1,-1])
    
    end = random.choice(list(range(l)))
    e_ar = random.choice([1,-1])
    
    e_o = int(seg[1])+end*e_ar
    b_o = int(seg[0])+beg*b_ar
    
    while b_o>=e_o:
        beg =random.choice(list(range(l)))
        b_ar = random.choice([1,-1])
        b_o = int(seg[0])+beg*b_ar
            
    if b_o<0:
        b_o = 0
    if e_o>max_len:
        e_o = max_len
        
    if b_o<e_o:
        
    
        return (b_o,e_o)
    else:
        return seg

def generate_n(seg,dis,max_len):
    import random
    neg = random.choice(range(max_len))
    lenth = random.choice(range(5,30))
    fot = int(lenth/2)
    end = lenth - fot
    
    beg = neg-fot
    end = neg+end
    while  True:
        neg = random.choice(range(max_len))
#         print(neg)
        lenth = random.choice(range(5,30))
        fot = int(lenth/2)
        end = lenth - fot

        beg = neg-fot
        end = neg+end
        #拉大距离间隔
        if ju_cover((beg,end),seg)== False and beg>=0:
            if end<seg[0]:

                if seg[0]-end>dis:
                    break
            else:
                if beg-seg[1]>dis:
                    break
    
    
    if end>max_len:
        
    
        return (beg,max_len)
    else:
        return (beg,end)
  

        
def corr_r(seg_li,max_len):
    r_li = []
    for item in seg_li:
        if int(item[0])>=max_len-2:
            continue
        
        r_li.append(item)
    return r_li



#最新推荐出来的数据


def get_data(insp_train):
    seq_store = []
    data_store = []
    label_store = []
    # print(insp_train)
    faied_li = []
    print(insp_train)
    f = open('./failed_gai_pdb.fasta','w')
    for st_d in insp_train:
        ins_seq = st_d[0]
        ins_label = st_d[1]
        ins_ac = st_d[2]
        ins_pre = st_d[3]

        # try:
        #     data_for_train = generate_single_training(ins_ac,ins_label ,ins_pre)
        # except:
        #     faied_li.append(ins_ac)
        #     continue
        data_for_train = generate_single_training(ins_ac,ins_label ,ins_pre)
        #加下面这行就可以排除没pdb的进行实验了
        # if data_for_train[0]==[]:
        #     continue
        # 如果没有pdb文件则产生失败，返回[]coordinate
        # print("ok")
        seq_store.append(ins_seq)
        data_store.append(data_for_train)
        label_store.append(ins_label)
        
        if data_for_train[0]==[]:
            faied_li.append(ins_ac)
            f.write('>'+ins_ac+'\n')
            f.write(ins_seq)
            f.write('\n')
            f1 = open(f"./结构待alphafold计算/{ins_ac}.fasta",'w')
            f1.write('>'+ins_ac+'\n')
            # print(ins_seq)
            # print('---------------------------------------')
            f1.write(ins_seq)
            f1.close()
            
            
        
    f.close()       
    print(faied_li) 
    
    print('***************uuuuuuuuuuuuuuuu*****************')
    from utils import generate_representation
    emb__seq = generate_representation([1]*len(seq_store),seq_store)
    print(emb__seq[0].size())
    #data_for_train  = [coords,extract_feat,extract_adj_mat,extract_mask,positive_for_train,negative_for_train]



    # tot_emb = []
    # for i,ba_data in enumerate(data_store):
    #     extract_feat = ba_data[1]
    #     for_cocat = emb__seq[i]
    #     cocat = for_cocat.unsqueeze(0)
    #     # print('-----------------')
    #     # print(cocat.size())
    #     # print(extract_feat.size())
    #     # print(len(seq_store[i]))
    #     if extract_feat!=[]:
    #         if cocat.size()[1]==extract_feat.size()[1]:
    #             f_emb = torch.concat((cocat,extract_feat),2)
    #         elif cocat.size()[1]>extract_feat.size()[1]:
    #             f_emb = torch.concat((cocat[:,:extract_feat.size()[1],:],extract_feat),2)
    #         else:
    #             f_emb = torch.concat((cocat,extract_feat[:,:cocat.size()[1],:]),2)
            
    #     # print(f_emb.size())
    #     tot_emb.append(f_emb)
    
    tot_emb = []
    for i,ba_data in enumerate(data_store):
        extract_feat = ba_data[1]
        for_cocat = emb__seq[i]
        cocat = for_cocat.unsqueeze(0)
        # print('-----------------')
        # print(cocat.size())
        # print(extract_feat.size())
        # print(len(seq_store[i]))
        f_emb = cocat
        l_l = int(1408 - 1280)
        if extract_feat!=[] and cocat.size()[1]==extract_feat.size()[1]:
           
            f_emb = torch.concat((cocat,extract_feat),2)
        else:
            zero_tensor = torch.zeros_like(f_emb)
            f_emb = torch.concat((cocat,zero_tensor[:,:,:128]),2)
        
        
        # if extract_feat!=[]:
        #     if cocat.size()[1]==extract_feat.size()[1]:
        #         f_emb = torch.concat((cocat,extract_feat),2)
        #     elif cocat.size()[1]>extract_feat.size()[1]:
        #         f_emb = torch.concat((cocat[:,:extract_feat.size()[1],:],extract_feat),2)
        #     else:
        #         f_emb = torch.concat((cocat,extract_feat[:,:cocat.size()[1],:]),2)
        # else:
        #     zero_tensor = torch.zeros_like(f_emb)
            
        # print(f_emb.size())
        tot_emb.append(f_emb)
        
    return data_store,label_store,tot_emb

def get_result(data_store,tot_emb,data_store2,tot_emb2):
    for k in range(1):
        

        #初始化模型
        model = gnn_classifier().cuda()   
        import torch.optim as optim
        optimizer = optim.Adam(model.parameters(), lr=0.000001)
        criterion = nn.BCELoss()
        acc = []
        
        #开始训练
        for epoch in range(25):
            # r_model = model
            for i,ba_data in enumerate(data_store):
                # print(ba_data[-2])
                #生成nls片段和negative片段的数据
                po_for_train = []
                for re_ in range(1):
                    for ttt in ba_data[-2]:
                        if ttt[0]!=ttt[1]:
                            po_for_train.append(ttt)
                if len(po_for_train)==0:
                    continue
                
                
                
                #在这里注意negative段应该怎么选择
                n_choose = 1
                nefor_t = []
                # for item in ba_data[-1][0:n_choose]:
                #     if item[0]!=item[1]:
                #         nefor_t.append(item)  
                for item in ba_data[-1]:
                    if item[0]!=item[1]:
                        nefor_t.append(item)  
                
                #补齐长度
                truel = len(nefor_t)
                
                #生成更多的训练数据
                max_len = tot_emb[i].size()[1]
                # print(embding_t[i].size())
                label = label_store[i]
                
                # choose_rc = 15
                choose_rc = 3
                append_len = truel-len(po_for_train)
                for _ in range(append_len):
                    t_p = generate_r(label ,choose_rc,max_len)
                    po_for_train.append(t_p)
                    
                # 再来个负片段生成会更好
                append_len = truel-len(nefor_t)
                for _ in range(append_len):
                    # n_p = generate_n(label ,10,max_len)
                    n_p = generate_n(label ,20,max_len)
                    # if n_p[1]-np[0]>1:
                    nefor_t.append(n_p)
                        
                    
                # generate_n(seg,dis,max_len):
                
                
                
                max_len =tot_emb[i].size()[1]
                # print('#############')
                # print(max_len)
                po_for_train = corr_r(po_for_train,max_len)
                nefor_t = corr_r(nefor_t,max_len)
                
                pre = model(tot_emb[i],ba_data[0],ba_data[2],po_for_train,nefor_t)
                # pre = model(embding_t[i],ba_data[0],ba_data[2],po_for_train,nefor_t)
                
                # print(pre.size() )
                
                po_la =[]
                if len(po_for_train)!=0:
                    po_la = [1.0]*len(po_for_train)
                    
                ne_la = [0.0]*len(nefor_t)
                # if len(ba_data[-1])!=0:
                #     ne_la = [0.0]*len(nefor_t)
                    
                # print(len(po_la))   
                # print(len(ne_la))       
                
                fin_la = []
                for item in po_la:
                    fin_la.append(item)
                for item in ne_la :
                    fin_la.append(item)
                    
                # print('-------------------------------')  
                if fin_la==[]:
                    continue  
                # print(fin_la)
                # print(pre)
                loss = criterion(pre, torch.tensor(fin_la).unsqueeze(0).cuda())
                # 后向传播
                # print(loss)
                
                # print(f'-------------{epoch}---------------')
                loss.backward()
                optimizer.step()
            # torch.save(model.state_dict(), f'./resconn_{epoch}_nls_model_not_include')
            
            t3,t2,t1 = show_acc(data_store2,tot_emb2,model)
            # print([t3,t2,t1])
            
            acc.append([t3,t2,t1])
            print(f'the ep is {epoch}')    
            print([t3,t2,t1])  
            
            acc_li.append(acc)    
        
    t3_li,t2_li,t1_li = zip(*acc)
    print(acc_li)
    print('----')      
    print(max(t3_li))
    print(max(t2_li))
    print(max(t1_li))
    print('ppppppp')
    return max(t3_li),max(t2_li),max(t1_li)

#训练数据
insp_train = load_mydict('./for_recom/insp_train_0.6')
data_store,label_store,tot_emb = get_data(insp_train)

import pandas as pd
import sys
parameter = sys.argv
#最新推荐出来的数据
fac = parameter[1]

# insp_train = load_mydict(f'./for_recom/hybrd2_{fac}')
insp_train = load_mydict(f'./for_recom/yeast_{fac}')
data_store2,label_store2,tot_emb2 = get_data(insp_train)
print(len(data_store2))


# f = open('./record7.txt','w')

t3,t2,t1 = get_result(data_store,tot_emb,data_store2,tot_emb2)
# print()
# f.write(str(t3)+' '+str(t2)+' '+str(t1))
    
# f.close()

#保存结果

import csv
with open(f'./消融实验/yeast_{fac}--.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    file.write('top3,top2,top1\n')
    file.write(f'{t3},{t2},{t1}')