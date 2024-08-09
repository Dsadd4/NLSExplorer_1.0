import os
import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from mamba import Mamba, MambaConfig
from utils import load_mydict, generate_representation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Attention(nn.Module):
    def __init__(self,hidden_dim):
        super(Attention, self).__init__()
        # The linear layer that maps from hidden state space to tag space
        self.atten_Matrix = nn.Linear(hidden_dim,1)
        self.relu = nn.ReLU()
        #最后投影到概率上面去
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

class gnn_classifier(nn.Module):
    def __init__(self):
        super(gnn_classifier, self).__init__()
        hidden_dim = 1280
        lstm_dim = int(hidden_dim / 4)
        cls_dim = lstm_dim * 2
        conv_dim = 256
        self.cls = nn.Linear(cls_dim, 1)
        self.lstm = nn.LSTM(int(hidden_dim), lstm_dim, 1, bidirectional=True, batch_first=True)
        self.sigmoi = nn.Sigmoid()
        self.att_fu = Attention(cls_dim)
        config = MambaConfig(d_model=int(hidden_dim / 2), n_layers=12)
        self.mamb = Mamba(config)


        config_1 = MambaConfig(d_model=hidden_dim, n_layers=3)
        self.mamb1 = Mamba(config_1)
    def forward(self, embedding, segments):
        batch_size, max_len, _ = embedding.size()
        embedding = embedding.cuda()
        results = []
        emb_li = []
        embedding = self.mamb1(embedding)+embedding
        embedding = self.mamb1(embedding)
        for batch_idx in range(batch_size):
            emb_batch = embedding[batch_idx].unsqueeze(0)
            segment_batch = segments[batch_idx]
            batch_results = []

            for segment in segment_batch:
                start, end = int(segment[0].item()), int(segment[1].item())
                if start == 0 and end == 0:
                    continue

                if end + 1 <= max_len:
                    posi_emb = emb_batch[:, start:end + 1, :]
                else:
                    posi_emb = emb_batch[:, start:max_len, :]

                lstm_out, _ = self.lstm(posi_emb)
                lstm_out = self.mamb(lstm_out) + lstm_out
                att_out, _ = self.att_fu(lstm_out)
                emb_li.append(att_out)
                cls_out = self.sigmoi(self.cls(F.relu(att_out)))

                batch_results.append(cls_out)

            if len(batch_results) > 0:
                max_result_len = max([res.size(1) for res in batch_results])
                padded_results = [F.pad(res, (0, max_result_len - res.size(1)), "constant", 0) for res in batch_results]
                batch_results = torch.cat(padded_results, dim=1)
                results.append(batch_results)

        if len(results) > 0:
            max_len_result = max([res.size(1) for res in results])
            results = [F.pad(res, (0, max_len_result - res.size(1)), "constant", 0) for res in results]
            results = torch.cat(results, dim=0)
        else:
            results = torch.tensor([]).cuda()
        
        return results, emb_li

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


def select_cor(label,pre_g):
    r_segm = []
    for segss in pre_g:
        if ju_cover(segss,label):
            r_segm.append((segss[0]+1,segss[1]+1))
    return r_segm

def del_recc(tp):
    store_dict ={}
    ret = []
    filt = 0
    for item in tp:
        drt = item
        Ac = drt[2]
        label = drt[1]
        pre_seg = drt[3]
        seq = drt[0]
        if Ac not in store_dict:
            store_dict[Ac] = [label,seq]
            ret.append(item)
        else:
           
            c1 = store_dict[Ac]
            if c1[1]==seq and c1[0]==label:
                print('filter')
                filt+=1
            else:
                ret.append(item)
    print(f'we filter in total {filt}')
    return ret


def check_and_return(EXNES):
    Ok_data=[]
    for item in EXNES:
        if len(item[0])>1022:
            seq = item[0][:1022]
            print(len(seq))
        else:
            seq = item[0]
       
        la = item[1]
        # print(la)
        # print(la[])

        if la[0]>=len(seq):
            continue
        else:
            Ok_data.append(item)
    return Ok_data




def generate_negative_sample(ins_lab, seq_len):
    """
    Generate a negative sample which does not overlap with the given region in ins_lab.
    The length of the negative sample should be between 6 and 50.
    
    Parameters:
    ins_lab (tuple): A tuple (start, end) indicating the region.
    seq_len (int): Length of the sequence.
    
    Returns:
    tuple: A new region (start, end) which is a negative sample.
    """
    start, end = ins_lab
    min_len = 6
    max_len = 50
    
    possible_starts_before = np.arange(0, max(0, start - min_len + 1))
    possible_starts_after = np.arange(min(seq_len, end + 1), seq_len - min_len + 1)
    
    possible_starts = np.concatenate([possible_starts_before, possible_starts_after])
    
    if possible_starts.size == 0:
        return None
    
    new_start = np.random.choice(possible_starts)
    new_len = np.random.randint(min_len, min(max_len, seq_len - new_start) + 1)
    new_end = new_start + new_len
    
    return (new_start, new_end)




negativedict = {}
positivedict = {}
class CustomDataset(Dataset):
    def __init__(self, data_store, tot_emb,labeldict):
        self.data_store = data_store
        self.tot_emb = tot_emb
        self.label_d = labeldict

    def __len__(self):
        return len(self.data_store)

    def __getitem__(self, idx):
        ba_data = self.data_store[idx]
        ins_ac = ba_data[0]
        ins_lab = self.label_d[ins_ac]
        # print(ins_lab)

        pre_seg = ba_data[2]
        seq = ba_data[3]
        seq_len = len(seq)
        # train_seg = ins_lab + pre_seg
        train_seg = pre_seg
        fin_la = []

        balance_train = []
        negtive = []
        positve_train = []

        for item in train_seg:
            real_loca = (item[0]+1,item[1]+1)
            fLAG = 0
            for labb in ins_lab:
                if ju_cover((item[0]+1,item[1]+1), labb):
                    fLAG = 1
            if fLAG == 1:
                
                fin_la.append(1.0)
                positve_train.append(item)
            else:
                negtive.append(item)
        balance_train.extend(positve_train)

        if ins_ac not in negativedict:
            negativedict[ins_ac] = negtive
        else:
            negativedict[ins_ac].extend(negtive)
            negativedict[ins_ac] = list(set(negativedict[ins_ac]))
        
        if ins_ac not in positivedict:
            positivedict[ins_ac] = positve_train
        else:
            positivedict[ins_ac].extend(positve_train)
            positivedict[ins_ac] = list(set(positivedict[ins_ac]))
       

        
        for i in range(len(positve_train)):
            if len(negtive)!=0:
                # print(positve_train)
                balance_train.append(random.choice(negtive))
                fin_la.append(0.0)
        


        if len(balance_train)==0:
            while(True):
                genesampl = generate_negative_sample(ins_lab[0], seq_len)
                Flags = 0
                print(f'generate negative:{genesampl}')
                print(f'This time labels are {ins_lab}')
                for labb in ins_lab:
                    if ju_cover((genesampl[0]+1,genesampl[1]+1), labb):
                        FLAG = 1
                if Flags == 0:
                    break    
            balance_train.append(genesampl)
            fin_la.append(0)
        return self.tot_emb[idx], torch.tensor(balance_train, dtype=torch.float32), torch.tensor(fin_la, dtype=torch.float32)


def custom_collate_fn(batch):
    # Padding tot_emb
    max_len_emb = max([item[0].size(0) for item in batch])
    batch_emb = torch.stack([torch.cat([item[0], torch.zeros(max_len_emb - item[0].size(0), item[0].size(1))]) for item in batch])
    
    batch_seg = [item[1] for item in batch]
    batch_labels = [item[2] for item in batch]
    
    batch_seg = pad_sequence(batch_seg, batch_first=True, padding_value=0)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0)
    
    return batch_emb, batch_seg, batch_labels


def count_hit(pre,lable,batch_seg_n):
    th = 0.9
    t_cout = 0
    b_g = 0
    ct = 0
    index = 0
    for _,prob in enumerate(pre):
        if int(lable[index])>=0.99 and prob>=th:
            ct=1
        index+=1
        b_g+=1
        if b_g==batch_seg_n:
            if ct == 1:
                t_cout+=1
            ct=0
            b_g=0

    return t_cout




print('d------------------d')



def Indicate_label_ge(ins_label, pre_seg):
    indi_li = []
    for sgggs in pre_seg:
        Flag = 0
        for label in ins_label:
            if ju_cover((sgggs[0]+1,sgggs[1]+1),label):
                Flag = 1        
        if Flag == 1:
            indi_li.append(1)     
        else:
            indi_li.append(0)
    return indi_li

import numpy as np
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_model(model, data_store, tot_emb, labeldict, batch_size=10, lr=0.000001, num_epochs=15, patience=10):
    dataset = CustomDataset(data_store, tot_emb, labeldict)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.BCELoss()
    early_stopping = EarlyStopping(patience=patience, path='checkpoint.pt')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        tohit = 0
        tonb = len(data_store)
        for batch_emb, batch_seg, batch_labels in dataloader:
            optimizer.zero_grad()
            batch_emb = batch_emb.cuda()
            batch_seg = batch_seg.cuda()
            batch_labels = batch_labels.cuda()
            outputs, _ = model(batch_emb, batch_seg)
            outputs = outputs.view(-1)  
            batch_labels = batch_labels.view(-1)  
            batch_seg_n = batch_seg.size()[1]
            ct = count_hit(outputs, batch_labels, batch_seg_n)
            tohit += ct
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        scheduler.step(avg_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')
        print(f'in total NLS {len(data_store)}')
        print(f'Hit {tohit} in total')
        print('----------------------------------')

        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model.state_dict(), f'./ASAZDM')

    return model

   
# from matplotlib import pyplot as plt
def merge_intervals(intervals):
    if not intervals:
        return []

    # 按开始时间排序
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        # 检查是否有重合或距离小于1
        if current[0] <= previous[1] + 1:
            # 融合片段
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)

    return merged

def get_data(insp_train,seq2emb):
    seq_store = []
    data_store = []
    label_store = []

    for st_d in insp_train:
        ins_seq = st_d[0]
        ins_label = st_d[1]
        ins_ac = st_d[2]
        ins_pre = st_d[3]
        ins_pre = merge_intervals(ins_pre)
        data_store.append([ins_ac,[ins_label],ins_pre,ins_seq])
        seq_store.append(ins_seq)
    print('***************uuuuuuuuuuuuuuuu*****************')
    emb_seq = [seq2emb[seq] for seq in seq_store]
    # from utils import generate_representation
    # emb_seq = generate_representation([1]*len(seq_store),seq_store)
    return data_store,emb_seq


# 获取测试数据
def get_test_data():
    # 请根据你的实际情况加载测试数据
    seq2embeding = {}
    total = []
    print('d------------------d')
    from utils import load_mydict
    insp_train = load_mydict('./A2KA_genetest/INSP_training')
    print(insp_train[1])
    total+= insp_train
    total = check_and_return(total)
    total = del_recc(total)
    test_data = total
    seq_store = [item[0] for item in test_data]
    embeddings = generate_representation([1]*len(seq_store), seq_store)
    for seq,emb in zip(seq_store,embeddings):
        seq2embeding[seq] = emb
    return test_data, embeddings , seq2embeding


def get_filter_total():
    total = []
    print('-------------we must check first-------------')
    insp_train = load_mydict('./A2KA_genetest/INSP_training')
    print(insp_train[1])
    total+= insp_train
    total = check_and_return(total)
    total = del_recc(total)
    return total



def get_label(insp_train, seq2emb):
    Labeledict = {}

    for st_d in insp_train:
        ins_seq = st_d[0]
        ins_label = st_d[1]
        ins_ac = st_d[2]
        ins_pre = st_d[3]
        if ins_ac not in Labeledict:
            Labeledict[ins_ac] = [ins_label]
        else:
            Labeledict[ins_ac].append(ins_label)
            Labeledict[ins_ac] = list(set(Labeledict[ins_ac]))
    return Labeledict





def test_model(model, test_data, embeddings):
    results = []
    labels_dict = {}
    for idx, data in enumerate(test_data):
        ins_ac = data[2]
        ins_label = data[1]
        if ins_ac in labels_dict:
            labels_dict[ins_ac].extend([ins_label])
        else:
            labels_dict[ins_ac] = [ins_label]

    with torch.no_grad():
        print(f' therer exists {len(test_data)} for test')
        for idx, data in enumerate(test_data):
            # print(f' The {idx} ')
            ins_ac = data[2]
            seq = data[0]
            for_storage_label =data[1]
            ins_label = labels_dict[ins_ac]
            pre_seg = data[3]
            pre_seg = merge_intervals(pre_seg)
            embedding = embeddings[idx].unsqueeze(0).cuda()
            prob, emb = model(embedding, [pre_seg])
            prob = prob.cpu().numpy()
            indicatt = Indicate_label_ge(ins_label, pre_seg)
            result = {
                'accession': ins_ac,
                'sequence':seq,
                'label': for_storage_label,
                'predicted_segment': pre_seg,
                'probability': prob,
                'indicator':indicatt
            }

            results.append(result)
    return results,labels_dict



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


def calculate_overlap_ratio(predicted_intervals, actual_intervals):
    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            previous = merged[-1]
            if current[0] <= previous[1]:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        return merged
    

    def calculate_overlap_length(predicted, actual):
            overlap_length = 0
            for p_start, p_end in predicted:
                for a_start, a_end in actual:
                    start = max(p_start, a_start)
                    end = min(p_end, a_end)
                    if start <= end:
                        overlap_length += end - start + 1
            return overlap_length

    all_intervals = predicted_intervals + actual_intervals
    merged_intervals = merge_intervals(all_intervals)

 
    union_length = sum(end - start+1 for start, end in merged_intervals)

    overlap_length = calculate_overlap_length(predicted_intervals, actual_intervals)

    ratio = overlap_length / union_length if union_length > 0 else 0

    return union_length, overlap_length, ratio
    
def cal_apc(pre_result,ac_label):
    atp = 0
    afs = 0
    for ac in pre_result:
        pre_s = pre_result[ac]
        NLS_LOC = ac_label[ac]
        # print(NLS_LOC)
        NLS_Whe_loc = [0 for it in NLS_LOC]
        for sgs in pre_s:
            flag = 0
            for idx,item in enumerate(NLS_LOC):
                # print(item)
                if ju_cover(sgs,item):
                    overlap,nooverlap = overlap_length(sgs,item)
                    flag = 1
                    NLS_Whe_loc[idx] = 1
                    break
            if flag == 0:
                overlap= 0
                nooverlap = sgs[1] - sgs [0]+1
       
            atp+= overlap
            afs+= nooverlap
        for indicc,Labl in zip(NLS_Whe_loc,NLS_LOC):
            if indicc == 0:
                afs+= Labl[1]-Labl[0]+1
    if (atp+afs) == 0:
        return 0
    return atp/(atp+afs)


def CCCheck_model(model, test_data, embeddings):
    print('Now we test model')
    results,ac_labels = test_model(model, test_data, embeddings)
    nb = 0
    fb = 0
    total_pre = 0
    print('The final result is processed by add 1 to fit the calculate')

    enforced_learns = []
    th = 0.9

    th_list = [i/100 for i in range(10,100)]
    nb_li = [0 for i in th_list]
    fb_li = [0 for i in th_list]
    pb_li = [0 for i in th_list]

    pre_result = [{} for i in th_list]
    
    for result in results:
        AC = result['accession']
        indicators = result['indicator']
        pre_segm = result['predicted_segment']
        # print(pre_segm)
        assert(len(result['label'])==2)
        # print(result)
        # print(pre_segm)
        # print('----see-structure------')
        # assert(0)

        probability = result['probability'][0]
        total_pre+= len(indicators)
        p = []
        f_p = []
        
        for ind,prob in zip(indicators,probability):
            if ind==1:
                p.append(prob)

            else:
                
                f_p.append(prob)
        
        idx = 0
        for innndx,thf in enumerate(th_list):
            for prob,segm in zip(probability,pre_segm):
                # print(segm)
                if prob>= thf:
                    if AC in pre_result[innndx]:
                        pre_result[innndx][AC].extend([segm] )
                    else:
                        pre_result[innndx][AC]=[segm]
                    
        for innndx,thf in enumerate(th_list):

            for f_ps in f_p:
                if  f_ps>=thf:
                    fb_li[innndx]+=1
            #这个是用来计算
            for probability  in p:
                if  probability>=thf:
                    pb_li[innndx]+=1
           
                #这个是用来计算recall的
            if len(p)!=0:
                if max(p)>=thf:
                    nb_li[innndx]+=1


        for f_ps in f_p:
            if  f_ps>=th:
                fb+=1
    aPC_li = [0 for i in th_list]

    print('Apc calculation!')
    for  ind,pre_re in enumerate(pre_result):
        # print(pre_re)
        aPC_li[ind] =  cal_apc(pre_re,ac_labels)
    # print(aPC_li)
    # assert(0)
    # print(f'We make a total prediction of :{total_pre}')

    print('-------------------------------------')
    save_flag = 0
    F1_li = [0]
    b_F1 = 0
    AL_aPC = 0
    thrrrr = 0
    for thsss,nbs,fbs,mnbs,aPC in zip(th_list,nb_li,fb_li,pb_li,aPC_li):
        # print(f'Threshold:{thsss} hit: {nbs} falsehit: {fbs} ')
        
        try:
            rcc = nbs/len(test_data)
            prec = mnbs/(mnbs+fbs)
            fff1 = rcc*prec*2/(rcc+prec)
            if fff1>=0.70:
                save_flag = 1
                F1_li.append(fff1)
            
        except:
            rcc = 0
            prec = 0
            fff1 = 0
        if fff1>b_F1:
            b_F1 = fff1
            AL_aPC = aPC
            thrrrr = thsss
        # print(f' F1 is {fff1} aPC:{aPC} thres is {thsss} ')
    # print(F1_li)
    print(f'Its Best F1 is {b_F1} aPC:{AL_aPC} thres is {thrrrr} ')
    print('-------------------------------------')
    return

def cCheck_model_inner(model, test_data, embeddings, seq2embeding):
    #用于训练
    CCCheck_model(model, test_data, embeddings)

    return 0


def get_check_data():
    # Load dataset for check
    seq2embeding = {}
    total = []
    print('d------------------d')
    from utils import load_mydict
    #Hyrid,  processed by A2KA first
    insp_train = load_mydict('./A2KA_genetest/Hybrid_test')
    #Yeast,  processed by A2KA first
    insp_train = load_mydict('./A2KA_genetest/Yeast_test')
    total+= insp_train
    total = check_and_return(total)
    total = del_recc(total)
    test_data = total
    seq_store = [item[0] for item in test_data]
    embeddings = generate_representation([1]*len(seq_store), seq_store)
    for seq,emb in zip(seq_store,embeddings):
        seq2embeding[seq] = emb
    return test_data, embeddings , seq2embeding


def DIRcheck():
    model = gnn_classifier().cuda()
    #indicate your model path
    try:
        Path = ''
        model.load_state_dict(torch.load(Path))
    except:
        print('model not found! please indicate a valid model path.')
    #prepare training data and load model
    test_data, embeddings, seq2embeding=  get_check_data()
    cCheck_model_inner(model, test_data, embeddings, seq2embeding)
    return 0


def get_result(batchs):
    model = gnn_classifier().cuda()
    #prepare training data and load model
    test_data, embeddings, seq2embeding= get_test_data()
    total = get_filter_total()
    data_store,tot_emb = get_data(total, seq2embeding)
    labeldict = get_label(total, seq2embeding)
    model = train_model(model,data_store, tot_emb, labeldict, batch_size=batchs, lr=0.000001, num_epochs=100)
    return 0


# DIRcheck is used for model performance check.
# DIRcheck()
#traingning and check model.
get_result(3)
