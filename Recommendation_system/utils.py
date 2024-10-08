
import pandas as pd
import numpy as np

def get_data(item,typ):
    BaseDir = './humploc_c/'
    object = item+ ' proteins.{}.csv'.format(typ)
    Path = BaseDir +object
    data = pd.read_csv(Path)
    return data['label'],data['seq']

from torch import nn, optim
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import requests

def request_page(url):

    response = requests.get(url)
    if response.status_code == 200:
        print('succucess!')
        return response.content
    else:
        print(response)
        print('failed try!')
        return None


def loop_find(target, str, start=0):
    li = []
    while True:
        beg = target.find(str, start)
        if beg == -1:
            break
        li.append(beg)
        start = beg + 1
    return li



def save_mydict(dict, name):
    # 字典保存
    import pickle
    f_save = open(name + '.pkl', 'wb')
    pickle.dump(dict, f_save)
    f_save.close()


def load_mydict(name):
    import pickle
    # # 读取
    f_read = open(name + '.pkl', 'rb')
    dict2 = pickle.load(f_read)
    f_read.close()
    return dict2

def train(model,Dataload,epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_losses, test_losses = [], []
    for i in range(epochs):
        for data,label in Dataload:
            optimizer.zero_grad()
            pre = model(data)
            loss = criterion(pre,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model


from tqdm import tqdm
def generate_representation(labels, datt):
    print('Now we begin to generate embedding \n------------****-----------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cpu")
    import esm
    
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    
    # model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    sequence_representations = []
    sequence_total = []
    labels_list = []
    
    for i in tqdm(range(0, len(datt), 10),desc='processing'):
        data = list(zip(labels[i:i + 10], datt[i:i + 10]))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # print('------')
        # print(len(batch_tokens))
        # print(len(labels_list))
        # print(len(sequence_representations))
        # print('------')
        for item in batch_labels:
            labels_list.append(item)

        batch_tokens = batch_tokens.cuda()
        # print(batch_tokens)
        # Extract per-residue representations
        with torch.no_grad():
            model = model.cuda()

            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]
        #         print(token_representations[1])
        #         print('representation now have {} data '.format(len(token_representations)))
        # print('total proecess {}'.format(i))
        for j, (_, seq) in enumerate(data):
            token_representations = token_representations.cpu()

            sequence_total.append(token_representations[j, 1: len(seq) + 1])
    return sequence_total

def generate_representation_includtoken(labels, datt):
    print('Now we begin to generate embedding \n------------****-----------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cpu")
    import esm
    # Load ESM-1b model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    
    
    
    
    batch_converter = alphabet.get_batch_converter()
    sequence_representations = []
    sequence_total = []
    token_total = []
    labels_list = []
   
    for i in tqdm(range(0, len(datt), 10),desc='processing'):
        data = list(zip(labels[i:i + 10], datt[i:i + 10]))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # print('------')
        # print(len(batch_tokens))
        # print(len(labels_list))
        # print(len(sequence_representations))
        # print('------')
        for item in batch_labels:
            labels_list.append(item)
        for token in batch_tokens:
            token_total.append(token)
            # print(token.size())

        batch_tokens = batch_tokens.cuda()
        # print(batch_tokens)
        # Extract per-residue representations
        with torch.no_grad():
            model = model.cuda()

            results = model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]
        #         print(token_representations[1])
        #         print('representation now have {} data '.format(len(token_representations)))
        # print('total proecess {}'.format(i))
        for j, (_, seq) in enumerate(data):
            token_representations = token_representations.cpu()

            sequence_total.append(token_representations[j, 1: len(seq) + 1])
    return sequence_total,token_total

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1280, 1)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        return torch.sigmoid(self.fc1(x))


def getBatch(batch_size, train_data):
    import random
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


import numpy as np

def get_padding(train_total):
    numpy_value=[]
    for item in train_total:
        num_list = list(item.numpy())
        while len(num_list)<1022:
            num_list.append((np.zeros(1280)))
        numpy_value.append(num_list)
    return numpy_value
