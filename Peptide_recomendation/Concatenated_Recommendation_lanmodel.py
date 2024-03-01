import torch
#python 单个序列推荐_大模型.py 0.15 5 f
# python 单个序列推荐_大模型.py 0.15 4 f
#0.15表示抓取的片段比例 4代表最小的片段长度 f表示要不要过滤掉单个位点
setdivice = 0
# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
# device=torch.device(f"cuda:{setdivice}" if torch.cuda.is_available() else "cpu")
# device2 = torch.device("cpu")

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



BaseDir = './Peptide_recomendation/deeploc_dataset/'
object = 'Swissprot_Train_Validation_dataset.csv'
Path = BaseDir +object
data = pd.read_csv(Path)

namelist=[]
for i in range(0,11):
    namelist.append(data.columns[4:-1].values[i])


from utils import save_mydict,load_mydict

mul_y = []
merge_l = zip(data[namelist[0]],data[namelist[1]],data[namelist[2]],data[namelist[3]],data[namelist[4]],data[namelist[5]],data[namelist[6]],data[namelist[7]],data[namelist[8]],data[namelist[9]],data[namelist[10]])
for item in merge_l:
    mul_y.append(list(item))

datts = data['Sequence']
mul_datt=[]
for item in datts:
    if len(item)>=1023:
        item = item[0:1022]
    mul_datt.append(item)
data0 = data

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
from Bio import SeqIO
import os
path = './Peptide_recomendation/single_input'
names_d = os.listdir(path)
print(names_d)
na_m = ''
for nd in names_d:
    if nd.split('.')[1]=='fasta':
        na_m = nd

for record in SeqIO.parse(path+'/'+na_m , "fasta"):
    print(record.seq)
# print(dir(record))
#这里得到ac信息
ac =    record.id.split('|')[1]    
seq_li = [record.seq]
# print(len(record.seq))
# print("*"*300)
#选出长度小于1024的序列 存在字典和列表里面
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

print(len(mul_datt))
to_cal = train_total[0]

#加载模型
EMB_DIM=1280
model2 = LSTMTagger( EMB_DIM, 1280).cuda()
# Path_ = "./4_enhance60_one_last"
Path_ = "./Peptide_recomendation/NLS_loc_modeltes"
# Path_ = "./model_store/NLS_loc_model_45tes"
model2.load_state_dict(torch.load(Path_))

from utils import getBatch
from torch.nn.utils.rnn import pad_sequence,pack_sequence
test_p=[]
true = []

att_dis = []
with torch.no_grad():
    model2.eval()
    embedding = to_cal.unsqueeze(0)
    # tag_scores = model2(pad_sequence(embedding,batch_first=True).cuda())
    tag_scores = model2(embedding.cuda())
    pre = torch.stack(tag_scores[1]).sum(0).cpu().detach().numpy()
    for rep in pre:
        att_dis.append(rep)
#         break

pre_seg_att = att_dis

# thred = 0.3

import sys


cofactor = float(parameter[1])
recom_complete = int(parameter[2])
seqlen = len(record.seq)

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
def seg_generate(pre_seg_att,cofactor,recom_complete,f_c):
    import numpy as np
    pre_seg = []
    for fig in pre_seg_att:
        y= F.softmax(torch.tensor(fig).reshape(-1)).cpu().numpy()
    #     print(y)
        p_y = np.log(300*y)
        c = F.relu(torch.tensor(p_y))
        import numpy as np
        length = len(seq)
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
                    
        
        pre_seg.append(ts_seg_d)
    return pre_seg


#     break
    

    
#我们现在需要不考虑单个位点信息的
# 单个位点信息去除函数

f_c = parameter[3]
pre_seg = seg_generate(pre_seg_att,cofactor,recom_complete,f_c)
    
print(pre_seg)
# print(pre_seg_att)
def show_simple_seq(seq,pre_segment):
    flag = 0
    for i,item in enumerate(seq):
        for th in pre_segment:
            if th[0]<=i and i<=th[1]:
                print(f'\033[7;31m{item}\033[0m',end='')
                flag = 1
                break
        if flag == 0:
            print(f'{item}',end='')
        flag=0
    print(' ')
        
show_simple_seq(mul_datt[0],pre_seg[0])

#开始封装完整函数
import os
import glob
from PIL import Image
 
def show_sequence_attention(atten_distribu,seq):
    read_img_size = (75,129)
    # 加载每个图像，并调整为200x200像素大小（如果需要）
    new_image = Image.new("RGB",(30000, 3000),(255,255,255))
    coefi = atten_distribu[0]
    now_image_size = (read_img_size[0],int(read_img_size[1]*atten_distribu[0]))
    
    image_file = f'./Peptide_recomendation/images/{seq[0]}.jpg'
    str_img = Image.open(image_file)
    str_img= str_img.resize(now_image_size)
    
    begin_point = (0,1000)
    new_image.paste(str_img, begin_point)
    
    now_left_down = (begin_point[0]+now_image_size[0],begin_point[1]+now_image_size[1])
    
    
#     new_image.show()
    # print(now_left_down)
    
    for i,str_ in enumerate(seq[1:]):
        
        image_file = f'./Peptide_recomendation/images/{str_}.jpg'
        str_img = Image.open(image_file)
        coefi = atten_distribu[i+1]
        
        now_image_size = (read_img_size[0],int(read_img_size[1]*coefi))
        str_img = str_img.resize(now_image_size)
        
        #左上角需要贴的位置
        now_left_up = (now_left_down[0],now_left_down[1]-now_image_size[1])
        # print(now_left_up)
        new_image.paste(str_img,now_left_up)
#         new_imagepaste(str_img,now_left_down)
#         new_image.show()
#         break
        #计算贴好图之后的左下角位置非常关键
        now_left_down = (now_left_down[0]+now_image_size[0],now_left_down[1])

    # new_image.show()
    new_image.save(f'./DANN_union/result/{ac}-seq.jpg')


from PIL import Image






import numpy as np
#筛子函数，循环起来就大的越大，小的越小
def soft_1000max(x):
    # 计算每行的最大值
    row_max = np.max(x)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp)
    s = 1000*x_exp / x_sum
    
    return s

dis = pre_seg_att[0]
for i in range(0,3):
    dis = soft_1000max(dis)
    
dis = [max(1, value) for value in dis]   
dis = [min(10, value) for value in dis]   
  
show_sequence_attention(dis,mul_datt[0])


    
import seaborn as sns
import matplotlib.pyplot as plt

def attenco_plt(att2):
    plt.figure(figsize=(50, 8), dpi=500)
    sns.lineplot(data=att2)
    
    plt.xlabel('length', fontsize=20)
    plt.yticks([])
    plt.xticks(fontsize=20)
    plt.title('The Attention Distribution Inside The Sequence', fontsize=20)
    
    plt.savefig(f'./DANN_union/result/{ac}-attdistri.png', bbox_inches='tight')


import matplotlib.pyplot as plt
import seaborn as sns

def attenco_plt(att2):
    plt.figure(figsize=(50, 8), dpi=500)
    ax = sns.lineplot(data=att2, legend=False, linewidth=2)  # 不显示图例，线条加粗
    
    # 设置坐标轴标签和标题的字体大小
    ax.set_xlabel('length', fontsize=38)
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=38)
    ax.set_title('The Attention Distribution Inside The Sequence', fontsize=38)
    
    # 保存图像
    plt.savefig(f'./DANN_union/result/{ac}-attdistri.png', bbox_inches='tight')

# 请注意，ac 是你的变量，你可能需要提前定义它。

def attenco_plt(att2):
    plt.figure(figsize=(50, 8), dpi=500)
    x_values = range(len(att2))
    plt.plot(x_values, att2, color='blue', linewidth=5)  # 绘制曲线
    plt.fill_between(x_values, min(att2), att2, color='blue', alpha=1)  # 填充曲线下方的区域

    plt.xlabel('length', fontsize=38)
    plt.yticks([])
    plt.tick_params(axis='x', labelsize=38)
    plt.title('The Attention Distribution Inside The Sequence', fontsize=38)

    plt.savefig(f'./DANN_union/result/{ac}-attdistri.png', bbox_inches='tight')



# print(pre_seg_att[0])
f_so =[item[0] for item in pre_seg_att[0] ]

attenco_plt(f_so)
    
    
    
# attenco_plt(pre_seg_att[0])

from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont
def cal_size(text,font_path, font_size):
    return ImageDraw.Draw(Image.new('RGB', (10, 10))).multiline_textsize(text, font=ImageFont.truetype(font_path, font_size))

def judge_loc(loc,segment):
    if loc<=segment[1] and loc>=segment[0]:
        return True
    else:
        return False

def text_to_image(text, font_path='./Peptide_recomendation/ARIAL.TTF', font_size=20, output_path='output.png',set_figure =(1005,1000),seg=[]):
    # 创建一个白色背景的图片
   
    image = Image.new('RGB', set_figure, color = (255, 255, 255))
    
    # 在图片上绘制文本
    d = ImageDraw.Draw(image)
    
    te_li = list(text)
    sum_wid = 0
    sum_height = 0
    for loc,item in enumerate(te_li):
        width, height = cal_size(item,font_path, font_size)
#         print(te_li)
    #     d.text((5,5), text, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
        flag = False
        for segment in seg:
            if judge_loc(loc,segment)==True:
                flag = True
        if flag == True:
            rect_position = [(sum_wid, sum_height), (sum_wid + width, sum_height + height)]
            d.rectangle(rect_position, fill=(255, 255, 0))
            d.multiline_text((sum_wid,sum_height), item, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
#             d.rectangle([(5, 5), (sum_wid,sum_height)], fill= (200, 0, 150))
        else:
            d.multiline_text((sum_wid,sum_height), item, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
#         d.multiline_text((width,5), text, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
        sum_wid+=width
        if sum_wid>set_figure[0]-20:
            sum_height+=height
            sum_wid =0
        
    
    
    # 保存图片
    image.save(output_path)

# 你想要的文本内容
# 转换文本并保存为图片
text = mul_datt[0]
seg = pre_seg[0] 

save_mydict([text,seg],'./DANN_union/vis_trans/seq_show')

for_recom_cal = [[ac,str(text),seg]]
save_mydict(for_recom_cal,'/data/liyifan/progres/single_input/single_rec')



def pymol_show(ac,pos_li):
    
    import py3Dmol

    # 创建3Dmol视图
    size=(800,800)
    # viewer = py3Dmol.view(width=300, height=300)
    viewer = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width=size[0], height=size[1])

    # 添加分子结构
    pdb_data = open(f"./Peptide_recomendation/single_input/{ac}.pdb").read()
    viewer.addModel(pdb_data, "pdb")
    
    # 设置标红样式
    viewer.setStyle({"cartoon": {"color": "white"}})
    
    for posi in pos_li:
        
        pos_str = [f"{posi[0]}-{posi[1]}"]
        # print(pos_str)
        viewer.setStyle({"resi":pos_str},{"cartoon":{"color":"red"}})
   


    # viewer.setStyle({"resi":["80-85"]},{"cartoon":{"color":"blue"}})
    # 在Jupyter Notebook中显示可视化
    viewer.zoomTo()
    # viewer.show()
    # 生成HTML字符串
    html_str = viewer.write_html()

    # 将HTML字符串保存到文件,输出给用户看
    with open(f"./DANN_union/result/{ac}_important_show.html", "w") as html_file:
        html_file.write(html_str)



# pymol_show(ac,pre_seg[0])
save_mydict([ac,pre_seg[0]],'./DANN_union/vis_trans/first_recom')


from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def annotate_bars(ax, height_offset=0.05, font_size=8):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() + height_offset),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=font_size)

def analyze_protein_structure(pdb_file, chain_id, residue_ranges, save_path=None):
    # 创建PDB解析器
    # 设置全局字体大小和加粗效果
    plt.rcParams['font.size'] = 38
    plt.rcParams['font.weight'] = 'bold'
    parser = PDBParser(QUIET=True)

    # 解析PDB文件
    structure = parser.get_structure("protein", pdb_file)

    # 用于存储结构信息的字典
    structure_data = {}

    # 创建DSSP对象
    dssp = DSSP(structure[0], pdb_file)  # Assuming there is only one model in the structure

    # 遍历蛋白质链中的残基
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                ppb = PPBuilder()
                polypeptides = ppb.build_peptides(chain)

                for poly_index, poly in enumerate(polypeptides):
                    for index, residue in enumerate(poly):
                        for range_name, (start_residue, end_residue) in residue_ranges.items():
                            if start_residue <= residue.id[1] <= end_residue:
                                # 获取结构信息
                                residue_id = (residue.parent.id, residue.id)
                                structure_data.setdefault(range_name, []).append(dssp[residue_id][2])

    # 输出结构信息统计
    print("Structure Information Counts:")
    for range_name, structure_info in structure_data.items():
        print(f"{range_name}: {len(structure_info)} residues")

    # 获取唯一的结构类型
    unique_structures = set([item for sublist in structure_data.values() for item in sublist])

    # 补充缺失的结构类型，将结构统计为0的数据加入字典中
    for range_name in residue_ranges.keys():
        if range_name not in structure_data:
            structure_data[range_name] = [0] * sum(len(v) for v in structure_data.values())

    # 绘制美化的条形图
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("Set2", n_colors=len(unique_structures))  # 使用Seaborn提供的调色板
    
    # 使用pd.concat将不同范围的结构信息连接起来
    df = pd.concat([pd.Series(structure_info, name=range_name) for range_name, structure_info in structure_data.items()], axis=1)
    
    ax = sns.countplot(data=pd.melt(df), x="variable", hue="value", palette=colors)
    plt.title("Structure information distribution",fontsize=45)
    plt.xlabel("NLS range",fontsize=35)
    plt.ylabel("Structure count",fontsize=28)
    plt.xticks(rotation=45,fontsize=15)  # 旋转x轴标签，使其更清晰可读
    # 设置legend字体大小
    legend = ax.legend(title="Legend", prop={'size': 25})
    legend.get_title().set_fontsize(25)
    sns.despine()  # 移除上、右边框线
    
#     # 在每个柱子上面显示 y 值
#     annotate_bars(ax)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
plt.rcParams.update(plt.rcParamsDefault)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Bio import PDB
from Bio.PDB import PDBParser, DSSP, PPBuilder



# def analyze_protein_structure(pdb_file, chain_id, residue_ranges, save_path=None):
#     # 省略部分代码...

#     # 绘制美化的条形图
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(12, 6))
#     colors = sns.color_palette("Set2", n_colors=len(unique_structures))  # 使用Seaborn提供的调色板

#     # 使用pd.concat将不同范围的结构信息连接起来
#     df = pd.concat([pd.Series(structure_info, name=range_name) for range_name, structure_info in structure_data.items()], axis=1)

#     ax = sns.countplot(data=pd.melt(df), x="variable", hue="value", palette=colors)
#     plt.title("Structure Information Distribution")
#     plt.xlabel("Nls Range")
#     plt.ylabel("Structure Count")
#     plt.xticks(rotation=45)  # 旋转x轴标签，使其更清晰可读
#     sns.despine()  # 移除上、右边框线

#     # 保存图像
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
#         print(f"Visualization saved to {save_path}")
#     else:
#         plt.show()

# # 恢复默认字体设置
# plt.rcParams.update(plt.rcParamsDefault)




# 调用函数进行分析、可视化并保存结果
pdb_file = f"./Peptide_recomendation/single_input/{ac}.pdb"
chain_id = "A"

seg
residue_ranges = {str(item):item for item in seg}


save_path = f"{ac}-struct-statis.png"
save_path = f"./DANN_union/result/{ac}_struct-statis.png"



analyze_protein_structure(pdb_file, chain_id, residue_ranges, save_path)
