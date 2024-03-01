from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_mydict
from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont



def pymol_show(ac,pos_li,pos_li2):
    
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
    
    #top3的颜色弄成不一样的
    color_set = ["green","blue","orange"]
    for color_index,posi in enumerate(pos_li2):
    
        pos_str = [f"{posi[0]}-{posi[1]}"]
        viewer.setStyle({"resi":pos_str},{"cartoon":{"color":color_set[color_index]}})

    # viewer.setStyle({"resi":["80-85"]},{"cartoon":{"color":"blue"}})
    # 在Jupyter Notebook中显示可视化
    viewer.zoomTo()
    # viewer.show()
    # 生成HTML字符串
    html_str = viewer.write_html()

    # 将HTML字符串保存到文件,输出给用户看
    with open(f"./DANN_union/result/{ac}_important_show.html", "w") as html_file:
        html_file.write(html_str)

def cal_size(text,font_path, font_size):
    return ImageDraw.Draw(Image.new('RGB', (10, 10))).multiline_textsize(text, font=ImageFont.truetype(font_path, font_size))

def judge_loc(loc,segment):
    if loc<=segment[1] and loc>=segment[0]:
        return True
    else:
        return False

def text_to_image(text, font_path='./Peptide_recomendation/ARIAL.TTF', font_size=20, output_path='output.png',set_figure =(1005,1000),seg=[],seg2=[]):
    # 创建一个白色背景的图片
   
    image = Image.new('RGB', set_figure, color = (255, 255, 255))
    
    # 在图片上绘制文本
    d = ImageDraw.Draw(image)
    print(seg)
    print(seg2)
    te_li = list(text)
    sum_wid = 0
    sum_height = 0
    for loc,item in enumerate(te_li):
        width, height = cal_size(item,font_path, font_size)
#         print(te_li)
    #     d.text((5,5), text, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
        flag = False
        flag2 = False
        
        for segment in seg:
            if judge_loc(loc,segment)==True:
                flag = True
        for segment2 in seg2:
            if judge_loc(loc,segment2)==True:
                flag2 = True
               
        # print('-----')
        # print(flag)
        # print(flag2)
        if flag == True :
            if flag2 == False:
                print('1')
                rect_position = [(sum_wid, sum_height), (sum_wid + width, sum_height + height)]
                d.rectangle(rect_position, fill=(255, 255, 0))
                d.multiline_text((sum_wid,sum_height), item, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
            else:
                print('2')
                rect_position = [(sum_wid, sum_height), (sum_wid + width, sum_height + height)]
                d.rectangle(rect_position, fill=(0, 255, 255))
                d.multiline_text((sum_wid,sum_height), item, fill=(0, 0, 0), font=ImageFont.truetype(font_path, font_size))
#            d.rectangle([(5, 5), (sum_wid,sum_height)], fill= (200, 0, 150))
            
        
        else:
            # print('3')
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

seq_show = load_mydict('./DANN_union/vis_trans/seq_show')
text = seq_show[0]






first_re = load_mydict('./DANN_union/vis_trans/first_recom')
last_re = load_mydict('./DANN_union/vis_trans/last_recom')
ac = first_re[0]
pos_li = list(set(first_re[1]) - set(last_re[1]))
pos_li2 = last_re[1]
# pymol_show(ac,pre_seg[0])
# print(first_re)
# print(last_re)
path = f'./DANN_union/result/{ac}-animal.png'
text_to_image(text, output_path=path,font_size=30,seg = first_re[1],seg2 = last_re[1])
pymol_show(ac,pos_li,pos_li2)