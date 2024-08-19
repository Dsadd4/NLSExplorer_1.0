
from collections import defaultdict
def get_continuous_patterns(s):
    substr_freq = defaultdict(int)
    
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            substr_freq[s[i:j]] += 1
    sorted_freq = sorted(substr_freq.items(), key=lambda x: x[1], reverse=True)
   
    ardicts = {substr: freq for substr, freq in sorted_freq}
    return ardicts

def merge_dictionaries(dict1, dict2):
    # 创建一个新字典，用于存储合并后的结果
    merged_dict = dict1.copy()
    
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value  # 如果键存在，值相加
        else:
            merged_dict[key] = value  # 如果键不存在，直接添加
    
    return merged_dict
def sort_dict_by_value(d, reverse=False):
    # 使用 sorted() 函数按值排序，返回一个按键值对排序的列表
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=reverse)
    
    # 将排序后的键值对列表转换为字典
    sorted_dict = dict(sorted_items)
    
    return sorted_dict

s = "从上述的数据库的收集结果可以看出， \
目前可靠的核定位信号的数目太少，这也是目前核定位信号预测方面的主要难点之一。\
 本文我主要是要利用 NLSdb2003 版以及 NLSdb 2017 版的核定位信号数据库\
 来构建基于核定位信号与非核定位信号的分类模型的数据集。"
s2 = '''
核定位信号（Nuclear Localization Signal, NLS）是一段特定的氨基酸序列，它能够指导蛋白质进入细胞核中。NLS是一种信号肽，通常位于蛋白质的C端或N端，能够与核进口受体（如karyopherins）结合，介导蛋白质通过核孔复合体进入细胞核。核定位信号在调控基因表达、细胞周期控制、信号传导等多种细胞功能中起着至关重要的作用。
以下是关于NLS的一些关键点：
NLS的分类：
经典NLS：经典NLS通常由一段富含赖氨酸（K）和精氨酸（R）的短肽序列组成。经典NLS可以进一步分为单部分NLS和双部分NLS。单部分NLS是一个连续的短肽序列，而双部分NLS由两个短肽序列组成，之间通常由10-12个氨基酸隔开。
非经典NLS：有些蛋白质携带的NLS不符合经典NLS的特征。这些非经典NLS通常具有多样性，可能包含不同的氨基酸组合或特殊的二级结构（如螺旋结构）。
NLS的识别和核进口机制：
Karyopherins的作用：Karyopherins是一类与NLS结合的核转运受体，主要包括importin α和importin β两种。Importin α能够识别并与NLS结合，而importin β则负责与核孔复合体相互作用，驱动蛋白质穿过核膜进入细胞核。
核孔复合体：核孔复合体（Nuclear Pore Complex, NPC）是细胞核膜上的一个大型蛋白质复合体，控制大分子进出细胞核的通道。NLS与importin复合物通过与NPC中的核孔蛋白相互作用，能够选择性地穿过核膜。
NLS的生物学功能：
基因表达调控：许多转录因子和其他与DNA结合的蛋白质依赖NLS进入细胞核，以启动或调控基因表达。例如，p53蛋白是一个著名的肿瘤抑制因子，其核定位对于其功能至关重要。
细胞周期控制：一些调控细胞周期的蛋白质在特定的细胞周期阶段会通过NLS被运输到细胞核，确保细胞分裂的正确进行。
信号转导：某些信号分子在细胞外信号刺激下，通过激活其NLS，进入细胞核以介导细胞响应。
NLS的研究与应用：
预测NLS：通过生物信息学工具可以预测蛋白质中可能存在的NLS。这对于理解蛋白质功能以及设计靶向药物非常重要。
人工设计NLS：在分子生物学实验中，研究者可以将NLS序列融合到非核定位的蛋白质中，使其能够进入细胞核，便于研究其在核内的功能。
NLS相关疾病：有些疾病与NLS功能的异常密切相关。例如，某些癌症中，p53的核定位异常会导致其失去正常的肿瘤抑制功能。此外，某些病毒（如HIV）也利用NLS来将自身的蛋白质或基因组导入宿主细胞核。
'''
segment = [s,s2]
def Continuous_mode(segment,cotsm= 10):
    results_cd = {}
    for strs in segment:
        tmp = get_continuous_patterns(strs)
        results_cd  = merge_dictionaries(results_cd , tmp)
    # if ' ' in results_cd:
    #     del results_cd[' ']
    results_cd = sort_dict_by_value(results_cd, reverse=True)
    cots = 0
    for item in results_cd :
        print(item)
        print(results_cd[item])
        if cots==cotsm:
            break
        cots+=1

Continuous_mode(segment,50)