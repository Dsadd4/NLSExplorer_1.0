from itertools import product
from utils import load_mydict
import numpy as np
from collections import Counter
import multiprocessing
from collections import defaultdict
from itertools import islice

def generate_combinations(length, k):
    # generate all possible composition
    numbers = list(range(1,k + 1))
    all_combinations = list(product(numbers, repeat=length))

    return all_combinations

def calculate_entropy_with_expectation(sequence, unknown_symbol='*', amino_acids='ACDEFGHIKLMNPQRSTVWY'):
    """
    To calculate the Shannon entropy of an amino acid sequence containing unknown amino acids, the method of maximum entropy is used.
    Parameters:
    sequence (str): The amino acid sequence containing unknown amino acids (with unknown amino acids represented by *)
    unknown_symbol (str): The symbol representing unknown amino acids
    amino_acids (str): All possible amino acid characters
    Returns:
    float: The Shannon entropy of the amino acid sequence
    """
    known_part = [aa for aa in sequence if aa != unknown_symbol]
    length_known = len(known_part)
    length_total = len(sequence)
    frequencies_known = Counter(known_part)
    frequencies_total = {aa: frequencies_known.get(aa, 0) for aa in amino_acids}
    num_unknowns = sequence.count(unknown_symbol)
    probability_unknown_each = num_unknowns / len(amino_acids)
    for aa in amino_acids:
        frequencies_total[aa] += probability_unknown_each
    probabilities_total = {aa: freq / length_total for aa, freq in frequencies_total.items()}
    entropy = -sum(p * np.log2(p) for p in probabilities_total.values() if p > 0)
    return entropy

def get_patterns(sequence, feature_groups, pattern_dict,entropth):
    for name in feature_groups:
        patterns = feature_groups[name]
        for pattern in patterns:
            for start_idx in range(0, len(sequence)):
                subseq = sequence[start_idx:]
                if sum(pattern) > len(subseq):
                    continue
                current_pos = 0
                pattern_str = ""
                failed_flag = False
                for i, num in enumerate(pattern):
                    if i % 2 == 0:  # Keep the sequence part
                        pattern_str += subseq[current_pos:current_pos + num]
                    else:  # Replace the sequence part with '*'
                        pattern_str += '*' * num
                    current_pos += num
                if not failed_flag and calculate_entropy_with_expectation(pattern_str)>entropth:
                    if pattern_str in pattern_dict:
                        pattern_dict[pattern_str] += 1
                    else:
                        pattern_dict[pattern_str] = 1
    return pattern_dict

def counting_star(set_name):
    i = 0
    for item in set_name:
        if item == '*':
            i+=1
    return i


def counting_diversity(set_name):

    li = []
    for item in set_name:
        li.append(item)
    
    return len(set(li))


def filter_gap_sets(select_lent,least_stars_n,entropy_th,final_cout,min_times):
    rt_dict = {}
    for sets in final_cout:
        if len(sets)==select_lent and counting_star(sets)>=least_stars_n and calculate_entropy_with_expectation(sets)>=entropy_th and final_cout[sets]>=min_times:
#             print(f'{sets}:{final_cout[sets]}')
            rt_dict[sets] = final_cout[sets]
    return rt_dict



def pos2seg(seq,listss):
    res = []
    for posit in listss:
        res.append(seq[posit[0]:posit[1]])
    return res


def process_segment(index, seg_li, f_ge,maxl,entropthss):
    liss = defaultdict(int)
    for i, seg in enumerate(seg_li):
        #This place to set the minimum entropy of discontinous
        local_liss = get_patterns(seg, {maxl: f_ge[maxl]}, {}, entropthss)
        for key, value in local_liss.items():
            liss[key] += value
        print(f'Chunk {index}: process the {i+1} sequence')
        print(f'Chunk {index} {len(seg_li)} sequence in total')
    return liss






def SCNLS_f(seq_li, f_ge, processnumber,maxl,entropthss):
    print(f'There are a total of {len(seq_li)} segments that need to be mined.')
    # Split the seq_li into processnumber pieces. 
    def chunks(data, n):
        it = iter(data)
        for _ in range(0, len(data), n):
            yield list(islice(it, n))
    
    chunked_seq_li = list(chunks(seq_li, len(seq_li) // processnumber))

    liss = defaultdict(int)
    with multiprocessing.Pool(processes=processnumber) as pool:
        results = [pool.apply_async(process_segment, args=(i, chunk, f_ge,maxl,entropthss)) for i, chunk in enumerate(chunked_seq_li)]
        for i, result in enumerate(results):
            try:
                segment_liss = result.get()
                for key, value in segment_liss.items():
                    liss[key] += value
        
                print(f'----------- Completed mining segment {i} --------')
            except Exception as e:
                print(f'Error processing chunk {i+1}: {e}')

    sorted_substring_counts = dict(sorted(liss.items(), key=lambda item: item[1], reverse=True))
    final_cout = {}
    for name in sorted_substring_counts:
        couttt = sorted_substring_counts[name]
        n_ame = name.strip('*')
        if n_ame not in final_cout:
            final_cout[n_ame] = couttt
    return final_cout



#define how many processors you used for calculation

#Maxl indicates the length of the list, and k represents the maximum length for a gap or a single continuous segment.

def function_mode(for_digg,k,maxl,processorsnumber,entropthss):
    
    f_ge = {}
    # k = 3
    # maxl  = 8
    # processorsnumber = 2
    for l in range(1,maxl+1):
        combinations = generate_combinations( k,l)
        f_ge[l] = combinations

    # print(f_ge)
    print(f"When the length of the k-th order subset is {k} and the maximum single item length is {l}, there are a total of {len(combinations)} possible scenarios.")
    RE = SCNLS_f(for_digg, f_ge, processorsnumber,maxl,entropthss)

    Show = 10
    for item in RE:
        print(f'Pattern:{item}')
        print(f'Occurence:{RE[item]}')
        Show-=1
        if Show==0:
            break

# SCNLS_N(transfered,f_ge,processorsnumber)

# final_count = get_final_dic(transfered,f_ge,processorsnumber)

import argparse

import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Add parameters
    parser.add_argument('--mode', type=str, required=True, help='Mode of operation (e.g., f)')
    parser.add_argument('--material', type=str, help='Path to the material file (e.g., example.csv)')
    parser.add_argument('--maxgap', type=int, help='Maximum gap allowed (e.g., 5)')
    parser.add_argument('--kths', type=int, help='Threshold value (e.g., 3)')
    parser.add_argument('--processor', type=int, help='Number of processors to use (e.g., 10)')
    parser.add_argument('--entropythreshold', type=float,default=0, help='the entropy threshold')

    # Parse parameters
    args = parser.parse_args()
    #default entropy
    entropthss = args.entropythreshold
    # Parameters usage
    print(f"Mode: {args.mode}")
    if args.mode == 'f':
        # for_digg = ['MQAKINSFFKPSSSSSGQSDFLLRHCAECGAKYAPGDELDEKNHQSFHKDYMYGLPFKGWQNEKAFTSPLKAQLIDTHFS',
        #             'FIKNRIVMVSENDSPAHRNKVQEVVKMMEVELGEDWILHQHCKVYLFISSQRISGCLVAEPIKEAFKLIASPDDERQLQKESSSSPSTSIQFGNIVLQREVSKRCRTSDDRLDNGVIVCEEEAKPAVCGIRAIWVSPSNRRKGIATWLLDTTRESFCNNGCMLEKSQLAFSQPSSIGRSFGSKYFGTCSFLLY',
        #             'IAASVTTDTDDGLAVWENNRNAIVNTYQRRSAITERSEVLKGCIEKTLKKGSSSVPKNHKKKRNYTQFHLEL']
        path = args.material
        data = pd.read_csv(path)
        # print(data)
        segment = list(data['Recommended Segment'])
        # print(segment)
        # assert(0)
        function_mode(segment, args.kths, args.maxgap, args.processor,entropthss)
    elif args.mode == 's':
        # if not args.segment:
        #     parser.error("--segment is required when mode is 's'")

        for_digg = [args.material]
        function_mode(for_digg, args.kths, args.maxgap, args.processor,entropthss)

    elif args.mode == 'n':
        from utils import load_mydict
        path = args.material
        tranfer_data = load_mydict(path)

        transfered = []
        for item in tranfer_data:
            seq = item[0]
            ac = item[2]
            segls = item[3]
            segmetl = pos2seg(seq,segls)
            transfered+=  segmetl
        # print(transfered)
        function_mode(transfered, args.kths, args.maxgap, args.processor,entropthss)
        pass
    else:
        parser.error(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()





# python SCNLS.py --mode f  --material example.csv --maxgap 3 --kths 3 --processor 3
# python SCNLS.py --mode n  --material 'Arabidopsis thaliana_0.5' --maxgap 3 --kths 3 --processor 10 
# python SCNLS.py --mode s  --material KKKKRRRJJJJKSJSAIJCOSJAOJD --maxgap 3 --kths 3 --processor 1 
"从上述的数据库的收集结果可以看出， 目前可靠的核定位信号的数目太少，这也是目前核定位信号预测方面的主要难点之一。 本文我主要是要利用 NLSdb2003 版以及 NLSdb 2017 版的核定位信号数据库来构建基于核定位信号与非核定位信号的分类模型的数据集。"

'''python SCNLS.py --mode s  --material "从上述的数据库的收集结果可以看出， 目前可靠的核定位信号的数目太少，这也是目前核定位 信号预测方面的主要难点之一。 本文我主要是要利用 NLSdb2003 版以及 NLSdb 2017 版的核定位信号数据库来构建基于核定位信号与非核定位信号的分类模型的数据集。" --maxgap 5 --kths
 5 --processor 1'''

''' python SCNLS.py --mode s  --material KKKKRRRJJrrJJccKSJSArrIJccCOSrrJAccOJDrrasccda --maxgap 3 --entropythreshold 0 --kths 3 --processor 1 '''