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


def process_segment(index, seg_li, f_ge):
    liss = defaultdict(int)
    for i, seg in enumerate(seg_li):
        local_liss = get_patterns(seg, {maxl: f_ge[maxl]}, {}, 2.5)
        for key, value in local_liss.items():
            liss[key] += value
        print(f'Chunk {index}: process the {i+1} sequence')
        print(f'Chunk {index} {len(seg_li)} sequence in total')
    return liss

def get_final_dic(full_d, f_ge, processnumber):
    seq_li = []
    for st in full_d:
        ss_li = st[2]
        for ss in ss_li:
            seq_li.append(ss)
    
    print(f'There are a total of {len(seq_li)} segments that need to be mined.')
    # Split the seq_li into processnumber pieces. 
    def chunks(data, n):
        it = iter(data)
        for _ in range(0, len(data), n):
            yield list(islice(it, n))
    
    chunked_seq_li = list(chunks(seq_li, len(seq_li) // processnumber))
    
    liss = defaultdict(int)

    with multiprocessing.Pool(processes=processnumber) as pool:
        results = [pool.apply_async(process_segment, args=(i, chunk, f_ge)) for i, chunk in enumerate(chunked_seq_li)]
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


#Maxl indicates the length of the list, and k represents the maximum length for a gap or a single continuous segment.
f_ge = {}
k = 3
maxl  = 5
for l in range(1,maxl+1):
    combinations = generate_combinations( k,l)
    f_ge[l] = combinations
    print(f"When the length of the k-th order subset is {k} and the maximum single item length is {l}, there are a total of {len(combinations)} possible scenarios.")


# example
tranfer_data = load_mydict(f'./Mine_material/Arabidopsis thaliana_0.5')
transfered = []
for item in tranfer_data:
    seq = item[0]
    ac = item[2]
    segls = item[3]
    segmetl = pos2seg(seq,segls)
    transfered.append([ac,seq,segmetl,[],[]])   

#define how many processors you used for calculation
processorsnumber = 10
final_count = get_final_dic(transfered,f_ge,processorsnumber)
