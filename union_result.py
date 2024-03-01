import os
import shutil

def clear_folder(folder_path):
    # 删除文件夹内所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def move_files(source_folder, destination_folder):
    # 将源文件夹内的文件移动到目标文件夹
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        try:
            shutil.move(source_path, destination_path)
        except Exception as e:
            print(f"Failed to move {source_path} to {destination_path}. Reason: {e}")

# 文件夹路径
folder1_path = './Peptide_recomendation/single_input'
folder1_path_ = './DANN_union/sequence_input'
folder2_path = './DANN_union/input_his'


folder3_path = './DANN_union/result'
folder4_path = './DANN_union/result_his'

folder5 ="./progres/recomenda_trans"


if os.listdir(folder5 ):
    clear_folder(folder5 )

if os.listdir(folder3_path ):
    move_files(folder3_path , folder4_path)
# 判断文件夹1是否有文件，有则删除
if os.listdir(folder1_path):
    move_files(folder1_path, folder2_path) 
    move_files(folder1_path_, folder1_path)
else:
    move_files(folder1_path_, folder1_path)






rec_rate = 0.3

# /data/liyifan/anaconda3/envs/NLSExplorer-pyto/bin/python
# /data/liyifan/anaconda3/envs/NLSExplorer-pyto/bin/python
# /data/liyifan/anaconda3/envs/NLSExplorer-progres/bin/python
# pasta your python environment address below
pyto_addres = ''
progres_addres = ''
# for example
# progres_addres = '/data/liyifan/anaconda3/envs/NLSExplorer-pyto/bin/python'
# pyto_addres = '/data/liyifan/anaconda3/envs/NLSExplorer-progres/bin/python'


os.system(f'{pyto_addres} ./Peptide_recomendation/Concatenated_Recommendation_result_generate.py {rec_rate} 5 f n ')
os.system(f'{pyto_addres} ./Peptide_recomendation/Concatenated_Recommendation_lanmodel.py {rec_rate} 5 f ')
os.system(f'{progres_addres} ./progres/Concatenated_Recommendation.py 3')
os.system('python top3-show.py')