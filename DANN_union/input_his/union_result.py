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
folder1_path = '/data/liyifan/片段计算/esm/单个序列input'
folder1_path_ = './单个序列输入'
folder2_path = './输入历史'


folder3_path = './单个序列推荐结果'
folder4_path = './结果历史保存'

folder5 ="/data/liyifan/progres/推荐片段中转"


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

os.system(f'/data/liyifan/anaconda3/envs/pyto/bin/python /data/liyifan/片段计算/esm/串联推荐_生成片段推荐结果.py {rec_rate} 5 f n ')
os.system(f'/data/liyifan/anaconda3/envs/pyto/bin/python /data/liyifan/片段计算/esm/串联推荐_单个序列大模型.py {rec_rate} 5 f ')
os.system('/data/liyifan/anaconda3/envs/progres/bin/python /data/liyifan/progres/串联推荐.py 3')
os.system('python top3-show.py')