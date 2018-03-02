import os
import json

path = 'E:/crop_data/AFEW-VA'
json_path = 'E:/AFEW-VA'
labels_path = '../AFEW-VA/crop/labels.txt'
data_path = '../AFEW-VA/crop/data.txt'

# # 一级目录
# for file1 in os.listdir(path):
#     # 二级目录
#     for file2 in os.listdir(os.path.join(path, file1)):
#         # 文件
#         for file in os.listdir(os.path.join(path, file1, file2)):
#             # 找到 png 文件
#             if file.find('.png')>0:
#                 # 重命名
#                 new_name = file2 + file
#                 os.rename(os.path.join(path, file1, file2, file), os.path.join(path, file1, file2, new_name))

# # 一级目录
# for file1 in os.listdir(path):
#     # 二级目录
#     for file2 in os.listdir(os.path.join(path, file1)):
#         # 文件
#         for file in os.listdir(os.path.join(path, file1, file2)):
#             # 找到 json 文件
#             if file.find('.json')>0:
#                 # 保存
#                 json_files.append(os.path.join(path, file1, file2, file))

json_files = []
va_labels = []
image_list = []


def findfiles(dir, wildcard, recursion, file, image_list):
    exts = wildcard.split(" ")
    files = os.listdir(dir)

    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            findfiles(fullname, wildcard, recursion, file, image_list)
        else:
            for ext in exts:
                if name.endswith('.png'):
                    file.write(fullname + ' 1' + "\n")
                    image_list.append(name)
                    break


def findjsons(dir, wildcard, recursion, list):
    exts = wildcard.split(" ")
    files = os.listdir(dir)

    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            findjsons(fullname, wildcard, recursion, list)
        else:
            for ext in exts:
                if name.endswith('.json'):
                    list.append(fullname)
                    break


with open(data_path, 'w') as file:
    findfiles(path, '.png', 1, file, image_list)

findjsons(json_path, '.json', 1, json_files)


# 解析 json 文件
for i in range(len(json_files)):

    with open(json_files[i], 'r') as f:
        data = json.load(f)
        data = data['frames']
        n = json_files[i].split('.')[0].split('\\')[-1]
        for k, v in data.items():
            va_labels.append([n+k+'.png', v['valence'], v['arousal']])

# 保存 labels
va_labels = sorted(va_labels)

# with open(labels_path, 'w') as f:
#     for i in range(len(va_labels)):
#         f.write(va_labels[i][0] + ' ' + str(va_labels[i][1]) + ' ' + str(va_labels[i][2]) + '\n')

# 保存 crop image labels
with open(labels_path, 'w') as f:
    for i in range(len(va_labels)):
        if va_labels[i][0] in image_list:
            f.write(va_labels[i][0] + ' ' + str(va_labels[i][1]) + ' ' + str(va_labels[i][2]) + '\n')
