from sklearn.model_selection import train_test_split
import os
from shutil import copyfile
# 读取所有文件。建立字典，Non映射为0， 其他映射为1.
# 1. 检查所有文件不重名。 有重名。这怎么办？
# 每个文件移动后加上V_和NV_前置。表示是否有漏洞。怎么随机选择呢？
# train_test_split。建立y之后，移动
data_path = ["data/vulner/Non_vulnerable", "data/vulner/Vulnerable_funcitons"]

file_origin_path = list()
file_label = list()

for dir_p in data_path:
    for roots, dirs, files in os.walk(dir_p):
        for file in files:
            path = os.path.join(roots, file)
            file_origin_path.append(path)
            if "Non" in path:
                file_label.append(0)
            else:
                file_label.append(1)

x_train, X_test, y_train, y_test = train_test_split(file_origin_path,
                                                    file_label,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=42,
                                                    stratify=file_label)

from collections import Counter
print("训练样本中，样本分布", Counter(y_train))
print("测试样本中，样本分布", Counter(y_test))


target = "data/vulner/train_data/raw"
for index, file in enumerate(x_train):
    start_tag = "Vulner_" if y_train[index] == 1 else "NotVulner_"
    file_name = start_tag+file.split("\\")[-1]
    copyfile(file, os.path.join(target, file_name))
    print("train file down")
    
target = "data/vulner/validate_data/raw"
for index, file in enumerate(X_test):
    start_tag = "Vulner_" if y_test[index] == 1 else "NotVulner_"
    file_name = start_tag+file.split("\\")[-1]
    copyfile(file, os.path.join(target, file_name))
    print("validate file down")


# x_train移动到train_data
# x_test 移动到validate_data