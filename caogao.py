import kagglehub
import csv
import os

# Download latest version
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")

print("Path to dataset files:", path)

csv_file_path = os.path.join(path, "diabetes.csv")

x_list = []
y_list = []
with open(csv_file_path, 'r') as csvfile:
    # 创建CSV读取器
    csvreader = csv.reader(csvfile)

    # 读取标题行
    header = next(csvreader)

    # 遍历CSV文件中的每一行
    for row in csvreader:
        x = []
        for i in range(len(row[:-1])):
            x.append(float(row[i]))
        x_list.append(x)
        y_list.append(float(row[-1]))            

print(x_list)
print(y_list)

