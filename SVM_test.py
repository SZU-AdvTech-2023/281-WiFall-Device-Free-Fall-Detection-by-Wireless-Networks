import os
import csv
import numpy as np
from scipy.stats import iqr, entropy

row_data = [] #原始数据
row_integers = [] #X雏形
row_strings = [] #Y雏形
i = 0

# 文件夹路径
folder_path = 'D:\Wifall\Test'
try:
    # 获取文件夹中所有csv文件的列表
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # 遍历文件列表
    for file_name in file_list:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 打开CSV文件并读取内容
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)

            # 读取CSV文件内容
            header = None
            second_row_integers = None
            second_row_strings = None

            for row in csv_reader:
                if header is None:
                    # 保存第一行作为标题（列名）
                    header = row
                else:
                    # 保存第二行的前28个数据为整数数组
                    second_row_integers = [int(cell) for cell in row[:28]]
                    # 保存第二行的后14个数据为字符串数组
                    second_row_strings = str(row[28:])
                    break  # 只需要读取第二行
            row_integers.append(second_row_integers)
            row_strings.append(second_row_strings)
            # 打印标题、第二行整数数据和第二行字符串数据
            #print("标题:", header)
           # print("第二行的前28个整数数据:", second_row_integers)
            #print("第二行的后14个字符串数据:", second_row_strings)

except FileNotFoundError:
    print(f"文件夹 '{folder_path}' 未找到。")
except Exception as e:
    print(f"发生错误: {str(e)}")

# 文件夹路径
folder_path = 'D:\Wifall\pca'
try:
    # 获取文件夹中所有csv文件的列表
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # 遍历文件列表
    for file_name in file_list:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 打开CSV文件并读取内容
        with open(file_path, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            pca_array = []
            for row in csv_reader:
                # 假设文件只有一列数据
                pca_array.append(float(row[0]))  # 转换为合适的数据类型
            row_data.append(pca_array)

except FileNotFoundError:
    print(f"文件夹 '{folder_path}' 未找到。")
except Exception as e:
    print(f"发生错误: {str(e)}")

#打印数组长度，长度都为11
#print(len(row_data[0]))

#开始记录输入神经网络的数组
X_data = []
Y_data = []

ns = []
so = []
TT= []
mad = []
qr = []
se = []
scr = []

#从已获取的数组中提取数据
for i in range(11):
    for j in range(14):
        Start = j
        End = j + 14

        #print("row_data length:", len(row_data))
        #print("Start index:", row_integers[Start])
        #print("End index:", row_integers[End])

        # 选择要分析的时间段
        segment = np.array(row_data[i][row_integers[i][Start]:(row_integers[i][End])])
        # 归一化标准差
        normalized_std = np.std(segment, axis=0) / np.mean(segment, axis=0)
        ns.append(normalized_std)
        # 信号强度偏移
        signal_offset = np.mean(segment, axis=0)
        so.append(signal_offset)
        # 运动周期
        TT.append(row_integers[i][End]-row_integers[i][Start])
        # 中位数绝对偏差
        median_absolute_deviation = np.median(np.abs(segment - np.median(segment, axis=0)), axis=0)
        mad.append(median_absolute_deviation)
        # 四分位间距
        quartile_range = iqr(segment, axis=0)
        qr.append(quartile_range)
        # 信号熵
        signal_entropy = np.apply_along_axis(entropy, axis=0, arr=segment.T)
        se.append(signal_entropy)
        # 信号变化速度
        signal_change_rate = np.diff(segment, axis=0)

        # 对信号变化速度进行进一步处理
        # 例如，计算其标准差、平均值等
        std_signal_change_rate = np.std(signal_change_rate, axis=0)
        scr.append(std_signal_change_rate)
        #mean_signal_change_rate = np.mean(signal_change_rate, axis=0)

#处理信号熵
se_test = np.array(se)
# 将无穷大替换为一个较大的有限值
se_test[np.isinf(se_test)] = np.nanmax(np.abs(se_test[np.isfinite(se_test)])) * 1e10
valid_indices = np.where(~np.isinf(se_test))
min_val = np.min(se_test[valid_indices])
max_val = np.max(se_test[valid_indices])
se_try = []
se_try = (se_test - min_val) / (max_val - min_val)

# 遍历label列表
for item in row_strings:
    # 使用eval函数将字符串列表转换为实际的列表对象
    inner_list = eval(item)
    # 遍历内部列表
    for action in inner_list:
        if action == 'Walk':
            Y_data.append(1)
        if action == 'Fall':
            Y_data.append(2)
        elif action == 'Stand':
            Y_data.append(3)
        elif action == 'Sit':
            Y_data.append(4)


#print(len(ns))
#print(so)
#print(TT)
#print(mad)
#print(qr)
#print(se)
#print(scr)
#print(len(Y_data))



Y_Label = np.array(Y_data)

# 假设你希望将这些数组写入一个名为 'output.csv' 的 CSV 文件
output_filename = 'svm_output.csv'

# 将数组组合成一个列表
arrays_to_write = [ns, so, TT, mad, qr, se_try, scr, Y_Label]

# 使用 'with' 语句打开文件并将数组写入 CSV 文件
with open(output_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 计算最大数组的长度（假设所有数组长度一样）
    max_length = max(len(arr) for arr in arrays_to_write)

    # 写入数据到 CSV 文件
    for i in range(max_length):
        row = [arr[i] if i < len(arr) else '' for arr in arrays_to_write]
        writer.writerow(row)

print(f"Arrays written to {output_filename} successfully.")





