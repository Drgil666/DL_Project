import numpy as np


def load_npz_and_get_first_row(npz_file):
    # 加载npz文件
    data = np.load(npz_file)

    # 获取npz文件中的所有项
    items = data.files

    # 打印出所有项
    print(f"Items in {npz_file}: {items}")

    # 返回第一个数组的第一行数据
    first_array = data[items[0]]
    first_row = first_array[0]

    return first_row


# 示例用法
npz_file = 'H:\\DL\\app\\DeepOnet\\dataset\\data_train_25.npz'  # 替换为你的npz文件路径
first_row_data = load_npz_and_get_first_row(npz_file)
print(f"First row of data: {first_row_data}")
