'''输入输出数据是50s，采样时间为0.01，,5000个点，时间是100s，采样时间是0.001,100001个点'''
import json
import os

'''测试数据代码'''
import torch
import matplotlib.pyplot as plt
from ..pinn_mionet.metrics import *
from ..pinn_mionet.model.model1 import PINN_MIONet
from app.pinn_mionet.config import Config
import numpy as np

# from scipy.integrate import cumtrapz

config = Config()

length_size = 3500
t_length_size = 35000


def show_pinn_mionet_test(data_path,model_path):
    global l2_error_x,mse_x,mae_x,l2_error_y,mse_y,mae_y,l2_error_z,mse_z,mae_z
    print('------test start--------')
    model = PINN_MIONet(config)

    train_data = np.load(data_path,allow_pickle=True)
    branch_input = train_data['X']
    branch_input_tensor = torch.tensor(branch_input,dtype=torch.float32)

    # input_coe = torch.tensor([50000, 65000, 42000], dtype=torch.float32)
    # branch_input_tensor = branch_input_tensor* input_coe

    branch_input_1 = branch_input_tensor[:,:3500,0]
    branch_input_2 = branch_input_tensor[:,:3500,1]
    branch_input_3 = branch_input_tensor[:,:3500,2]

    # a4 =branch_input_tensor.numpy()
    # df4 = pd.DataFrame(a4).T
    # df4.to_excel('Tc8000_1.xlsx', index=False, header=False)

    train_t = train_data['t'].astype(np.float32)
    len_train_t = len(train_t)
    print(len_train_t)
    trunk_input_temp = train_t.reshape(len_train_t,1)
    trunk_input_1 = trunk_input_temp[:t_length_size,:]
    trunk_input_2 = trunk_input_1[::10,:]
    trunk_input = torch.tensor(trunk_input_2,dtype=torch.float32)

    label_train = train_data['Y'][:,:,3:6]
    label_train_temp = label_train[:,:3500,:]
    label_train_tensor = torch.tensor(label_train_temp,dtype=torch.float32)

    model_path1 = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(model_path1)

    trunk_input = trunk_input.clone().requires_grad_(True)
    model.eval()
    with torch.no_grad():
        output_Yita_one,output_Yita_two,output_Yita_three = \
            model(branch_input_1,
                  branch_input_2,
                  branch_input_3,
                  trunk_input)

    pre_value_x = output_Yita_one[:,:]/10
    pre_value_y = output_Yita_two[:,:]/210
    pre_value_z = output_Yita_three[:,:]/140

    labels_test_temp = label_train_tensor.cpu().detach().numpy()
    output_Yita_one_temp = output_Yita_one.cpu().detach().numpy()
    output_Yita_two_temp = output_Yita_two.cpu().detach().numpy()
    output_Yita_three_temp = output_Yita_three.cpu().detach().numpy()
    test_nums = labels_test_temp.shape[0]
    print("test nums: ",test_nums)
    # TODO:暂时硬编码到10
    test_nums = 10
    result = []
    for i in range(0,test_nums):
        print('-----------------------------------',i)

        print(max(branch_input_1[i,:]))
        r2_1 = R2(labels_test_temp[i,:,0],output_Yita_one_temp[i,:]/10)
        r2_2 = R2(labels_test_temp[i,:,1],output_Yita_two_temp[i,:]/210)
        r2_3 = R2(labels_test_temp[i,:,2],output_Yita_three_temp[i,:]/140)
        print('-----r2_1------',r2_1)
        print('-----r2_2------',r2_2)
        print('-----r2_3------',r2_3)

        l2_error_x = l2_relative_error(label_train_tensor[i,:,0],pre_value_x[i,:])
        mse_x = mean_squared_error(label_train_tensor[i,:,0],pre_value_x[i,:])
        mae_x = MAE(label_train_tensor[i,:,0],pre_value_x[i,:])

        l2_error_y = l2_relative_error(label_train_tensor[i,:,1],pre_value_y[i,:])
        mse_y = mean_squared_error(label_train_tensor[i,:,1],pre_value_y[i,:])
        mae_y = MAE(label_train_tensor[i,:,1],pre_value_y[i,:])

        l2_error_z = l2_relative_error(label_train_tensor[i,:,2],pre_value_z[i,:])
        mse_z = mean_squared_error(label_train_tensor[i,:,2],pre_value_z[i,:])
        mae_z = MAE(label_train_tensor[i,:,2],pre_value_z[i,:])
        data = {"l2_error_x":float(l2_error_x),
                "mse_x":float(mse_x),
                "mae_x":float(mae_x),
                "l2_error_y":float(l2_error_y),
                "mse_y":float(mse_y),
                "mae_y":float(mae_y),
                "l2_error_z":float(l2_error_z),
                "mse_z":float(mse_z),
                "mae_z":float(mae_z),
                }
        result.append(data)
        print('-----111111111------',l2_error_x,mse_x,mae_x)
        print('-----22222222------',l2_error_y,mse_y,mae_y)
        print('-----33333333------',l2_error_z,mse_z,mae_z)

        plt.figure(figsize=(8,15))
        plt.suptitle(f'Comparison of Actual vs Predicted Values - Sample {i}',y=1.02)

        plt.subplot(4,1,1)
        plt.plot(branch_input_1[i,:],label=f'Input Tc',color='blue',linestyle='-',linewidth=1)
        plt.title('Input')
        plt.legend()
        plt.grid(True)

        # plt.subplot(6, 1, 2)
        # plt.plot(branch_input_2[i, :], label=f'Input Tc', color='blue', linestyle='-', linewidth=1)
        # plt.title('Input')
        # plt.legend()
        # plt.grid(True)
        #
        # plt.subplot(6, 1, 3)
        # plt.plot(branch_input_3[i, :], label=f'Input Tc', color='blue', linestyle='-', linewidth=1)
        # plt.title('Input')
        # plt.legend()
        # plt.grid(True)

        # 子图1: X分量
        plt.subplot(4,1,2)
        plt.plot(labels_test_temp[i,:,0],label=f'Actual yita',color='blue',linewidth=1)
        plt.plot(output_Yita_one_temp[i,:]/10,label=f'Predicted yita',color='red',linewidth=1)
        plt.title('Output1')
        plt.legend()
        plt.grid(True)

        # 子图2: Y分量
        plt.subplot(4,1,3)
        plt.plot(labels_test_temp[i,:,1],label='Actual yita',color='blue',linewidth=1)
        plt.plot(output_Yita_two_temp[i,:]/210,label='Predicted yita',color='red',linewidth=1)
        plt.title('Output2')
        plt.legend()
        plt.grid(True)

        # 子图3: Z分量
        plt.subplot(4,1,4)
        plt.plot(labels_test_temp[i,:,2],label='Actual yita',color='blue',linewidth=1)
        plt.plot(output_Yita_three_temp[i,:]/140,label='Predicted yita',color='red',linewidth=1)
        plt.title('Output3')
        plt.legend()
        plt.grid(True)

        # 调整布局防止重叠
        plt.tight_layout()
        plt.savefig(os.path.join('app/static/img/'+'deep_png'+str(i)+'.png'))
        # 显示图形
        # plt.show()
        plt.close()

    return result,test_nums
