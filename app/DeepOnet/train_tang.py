'''柔性翻板数据(唐工)'''
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from django.http import HttpRequest
from torch import optim

from app.DeepOnet.deeponet import Model
from app.DeepOnet.metrics import *

re = HttpRequest()


def show1(re,y):
    device = torch.device("cuda:0")
    res = ""
    y = int(y)
    seed = 12
    np.random.seed(seed)
    torch.manual_seed(seed)

    dt = 0.001
    min_t = 0
    max_t = 3
    a = np.arange(min_t,max_t,dt)
    lt = a.shape[0]

    # data_train = np.load('dataset/data_train_25.npz',allow_pickle=True)
    #
    # branch_input = data_train['X'].astype(np.float32)
    # branch_input = branch_input[:,0:lt:5]
    # t = data_train['t'].astype(np.float32)
    # len_t = len(t)
    # trunk_input = t.reshape(len_t,1)
    # trunk_input = trunk_input[:lt:5]
    #
    # labels_train = data_train['Y'].astype(np.float32)
    # labels_train = torch.tensor(labels_train)
    # labels_train = labels_train[:,0:lt:5]
    #
    model = Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # #
    num_epochs = 30000

    # start_time_train = time.time()
    # loss_epoches = []
    # for epoch in range(num_epochs):
    #     model.train()
    #     optimizer.zero_grad()
    #     outputs = model(branch_input,trunk_input)
    #     loss = criterion(outputs,labels_train)
    #     loss.backward()
    #     optimizer.step()
    #     if (epoch + 1) % 1000 == 0:
    #         loss_epoches.append(loss.item())
    #         print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, num_epochs, loss.item()))
    #
    # #compute train time
    # end_time_train = time.time()
    # train_time = end_time_train-start_time_train
    # print("Total train time: {:.4f} second".format(train_time))
    # print('epoch', loss_epoches)

    # # # #save model
    # model_path = os.path.join('model_save\\deeponet_25_3s_0.005_lt_%s.pth'%(lt))
    # torch.save(model.state_dict(),model_path)
    # load model
    model_path1 = 'H:\\DL\\app\\DeepOnet\\model_save\\deeponet_25_3s_0.005_lt_3000.pth'
    model.load_state_dict(torch.load(model_path1))

    model.eval()

    data_test = np.load('H:\\DL\\app\\upload_testFile\\data_test_7.npz',allow_pickle=True)
    t = data_test['t'].astype(np.float32)
    len_t = len(t)
    trunk_input_test = t.reshape(len_t,1)
    trunk_input_test = trunk_input_test[0:lt:5]

    branch_input_test = data_test['X'].astype(np.float32)
    branch_input_test = branch_input_test[y,0:lt:5]

    labels_test = data_test['Y'].astype(np.float32)
    labels_test = torch.tensor(labels_test)
    labels_test = labels_test[y,0:lt:5]
    # print("branch_input_test.shape",len(branch_input_test))

    start_time_test = time.time()

    with torch.no_grad():
        outputs_test = model(branch_input_test,trunk_input_test)
    end_time_test = time.time()
    test_time = end_time_test-start_time_test

    # print("Total test time: {:.4f} second".format(test_time))
    # res += f"Total test time: {test_time:.4f} seconds\n"

    l2_error = l2_relative_error(labels_test,outputs_test)
    mse = mean_squared_error(labels_test,outputs_test)

    data = [
        ["预测所需时间(秒)","{:.8f} ".format(test_time)],
        ["L2相对误差","{:.8f}".format(l2_error)],
        ["均方误差","{:.8f}".format(mse)]
    ]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
    plt.rcParams['axes.unicode_minus'] = False
    # 可选：将表格保存为图片
    fig,ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=data,cellLoc='center',loc='center')
    table.auto_set_font_size(False)  # 关闭自动设置字体大小
    table.set_fontsize(14)  # 设置字体大小为14
    table.scale(1.2,1.5)  # 调整表格的行距和列距

    filename = f'H:\\DL\\app\\static\\img\\fitting_table_metrics{y}.png'
    # 保存为图片
    plt.savefig(filename,bbox_inches='tight')
    plt.close()

    # print('l2_error is',l2_error)
    # print('mse is',mse)

    # res += f"l2_error is:{l2_error}\n"
    # res += f"mse is:{mse}\n"

    # save results
    # results = pd.DataFrame(outputs_test.numpy())
    # np.savez('deeponet_results.npz',Onet_result = results)

    # save error
    # errors = np.abs(labels_test - outputs_test)
    # error_df = pd.DataFrame(errors.numpy())
    #
    # np.savez('deeponet_error.npz',Onet_error =error_df )

    # # 绘制真实值和预测值的对比图
    # x_data = np.arange(0, 3, 0.005)
    # plt.figure(figsize=(10, 6))
    # plt.plot(x_data, labels_test[y, :], label='True')
    # plt.plot(x_data, outputs_test[y, :], label='Predicted')
    # plt.title('True vs Predicted Values')
    # plt.xlabel('Time(s)')
    # plt.ylabel('Disturbance torque')
    # plt.legend()
    # plt.grid(True)
    #
    # # plt.show()

    x_data = np.arange(0,3,0.005)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 推荐使用SimHei字体显示中文
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15,20))

    # plt.subplot(2, 1, 1)
    plt.plot(x_data,branch_input_test[:],color='r',label='系统输入')
    # plt.title('input data')
    plt.xlabel('时间(s)')
    plt.ylabel('角加速度')
    plt.legend(fontsize='18')
    plt.grid(True)

    filename = f'H:\\DL\\app\\static\\img\\fitting_1_png{y}.png'
    # 保存为图片
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(15,20))
    # plt.subplot(2, 1, 2)
    plt.plot(x_data,labels_test[:],label='系统真实响应值')
    plt.plot(x_data,outputs_test[:],label='预测响应值')
    plt.xlabel('时间(s)')
    plt.ylabel('干扰力矩')
    plt.legend(fontsize='18')
    plt.grid(True)

    filename = f'H:\\DL\\app\\static\\img\\fitting_2_png{y}.png'
    # 保存为图片
    plt.savefig(filename)
    plt.close()
    # plt.show()
    # 将图像保存为字节流
    # buffer = BytesIO()
    # plt.savefig(buffer, format='png')
    # buffer.seek(0)
    # plt.close()
    #
    # # 将字节流转换为base64编码
    # image_png = buffer.getvalue()
    # buffer.close()
    # image_str = base64.b64encode(image_png).decode('utf-8')
    # return image_str,res


if __name__ == '__main__':
    show1(1)
