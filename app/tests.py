import tensorflow as tf

#
csv_file_path = 'example.csv'
# 列名和数据列表
data = []
# 使用pandas加载CSV文件

df = pd.read_csv(csv_file_path)
column_names = df.columns[:-1]
y = 'y'
# 分离参数部分
parameters = [col.split('*')[0] for col in column_names]
variables = [col.split('*')[1] for col in column_names]
print(parameters)
print(variables)
Y_TRAIN = df[y]
X_TRAIN = df.drop(columns=[y],axis=1)
input_shape = X_TRAIN.shape[1:]
print(Y_TRAIN)
formula = y+' = '
for i in range(len(parameters)):
    formula += (parameters[i]+' * '+variables[i])
    if i < len(parameters)-1:
        formula += ' + '
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1,input_shape=input_shape)  # 输入层，接收 x(t) 和 u(t)
])

# 编译模型
model.compile(optimizer='adam',loss='mean_squared_error')

# 训练模型

model.fit(X_TRAIN[:10],Y_TRAIN[:10],epochs=50000)

# 获取模型参数
weights,biases = model.layers[0].get_weights()
output = y+' = '
for i in range(len(weights)):
    output += (str(weights[i])[1:-1]+' * '+variables[i])
    if i < len(weights)-1:
        output += ' + '
fina_parameters = ''
for i in range(len(weights)):
    fina_parameters += (parameters[i]+': '+str(weights[i])[1:-1])
    if i < len(weights)-1:
        fina_parameters += ', '

print(weights)
print(formula)
print(output)
print(fina_parameters)

#
#
# with open("C:\\Users\\92476\\Desktop\\data.csv", 'rb') as file:
#     # 将文件内容读取为字节序列
#     csv_data = file.read()
#
# # 将字节序列解码为 UTF-8 编码的字符串
# csv_data_string = csv_data.decode('utf-8')
#
# # 使用 StringIO 创建文件类对象供 Pandas 读取
# csv_data_io = io.StringIO(csv_data_string)
#
# # 从文件类对象中读取 CSV 数据并加载到 Pandas DataFrame 中
# df = pd.read_csv(csv_data_io)
# CNN = keras.models.load_model("C:\\Users\\92476\\Downloads\\CNN1D_test.keras")
# LSTM = keras.models.load_model("C:\\Users\\92476\\Downloads\\LSTM_test.keras")
# CON = keras.models.load_model("C:\\Users\\92476\\Downloads\\concatenate_CNN1D_LSTM.keras")
#
# class WeightedSumLayer(tf.keras.layers.Layer):
#     def __init__(self, custom_weights, **kwargs):
#         super(WeightedSumLayer, self).__init__(**kwargs)
#         self.custom_weights = custom_weights
#
#     def call(self, inputs):
#         weighted_sum = tf.zeros_like(inputs[0])  # 初始化加权和为零向量
#         for i, input_tensor in enumerate(inputs):
#             weighted_sum += self.custom_weights[i] * input_tensor
#         return weighted_sum
#
#     def get_config(self):
#         config = super(WeightedSumLayer, self).get_config()
#         config.update({
#             'custom_weights': self.custom_weights
#         })
#         return config
# custom_objects = {'WeightedSumLayer': WeightedSumLayer}
# VOTE = keras.models.load_model("C:\\Users\\92476\\Downloads\\vote_CNN1D_LSTM.keras", custom_objects=custom_objects)
# label_encoder = preprocessing.LabelEncoder()
# df.dropna(axis=1, how='all', inplace=True)
# df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])
# df['diagnosis'].unique()
# y = df['diagnosis']
# X = df.drop(columns=['diagnosis'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train, y_train)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# CNN_y_pred = (CNN.predict(X_test) > 0.5).astype("int32")
# LSTM_y_pred = (LSTM.predict(X_test) > 0.5).astype("int32")
# CON_y_pred = (CON.predict(X_test) > 0.5).astype("int32")
# VOTE_y_pred = (VOTE.predict(X_test) > 0.5).astype("int32")
# cnn_p = CNN.predict(X_test)
# lstm_p = LSTM.predict(X_test)
# con_p = CON.predict(X_test)
# vote_p = VOTE.predict(X_test)
# # print("accuracy")
# # print(accuracy_score(y_test, y_pred))
# #
# # print("recall")
# # CNN_recall = recall_score(y_test, CNN_y_pred)
# # LSTM_recall = recall_score(y_test, LSTM_y_pred)
# # CON_recall = recall_score(y_test, CON_y_pred)
# # VOTE_recall = recall_score(y_test, VOTE_y_pred)
# # print(CNN_recall)
# # print(LSTM_recall)
# # print(CON_recall)
# # print(VOTE_recall)
#
# # CNN 模型的评价指标
# CNN_accuracy = accuracy_score(y_test, CNN_y_pred)
# CNN_precision = precision_score(y_test, CNN_y_pred)
# CNN_recall = recall_score(y_test, CNN_y_pred)
# CNN_f1 = f1_score(y_test, CNN_y_pred)
# CNN_auc = roc_auc_score(y_test, cnn_p)
# CNN_conf_matrix = confusion_matrix(y_test, CNN_y_pred)
# CNN_FP = CNN_conf_matrix[0,1]
# CNN_TN = CNN_conf_matrix[0,0]
# CNN_FPR = CNN_FP / (CNN_FP + CNN_TN)
#
# # LSTM 模型的评价指标
# LSTM_accuracy = accuracy_score(y_test, LSTM_y_pred)
# LSTM_precision = precision_score(y_test, LSTM_y_pred)
# LSTM_recall = recall_score(y_test, LSTM_y_pred)
# LSTM_f1 = f1_score(y_test, LSTM_y_pred)
# LSTM_auc = roc_auc_score(y_test, lstm_p)
# LSTM_conf_matrix = confusion_matrix(y_test, LSTM_y_pred)
# LSTM_FP = LSTM_conf_matrix[0,1]
# LSTM_TN = LSTM_conf_matrix[0,0]
# LSTM_FPR = LSTM_FP / (LSTM_FP + LSTM_TN)
#
# # CON 模型的评价指标
# CON_accuracy = accuracy_score(y_test, CON_y_pred)
# CON_precision = precision_score(y_test, CON_y_pred)
# CON_recall = recall_score(y_test, CON_y_pred)
# CON_f1 = f1_score(y_test, CON_y_pred)
# CON_auc = roc_auc_score(y_test, con_p)
# CON_conf_matrix = confusion_matrix(y_test, CON_y_pred)
# CON_FP = CON_conf_matrix[0,1]
# CON_TN = CON_conf_matrix[0,0]
# CON_FPR = CON_FP / (CON_FP + CON_TN)
#
# # VOTE 模型的评价指标
# VOTE_accuracy = accuracy_score(y_test, VOTE_y_pred)
# VOTE_precision = precision_score(y_test, VOTE_y_pred)
# VOTE_recall = recall_score(y_test, VOTE_y_pred)
# VOTE_f1 = f1_score(y_test, VOTE_y_pred)
# VOTE_auc = roc_auc_score(y_test, vote_p)
# VOTE_conf_matrix = confusion_matrix(y_test, VOTE_y_pred)
# VOTE_FP = VOTE_conf_matrix[0,1]
# VOTE_TN = VOTE_conf_matrix[0,0]
# VOTE_FPR = VOTE_FP / (VOTE_FP + VOTE_TN)
#
# # 输出评价指标
# print("CNN模型:")
# print("准确率 (Accuracy): {:.4f}".format(CNN_accuracy))
# print("精确率 (Precision): {:.4f}".format(CNN_precision))
# print("召回率 (Recall): {:.4f}".format(CNN_recall))
# print("F1 分数 (F1 Score): {:.4f}".format(CNN_f1))
# print("AUC: {:.4f}".format(CNN_auc))
# print("假正例率 (False Positive Rate): {:.4f}".format(CNN_FPR))
#
# print("\nLSTM模型:")
# print("准确率 (Accuracy): {:.4f}".format(LSTM_accuracy))
# print("精确率 (Precision): {:.4f}".format(LSTM_precision))
# print("召回率 (Recall): {:.4f}".format(LSTM_recall))
# print("F1 分数 (F1 Score): {:.4f}".format(LSTM_f1))
# print("AUC: {:.4f}".format(LSTM_auc))
# print("假正例率 (False Positive Rate): {:.4f}".format(LSTM_FPR))
#
# print("\nCON模型:")
# print("准确率 (Accuracy): {:.4f}".format(CON_accuracy))
# print("精确率 (Precision): {:.4f}".format(CON_precision))
# print("召回率 (Recall): {:.4f}".format(CON_recall))
# print("F1 分数 (F1 Score): {:.4f}".format(CON_f1))
# print("AUC: {:.4f}".format(CON_auc))
# print("假正例率 (False Positive Rate): {:.4f}".format(CON_FPR))
#
# print("\nVOTE模型:")
# print("准确率 (Accuracy): {:.4f}".format(VOTE_accuracy))
# print("精确率 (Precision): {:.4f}".format(VOTE_precision))
# print("召回率 (Recall): {:.4f}".format(VOTE_recall))
# print("F1 分数 (F1 Score): {:.4f}".format(VOTE_f1))
# print("AUC: {:.4f}".format(VOTE_auc))
# print("假正例率 (False Positive Rate): {:.4f}".format(VOTE_FPR))
