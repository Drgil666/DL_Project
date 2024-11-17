import gc
import io
import os
import pickle
import time
import zipfile
from io import BytesIO

import cv2
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf
from django.contrib.auth import authenticate,login
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.http import JsonResponse,HttpResponse
from django.shortcuts import render,redirect,get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from fuzzywuzzy import fuzz
from keras import Model
from keras import Sequential
from keras.applications import MobileNet,ResNet50,VGG16,VGG19,InceptionV3,MobileNetV2,DenseNet121,DenseNet169, \
    DenseNet201
from keras.layers import (Flatten,Dense,Dropout,BatchNormalization,Conv1D,Conv2D,GlobalAveragePooling1D,
                          GlobalAveragePooling2D,LSTM,MultiHeadAttention,
                          LayerNormalization,
                          Input,Activation,Add,GRU,Reshape,Concatenate)
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from app.DeepOnet.train_tang import show1
from app.Operator_network.Models.deeponet import DeepOnet
from app.Operator_network.Models.mionet import MIONet
from app.models import RegistrationForm,CSVFile,MODELFile,IMGFile

# global data
df = pd.DataFrame()
img_path = ''
epoch_g = 0
epoch_input = 0
optimizer = 0
l_r = 0
n_classes = 0
n_layers = 0
n_filters = 0
k_size = 0
step = 0
d_r = 0
l_a = ''
d_a = ''
split = 0.0
model_name = ''
path = ''
logs_g = []
estimate_time = 0
accuracy_g = {}
loss_g = {}
val_accuracy_g = {}
val_loss_g = {}
accuracy_score_g = 0
label = ''
label1 = ''
label2 = ''
data_type = 0
model_selection = 0
classes = []
class_weights = {}
login_history = 0
username = ''
output_formula = ''
formula = ''
final_parameters = ''
weight_log = []
model_introduction = {
    "CNN 1D":
        "\nCNN 1D:"
        "\n网络类型: 卷积神经网络"
        "\n处理数据类型: 文本数据"
        "\n耗时: 耗时较短"
        "\n描述: 一维的的卷积神经网络（CNN）是一种较为经典和简单的用于处理一维序列型数据（如文本数据）的神经网络模型。它通过卷积操作来提取文本数据中的局部特征，并通过池化操作来减少特征维度，最终利用全连接层进行分类或回归等任务。在任务类型方面，一维文本数据的CNN适用于多种自然语言处理（NLP）任务，例如文本分类、文本匹配、命名实体识别（NER）、序列标注等。这些任务涵盖了许多实际场景，如情感分析、问答系统、医疗记录分析、金融新闻情感分析等。在领域方面，一维文本数据的CNN广泛应用于自然语言处理领域。它在社交媒体分析、新闻媒体分析、医疗领域和金融领域等方面都有着重要作用。例如，在社交媒体分析中，它可以用于分析用户的情感倾向和话题热度；在医疗领域，它可以用于分析医疗记录中的症状描述和诊断结果。针对训练和评估一维文本数据CNN，有许多常见的数据集可供使用。这些数据集包括IMDB电影评论数据集、20 Newsgroups数据集、Yelp商家评价数据集和CoNLL 2003数据集等。这些数据集覆盖了不同的任务和语言，为模型的训练和评估提供了丰富的资源。，一维文本数据的卷积神经网络在自然语言处理领域具有广泛的应用，并且适用于多种任务类型，有着丰富的数据集可供选择。"
    ,
    "LSTM":
        "\nLSTM : "
        "\n网络类型: 长短期记忆网络"
        "\n数据类型: 时序数据、序列数据、文本数据、音频数据、视频数据 "
        "\n耗时: 相对较长，特别是在处理长序列数据时 "
        "\n描述: 长短期记忆网络（LSTM）是一种常用于处理序列数据的循环神经网络（RNN）变体。它通过引入门控机制来解决传统RNN中的梯度消失和梯度爆炸等问题，使得模型能够更好地捕捉长期依赖关系。LSTM的核心结构包括一个记忆单元（memory cell）和三个门：遗忘门（forget gate）、输入门（input gate）和输出门（output gate）。这些门控制着信息的流动，允许模型在处理序列数据时选择性地保留或遗忘信息，从而更好地适应不同任务的需求。在任务类型方面，LSTM广泛应用于处理序列数据的各种任务，包括但不限于文本生成、语言建模、机器翻译、情感分析、时间序列预测等。由于其能够有效地捕捉长期依赖关系，LSTM在处理这些任务时通常能够取得较好的性能。在领域方面，LSTM被广泛应用于自然语言处理（NLP）、语音识别、视频分析、时间序列分析等领域。例如，在NLP领域，LSTM可以用于文本生成、命名实体识别、句法分析等任务；在语音识别领域，LSTM可以用于语音识别、语音情感识别等任务。针对训练和评估LSTM模型，通常使用各种数据集，如Penn Treebank数据集、IMDB电影评论数据集、TIMIT语音数据集等。这些数据集涵盖了不同的任务和领域，为LSTM模型的训练和评估提供了丰富的资源。"
    ,
    "Transformer":
        "\nTransformer: "
        "\n网络类型: 序列到序列网络，Seq2Seq"
        "\n数据类型: 文本数据、序列数据"
        "\n耗时: 较长，特别是在大规模文本数据上"
        "\n描述: Transformer是一种革命性的神经网络架构，用于处理序列数据，特别是自然语言处理（NLP）领域。相比传统的循环神经网络（RNN）和长短期记忆网络（LSTM），Transformer引入了注意力机制，通过自注意力机制来捕捉序列中各个位置之间的依赖关系，避免了传统RNN和LSTM中的循环结构，从而加速了训练并提高了模型性能。Transformer的核心结构包括编码器（Encoder）和解码器（Decoder）。编码器负责将输入序列映射到隐藏表示空间，而解码器则在给定编码器的隐藏表示的情况下生成输出序列。每个编码器和解码器均由多个相同的层堆叠而成，每一层都由多头自注意力机制和前馈神经网络组成。在任务类型方面，Transformer广泛应用于各种NLP任务，包括但不限于机器翻译、文本生成、命名实体识别、问答系统等。由于其能够同时处理长距离依赖关系和并行计算，Transformer在处理这些任务时通常能够取得较好的性能。在领域方面，Transformer被广泛应用于自然语言处理领域。例如，在机器翻译领域，Transformer模型如Google的“Transformer”和OpenAI的GPT（Generative Pre-trained Transformer）系列都取得了显著的进展；在文本生成领域，GPT模型能够生成高质量的文本，具有很强的应用潜力。针对训练和评估Transformer模型，通常使用各种大型语料库，如WMT机器翻译数据集、COCO图像描述数据集等。这些数据集涵盖了不同的任务和语言，为Transformer模型的训练和评估提供了丰富的资源。综上所述，Transformer作为一种革命性的神经网络架构，在NLP领域具有广泛的应用。它通过引入注意力机制，能够有效地捕捉序列中的长距离依赖关系，适用于各种NLP任务，并且有着丰富的数据集可供选择。"
    ,
    "GRU":
        "\nGRU"
        "\n描述: 门控循环单元模型"
        "\n数据类型: 时序数据、序列数据"
        "\n参数类型: 序列数据"
        "\n耗时: 相对较短，训练速度较快"
        "\n可调参数: 隐藏层维度, 学习率, 循环次数, 等"
        "\n通用参数: 学习率, 优化器类型, 批量大小, 等"
        "\n最佳层数: 1 到 3 层"
    ,
    "BidirectionalLSTMModel":
        "\nBidirectionalLSTM: "
        "\n描述: 双向长短期记忆网络"
        "\n数据类型: 时序数据、序列数据"
        "\n参数类型: 序列数据"
        "\n耗时: 相对较长，特别是在处理长序列数据时"
        "\n可调参数: 隐藏层维度, 学习率, 循环次数, 等"
        "\n通用参数: 学习率, 优化器类型, 批量大小, 等"
        "\n最佳层数: 1 到 3 层"
    ,
    "tcnModel":
        "\nTCN: "
        "\n描述: 时间卷积网络"
        "\n数据类型: 时序数据、序列数据"
        "\n参数类型: 序列数据"
        "\n耗时: 相对较短，训练速度较快"
        "\n可调参数: 卷积核大小, 步长, 层数, 通道数, 等"
        "\n通用参数: 学习率, 优化器类型, 批量大小, 等"
        "\n最佳层数: 3 到 10 层"
    ,
    "ResNet 1D":
        "\nResNet 1D: "
        "\n网络类型: 残差网络"
        "\n数据类型: 文本数据"
        "\n耗时: 较长，特别是在复杂网络结构和大规模数据集上"
        "\n描述: 一维ResNet是Residual Network（残差网络）在处理一维序列数据时的应用。ResNet是一种深度神经网络架构，通过引入残差块（residual blocks）来解决深层网络训练中的梯度消失和梯度爆炸等问题，从而使得能够训练更深的网络模型。在一维ResNet中，残差块被设计为处理一维序列数据的特征提取和表示学习。一维ResNet的核心结构由多个残差块堆叠而成，每个残差块通常由卷积层、批量归一化层和激活函数构成。在每个残差块中，输入数据会通过一条“跳跃连接”（skip connection）直接传递到残差块的输出端，与残差块内部的特征学习过程相结合。这种设计能够有效地减轻了梯度消失问题，使得网络更易于训练，并且能够有效地学习到数据的高级特征。一维ResNet广泛应用于处理时序数据，如时间序列、信号处理、语音识别等领域。它在时序数据建模任务中通常能够取得较好的性能，例如对于序列分类、序列预测等任务。在自然语言处理领域，一维ResNet也被用于文本分类、情感分析等任务中，通常在处理较长的文本序列时能够取得良好的效果。总之，一维ResNet是一种用于处理一维序列数据的深度神经网络模型，通过引入残差块解决了深层网络训练中的梯度问题，广泛应用于时序数据处理和自然语言处理等领域。"

}


def index(request):
    if request.method == 'POST':
        global login_history,username
        username = request.POST.get('username')
        print(username)
        password = request.POST.get('password')
        print(password)
        user = authenticate(request,username=username,password=password)
        print('1')
        if user is not None:
            print('2')
            login(request,user)
            login_history = 1
            return redirect('navigation')  # Change 'upload' to your upload page URL name
        else:
            print('3')
            error = 'Invalid username or password'
            return render(request,'index.html',{'error':error})
    else:
        return render(request,'index.html')


def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        print(form.is_valid())
        if form.is_valid():
            username = form.cleaned_data['username']
            email = form.cleaned_data['email']
            password = form.cleaned_data['password1']
            user = User.objects.create_user(username=username,email=email,password=password)
            user.save()
            return redirect('index')  # 返回登录界面的 URL 名称
    else:
        form = RegistrationForm()
    return render(request,'register.html',{'form':form})


# 定义一个空字典，用于存储分词后的结果
segmented_model_introduction = {}


def tokenize_zh(text):
    stopwords_file = "hit_stopwords.txt"
    stopwords = set()
    with open(stopwords_file,"r",encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip())

    text = text.replace("\n","")
    segmented_text = [word for word in jieba.lcut_for_search(text) if word.strip() and word not in stopwords]
    return segmented_text


# 对每个模型的介绍文本进行分词
for model,introduction in model_introduction.items():
    # 使用 tokenize_zh 函数对介绍文本进行分词
    segmented_text = tokenize_zh(introduction)
    # 将分词结果存储到新的字典中
    segmented_model_introduction[model] = segmented_text

# 打印分词结果
for model,segmented_text in segmented_model_introduction.items():
    print(f"模型: {model}")
    print("分词结果:",segmented_text)
    print()
print('0')
with open('keybert_model.pkl','rb') as f:
    kw_model = pickle.load(f)


def keyword(text):
    keywords = kw_model.extract_keywords(text,vectorizer=CountVectorizer(tokenizer=tokenize_zh,token_pattern=None))
    print(keywords)
    return keywords


def process_uploaded_csv(csv_file):
    global df

    # Read the contents of the InMemoryUploadedFile as bytes
    csv_data = csv_file.read()

    # Convert bytes to string using UTF-8 encoding
    csv_data_string = csv_data.decode('utf-8')

    # Use StringIO to create a file-like object for Pandas to read from
    csv_data_io = io.StringIO(csv_data_string)

    # Load CSV data into a Pandas DataFrame
    df = pd.read_csv(csv_data_io)

    # Now you can process the DataFrame as needed
    return df


def process_uploaded_zip(zip_file):
    zip_content = zip_file.read()
    # 创建一个内存中的字节流对象
    zip_file = BytesIO(zip_content)

    # 解压缩文件
    with zipfile.ZipFile(zip_file,'r') as zip_ref:
        # 假设解压后的文件保存在/tmp目录下
        extract_dir = '/app/zip_temp/'
        zip_ref.extractall(extract_dir)
        # 构建解压后的文件列表
        root_dir = zip_ref.namelist()[0].split('/')[0]
        extracted_files = os.path.join(extract_dir,root_dir)
        # 返回解压后的文件路径列表
        return extracted_files


def process_images_grey(image_folder,img_label,img_size):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder,filename),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
            img = np.dstack([img,img,img])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(img_label)
    return images,labels


def process_images_normal(image_folder,img_label,img_size):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder,filename))
        if img is not None:
            img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
            img = np.dstack([img,img,img])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(img_label)
    return images,labels


def img_process(p,label_1,label_2):
    global classes,class_weights,split
    path1 = os.path.normpath(os.path.join(p,label_1))
    path2 = os.path.normpath(os.path.join(p,label_2))
    print(path1,path2)
    class1_images = []
    class2_images = []
    class1_labels = []
    class2_labels = []
    is_grey = 0
    img_size = 224

    for filename in os.listdir(path1):
        img = cv2.imread(os.path.join(path1,filename))
        if len(img.shape) == 2:
            is_grey = 0
        elif len(img.shape) == 3:
            is_grey = 1
        break
    print(is_grey)

    if is_grey == 0:
        class1_images,class1_labels = process_images_normal(path1,label_1,img_size)
        class2_images,class2_labels = process_images_normal(path2,label_2,img_size)

    elif is_grey == 1:
        class1_images,class1_labels = process_images_grey(path1,label_1,img_size)
        class2_images,class2_labels = process_images_grey(path2,label_2,img_size)

    class_images = class1_images+class2_images
    class_labels = class1_labels+class2_labels

    label_encoder = preprocessing.LabelEncoder()
    class_labels = label_encoder.fit_transform(class_labels)
    classes = label_encoder.classes_

    print(class_labels.shape)
    X = class_images
    y = class_labels
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y),y=y)
    class_weights = dict(enumerate(class_weights))
    print(class_weights)
    X = np.array(X)
    y = np.array(y)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split,random_state=42)
    input_shape = X_train.shape[-3:]

    # 释放内存
    del class1_images
    del class2_images
    del class1_labels
    del class2_labels
    del X
    del y
    gc.collect()

    return X_train,X_test,y_train,y_test,input_shape


def img_load(p):
    global classes,class_weights,split

    is_grey = 0
    img_size = 224

    for filename in os.listdir(p):
        img = cv2.imread(os.path.join(p,filename))
        if len(img.shape) == 2:
            is_grey = 0
        elif len(img.shape) == 3:
            is_grey = 1
        break
    print(is_grey)
    images = []
    if is_grey == 0:
        for filename in os.listdir(p):
            img = cv2.imread(os.path.join(p,filename))
            if img is not None:
                img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
                img = np.dstack([img,img,img])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                images.append(img)

    elif is_grey == 1:
        for filename in os.listdir(p):
            img = cv2.imread(os.path.join(p,filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
                img = np.dstack([img,img,img])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                images.append(img)

    X = images
    X = np.array(X)
    input_shape = X.shape[-3:]
    # 释放内存
    gc.collect()

    return X,input_shape


def csv_process(dataframe):
    global label,classes,split
    label_encoder = preprocessing.LabelEncoder()
    dataframe.dropna(axis=1,how='all',inplace=True)
    dataframe[label] = label_encoder.fit_transform(dataframe[label])
    classes = label_encoder.classes_
    dataframe[label].unique()
    y = dataframe[label]
    X = dataframe.drop(columns=[label],axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=split,random_state=42)
    print(X_train,y_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
    input_shape = X_train.shape[1:]
    print(X_train,y_train)
    return X_train,X_test,y_train,y_test,input_shape


def csv_load(dataframe):
    X = dataframe
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0],X.shape[1],1))
    input_shape = X.shape[1:]
    return X,input_shape


def model_coordinate(data):
    # x是数据类型（文本 -1/图像 1）
    # y是任务类型（分类 -1/预测 1）
    # z是时间消耗（慢 -1~快 1）
    model_rank = {}
    vectors = {
        'GRU':np.array([1,1]),
        'Transformer':np.array([-1,1]),
        'CNN 1D':np.array([-1,-1]),
        'LSTM':np.array([-1,-0.5]),
    }
    for name,vector in vectors.items():
        distance = np.sqrt(np.sum((vector-data)**2))
        model_rank.update({name:distance})

    sorted_rank = sorted(model_rank,key=model_rank.get)
    top_three_keys = sorted_rank[:3]
    return top_three_keys


def model_choice(model_id,input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                 layer_activation,dense_activation):
    # 定义模型函数的字典
    model_functions = {
        0:CNN,
        1:LsTM,
        2:Transformer,
        # 3: GRUModel,
        # 4: BidirectionalLSTMModel,
        # 5: tcnModel,
        # 6: Resnet,
        # 7: Mobilenet,
        # 8: CNN_2D,
        # 9: TimeDistributedDenseModel,
        # 10: Resnet50,
        # 11: vgg16,
        # 12: vgg19,
        # 13: inception_v3,
        # 14: mobilenet_v2,
        # 15: densenet_121,
        # 16: densenet_169,
        # 17: densenet_201
        3:deeponet,
        4:mionet
    }

    # 根据 model_id 调用相应的模型函数
    if model_id in model_functions:
        if model_id == 3:
            print("进来了")
        model_function = model_functions[model_id]
        if model_function == 0:
            # CNN
            return CNN(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                       layer_activation,dense_activation)
        elif model_function == 1:
            # LSTM
            return LsTM(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                        layer_activation,dense_activation)
        elif model_function == 2:
            # Transformers
            return Transformer(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                               layer_activation,dense_activation)
        elif model_function == 3:
            # deeponet
            return deeponet(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                            layer_activation,dense_activation)
        elif model_function == 4:
            # mionet
            return mionet(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                          layer_activation,dense_activation)
    # TODO:需要再这里补充模型选择的功能
    else:
        raise ValueError("Invalid model_id: {}".format(model_id))


def model_compile(X_train,X_test,y_train,y_test,model,optimizer_input,l_r_input,model_n):
    global epoch_g,epoch_input,data_type,class_weights,path

    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.start_time = None


        def on_epoch_begin(self,epoch,logs=None):
            self.start_time = time.time()


        def on_epoch_end(self,epoch,logs=None):
            global epoch_g
            global logs_g
            global estimate_time
            all_log = {'epoch':epoch+1}
            all_log.update(logs)
            logs_g.append(all_log)
            epoch_g = epoch+1
            end_time = time.time()
            estimate_time = (end_time-self.start_time) * (epoch_input-epoch_g)
            print(logs)

    if optimizer_input == 0:
        model.compile(optimizer=Adam(learning_rate=l_r_input),loss='mean_squared_error',metrics=['accuracy'])
    callback = MyCallback()
    if data_type == 1:
        history = model.fit(X_train,y_train,epochs=epoch_input,validation_data=(X_test,y_test),verbose=1,
                            callbacks=[callback])
    elif data_type == 2:
        batch_size = 16
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size)
        test_generator = test_datagen.flow(X_test,y_test,batch_size=batch_size)
        history = model.fit(train_generator,epochs=epoch_input,validation_data=test_generator,
                            verbose=1,
                            callbacks=[callback],class_weight=class_weights)
    elif data_type == 3:
        # TODO:这里的逻辑也需要修改
        model_n = "deeponet_zhouqi_5s_0.005.pth"
        path = os.path.join('app/model_temp/',model_n)
        model.save(path)
        return model,path
    model_n = f"{model_n}.keras"
    path = os.path.join('app/model_temp/',model_n)
    model.save(path)

    return history,model,y_test,X_test,path


def download_model(request):
    # 模型文件路径
    global path,model_name
    model_path = path
    model_n = f"{model_name}.keras"
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return HttpResponse("模型文件不存在",status=404)

    # 打开模型文件并读取内容
    with open(model_path,'rb') as f:
        model_content = f.read()

    # 构建HTTP响应
    response = HttpResponse(model_content,content_type='application/octet-stream')

    # 设置响应头，指示浏览器下载文件
    response['Content-Disposition'] = f'attachment; filename="{model_n}"'
    return response


def deeponet(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
             dense_activation):
    # TODO:此处需要补充细节
    return DeepOnet()


def mionet(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
           dense_activation):
    # TODO:此处需要补充细节
    return MIONet()


def Mobilenet(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
              dense_activation):
    base_model = MobileNet(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="Mobilenet")

    return model


def Resnet50(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
             dense_activation):
    base_model = ResNet50(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="ResNet50")

    return model


def vgg16(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
          dense_activation):
    base_model = VGG16(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="VGG16")

    return model


def vgg19(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
          dense_activation):
    base_model = VGG19(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="VGG19")

    return model


def inception_v3(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
                 dense_activation):
    base_model = InceptionV3(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="InceptionV3")

    return model


def mobilenet_v2(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
                 dense_activation):
    base_model = MobileNetV2(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="MobileNetV2")

    return model


def densenet_121(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
                 dense_activation):
    base_model = DenseNet121(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="DenseNet121")

    return model


def densenet_169(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
                 dense_activation):
    base_model = DenseNet169(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="DenseNet169")

    return model


def densenet_201(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
                 dense_activation):
    base_model = DenseNet201(weights=None,include_top=False,input_shape=input_shape)

    x = base_model.output

    # Add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Add a logistic layer
    predictions = Dense(1,activation="sigmoid")(x)

    model = Model(inputs=base_model.input,outputs=predictions,name="DenseNet201")

    return model


def CNN_2D(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
           dense_activation):
    model = Sequential(name="CNN_2D")
    model.add(Input(shape=input_shape,name='CNN2D'))

    for i in range(num_layers):
        model.add(Conv2D(filters=num_filters,kernel_size=kernel_size,strides=strides,activation=layer_activation,
                         name='cnnconv2d'+str(i)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(64,activation=layer_activation,name='cnn2ddense1'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes,activation=dense_activation,name='cnn2ddense2'))

    return model


def CNN(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
        dense_activation):
    model = Sequential(name="CNN_1d")
    model.add(Input(shape=input_shape,name='CNN1D'))

    for i in range(num_layers):
        model.add(Conv1D(filters=num_filters,kernel_size=kernel_size,strides=strides,activation=layer_activation,
                         name='cnnconv1d'+str(i)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(64,activation=layer_activation,name='cnndense1'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes,activation=dense_activation,name='cnndense2'))

    return model


def LsTM(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
         dense_activation):
    model = Sequential(name="LSTM")
    model.add(Input(shape=input_shape,name='LSTM'))

    for i in range(num_layers-1):
        model.add(LSTM(units=num_filters,activation=layer_activation,return_sequences=True,name='lstm'+str(i)))
        # 增加 return_sequences=True，使得每个 LSTM 层都返回完整的序列
        model.add(Dropout(dropout_rate))

    model.add(LSTM(units=num_filters,activation=layer_activation,name='lstmfina'))
    # 最后一个 LSTM 层不需要返回完整序列
    model.add(Dense(num_classes,activation=dense_activation,name='lstmdense'))

    return model


def GRUModel(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
             dense_activation):
    model = Sequential(name="GRU")
    model.add(Input(shape=input_shape,name='GRU'))

    for i in range(num_layers-1):
        model.add(GRU(units=num_filters,activation=layer_activation,
                      return_sequences=True))  # 增加 return_sequences=True，使得每个 GRU 层都返回完整的序列
        model.add(Dropout(dropout_rate))

    model.add(GRU(units=num_filters,activation=layer_activation))  # 最后一个 GRU 层不需要返回完整序列
    model.add(Dense(num_classes,activation=dense_activation))

    return model


def BidirectionalLSTMModel(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                           layer_activation,dense_activation):
    model = Sequential([Input(shape=input_shape,name='BidirectionalLSTM')],name="BidirectionalLSTM")
    # 添加多层 LSTM
    for i in range(num_layers):
        model.add(tf.keras.layers.Bidirectional(
            LSTM(units=num_filters,activation=layer_activation)))
        model.add(Dropout(dropout_rate))
        model.add(Reshape((1,128)))
    model.add(Flatten())
    # 添加最后的全连接层
    model.add(Dense(num_classes,activation=dense_activation))

    return model


def tcnModel(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
             dense_activation):
    model = Sequential(
        [Input(shape=input_shape,name='TCN'),Conv1D(filters=num_filters,kernel_size=kernel_size,strides=strides,
                                                    activation=layer_activation)],name="TCN")

    for i in range(num_layers-1):
        model.add(
            tf.keras.layers.Conv1D(filters=num_filters,kernel_size=kernel_size,strides=strides,
                                   activation=layer_activation,padding='causal'))
        model.add(Dropout(dropout_rate))

    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(num_classes,activation=dense_activation))

    return model


def TimeDistributedDenseModel(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,
                              layer_activation,dense_activation):
    model = tf.keras.models.Sequential(name="TimeDistributedDense")

    # 添加输入层
    model.add(Input(shape=input_shape,name='TimeDistributedDense'))

    # 添加 TimeDistributed 密集层
    for i in range(num_layers):
        model.add(tf.keras.layers.TimeDistributed(Dense(num_filters,activation=layer_activation)))
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # 添加全局池化层或展平层
    model.add(tf.keras.layers.GlobalAveragePooling1D())  # 也可以使用 GlobalMaxPooling1D 或 Flatten

    # 添加输出层
    model.add(tf.keras.layers.Dense(num_classes,activation=dense_activation))

    return model


def Transformer(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
                dense_activation):
    num_heads = 2
    embed_dim = 32
    ff_dim = 64
    inputs = Input(shape=input_shape,name='Transformer')
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    for i in range(num_layers):
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(num_heads=num_heads,key_dim=embed_dim)(x,x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(inputs+attn_output)

        # Feed Forward Network
        ffn_output = Dense(ff_dim,activation=layer_activation)(out1)
        ffn_output = Dense(embed_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(out1+ffn_output)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes,activation=dense_activation)(x)

    model = tf.keras.Model(inputs=inputs,outputs=outputs,name="Transformer")
    return model


def Resnet(input_shape,num_layers,num_classes,num_filters,kernel_size,strides,dropout_rate,layer_activation,
           dense_activation):
    num_blocks = 4
    inputs = Input(shape=input_shape,name='Resnet')
    x = Conv1D(filters=num_filters,kernel_size=kernel_size,strides=strides,padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(layer_activation)(x)

    for i in range(num_blocks):
        # Shortcut connection
        shortcut = x

        # Residual block
        x = Conv1D(filters=num_filters,kernel_size=kernel_size,strides=strides,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(layer_activation)(x)
        x = Conv1D(filters=num_filters,kernel_size=kernel_size,strides=strides,padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([shortcut,x])
        x = Activation(layer_activation)(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes,activation=dense_activation)(x)

    model = tf.keras.Model(inputs=inputs,outputs=outputs,name="Resnet")
    return model


def upload(request):
    if request.method == 'POST':
        global df,label,label1,label2,model_selection,data_type,img_path
        file = request.FILES['data_File']
        label = request.POST.get('label')
        label1 = request.POST.get('label1')
        label2 = request.POST.get('label2')
        print(label)
        model_selection = int(request.POST.get('model_selection'))
        data_type = request.POST.get('data_type')
        data_type = int(data_type)
        print(data_type)
        user = request.user
        file_name = file.name
        if data_type == -1:
            df = process_uploaded_csv(file)
            csv_file_obj = CSVFile.objects.create(user=user,file=file,file_name=file_name)
            print(df)
        elif data_type == 1:
            zip_file_obj = IMGFile.objects.create(user=user,file=file,file_name=file_name)
            img_path = process_uploaded_zip(file)
        current_user = request.user
        print(current_user)
        print(current_user.is_authenticated)
        return redirect('/train/')
    return render(request,'upload.html')


@csrf_exempt
def model_recommendation(request):
    global data_type
    if request.method == 'POST':
        data_type = request.POST.get('data_type')
        data_type = float(data_type)
        time_demand = request.POST.get('time_demand')
        time_demand = float(time_demand)
        data_info = np.array([data_type,time_demand])
        top_3 = model_coordinate(data_info)
        r_t = []
        print(top_3)
        b_i = "根据您的数据类型和时间需求，推荐以下三个模型: "+top_3[0]+", "+top_3[1]+" 和 "+top_3[2]
        e_i = "如果您还想了解和使用别的模型，可以在以下输入框中输入您的需求，系统将会按照您的描述推荐匹配的模型"
        print(top_3[0])
        print(type(top_3[0]))
        print(top_3[1])
        print(model_introduction['CNN 1D'])
        m_1 = model_introduction[top_3[0]]
        m_2 = model_introduction[top_3[1]]
        m_3 = model_introduction[top_3[2]]
        m_r = b_i+'\n'+m_1+'\n'+m_2+'\n'+m_3+'\n\n'+e_i
        data = {"model_recommendation":m_r}
        return JsonResponse(data)


@csrf_exempt
def get_text(request):
    if request.method == 'POST':
        text = request.POST['text']
        print(text)
        global model_introduction
        k_w = keyword(text)
        # 计算每个模型与输入语句的相似度
        similarities = {}
        for model,segmented_text in segmented_model_introduction.items():
            # 将模型的分词结果与关键词列表进行比较，并计算相似度得分
            similarity_score = 0
            for k_word,score in k_w:
                for word in segmented_text:
                    similarity_score += fuzz.token_sort_ratio(k_word,word) * score
            similarities[model] = round(similarity_score,2)
        # 根据相似度得分对模型进行排序
        sorted_models = sorted(similarities,key=lambda x:similarities[x],reverse=True)
        # 返回排名最高的模型作为推荐结果
        top_model = sorted_models[0]
        second_model = sorted_models[1]
        third_model = sorted_models[2]
        score_all = similarities[top_model]+similarities[second_model]+similarities[third_model]
        b_i = '根据您的描述，推荐以下三个模型'
        f_i = f"\n\n推荐模型：{top_model}，相似度得分：{similarities[top_model]}"+model_introduction[top_model]
        s_i = f"\n\n推荐模型：{second_model}，相似度得分：{similarities[second_model]}"+model_introduction[second_model]
        t_i = f"\n\n推荐模型：{third_model}，相似度得分：{similarities[third_model]}"+model_introduction[third_model]
        m_i = b_i+f_i+s_i+t_i
        print(m_i)
        if score_all != 0:
            text_output = m_i
        else:
            text_output = '抱歉，暂时无法理解您的描述，请重新输入。'
        data = {"text_output":text_output}
        return JsonResponse(data)


@csrf_exempt
def train(request):
    if request.method == 'POST' and request.POST.get('id') == '0':
        global df,classes,img_path,label1,label2,data_type,model_name
        global accuracy_g,loss_g,val_accuracy_g,val_loss_g,accuracy_score_g
        global optimizer,l_r
        if data_type == -1:
            X_train,X_test,y_train,y_test,input_shape = csv_process(df)
        elif data_type == 1:
            X_train,X_test,y_train,y_test,input_shape = img_process(img_path,label1,label2)
        # elif data_type == 3:
        model = model_choice(model_selection,input_shape,n_layers,n_classes,n_filters,k_size,step,
                             d_r,l_a,d_a)
        print(model.summary())
        history,model,y_test,X_test,model_path = model_compile(X_train,X_test,y_train,y_test,model,optimizer,
                                                               l_r,model_name)

        user = request.user
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            return HttpResponse("模型文件不存在",status=404)
        # 打开模型文件并读取内容
        with open(model_path,'rb') as f:
            model_content = f.read()
        file_name = os.path.basename(model_path)
        model_file_obj = MODELFile.objects.create(user=user,file=ContentFile(model_content,name=file_name),
                                                  file_name=file_name,classes=classes)
        accuracy_g = history.history['accuracy']
        print(accuracy_g)
        loss_g = history.history['loss']
        print(loss_g)
        val_accuracy_g = history.history['val_accuracy']
        print(val_accuracy_g)
        val_loss_g = history.history['val_loss']
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(y_pred)
        y_pred_class = [classes[i[0]] for i in y_pred]
        print(y_pred_class)
        accuracy_score_g = accuracy_score(y_test,y_pred)
        print(accuracy_score_g)
        print(classes)
        del model
        del history
        del X_train,X_test,y_train,y_test,input_shape
        del model_content
        gc.collect()
        return HttpResponse("TRAIN COMPLETED")
    else:
        return render(request,'train.html')


@csrf_exempt
def load(request):
    return render(request,'load.html')


def show(request):
    return render(request,'show.html')


@csrf_exempt
def get_data(request):
    if request.method == 'POST':
        global epoch_g,epoch_input
        global estimate_time
        global logs_g
        rounded_time = round(estimate_time,2)
        data = {"epoch":epoch_g,"time":[rounded_time],"log":logs_g}
        return JsonResponse(data)


@csrf_exempt
def get_parameter(request):
    if request.method == 'POST':
        global epoch_input,optimizer,l_r,n_classes,n_layers,n_filters,k_size,step,d_r,l_a,d_a,model_name,split
        epoch_input = int(request.POST['epoch'])
        optimizer = int(request.POST['optimizer'])
        l_r = float(request.POST['l_r'])
        n_classes = int(request.POST['num_classes'])-1
        n_layers = int(request.POST['num_layers'])
        n_filters = int(request.POST['num_filters'])
        k_size = int(request.POST['kernel_size'])
        step = int(request.POST['strides'])
        d_r = float(request.POST['dropout_rate'])
        l_a = request.POST['layer_activation']
        d_a = request.POST['dense_activation']
        model_name = request.POST['model_name']
        split = 1-float(request.POST['split']) / 100.0
        data = {"epoch":epoch_input}
        return JsonResponse(data)


@csrf_exempt
def model_data(request):
    if request.method == 'POST':
        global epoch_g
        global accuracy_g
        global loss_g
        global val_accuracy_g
        global val_loss_g
        global accuracy_score_g
        epoch_range = list(range(1,epoch_g+1))
        print(epoch_range)
        print(accuracy_g)
        data = {"accuracy":accuracy_g,"val_accuracy":val_accuracy_g,"loss":val_loss_g,
                "val_loss":val_loss_g,"accuracy_score":accuracy_score_g,"epoch":epoch_range}
        return JsonResponse(data)


@csrf_exempt
def get_result(request):
    if request.method == 'POST':
        global df,label,classes,data_type,img_path
        user = request.user
        file = request.FILES.get('data_File')
        model_file = request.FILES.get('model_File')
        data_type = int(request.POST['data_type'])
        if data_type == -1:
            df = process_uploaded_csv(file)
            X,input_shape = csv_load(df)
        elif data_type == 1:
            img_path = process_uploaded_zip(file)
            X,input_shape = img_load(img_path)
        model_n = model_file.name
        model_file_path = os.path.join('app/model_temp/',model_n)
        # 将上传的模型文件保存到临时路径
        with open(model_file_path,'wb+') as destination:
            for chunk in model_file.chunks():
                destination.write(chunk)
        loaded_model = keras.models.load_model(model_file_path)

        print(loaded_model.summary())
        y_pred = (loaded_model.predict(X) > 0.5).astype("int32")
        m_obj = MODELFile.objects.get(user=user,file_name=model_n)
        class_l = m_obj.classes
        class_l = class_l[1:-1].replace("'","").strip()
        class_l = class_l.split()
        print(class_l)
        results = [class_l[i[0]] for i in y_pred]
        print(results)
        data = {"result":results}
        return JsonResponse(data)


class WeightedSumLayer(tf.keras.layers.Layer):
    def __init__(self,custom_weights):
        super(WeightedSumLayer,self).__init__()
        self.custom_weights = custom_weights


    def call(self,inputs):
        weighted_sum = tf.zeros_like(inputs[0])  # 初始化加权和为零向量
        for i,input_tensor in enumerate(inputs):
            weighted_sum += self.custom_weights[i] * input_tensor
        return weighted_sum


    def get_config(self):
        config = super().get_config()
        config.update({
            'custom_weights':self.custom_weights
        })
        return config


@csrf_exempt
def ensemble(request):
    if request.method == 'POST':
        user = request.user
        # 获取上传的文件
        global df,label,split,data_type,path,model_name,label1,label2,img_path
        global accuracy_g,loss_g,val_accuracy_g,val_loss_g,accuracy_score_g
        split = 0.2
        file = request.FILES.get('data_File')

        model_name = request.POST['model_name']
        data_type = int(request.POST['data_type'])
        ensemble_type = int(request.POST['ensemble_type'])
        uploaded_files = request.FILES.getlist('models')  # 获取名为 'models' 的文件列表

        if data_type == -1:
            label = request.POST['label']
            df = process_uploaded_csv(file)
            X_train,X_test,y_train,y_test,input_shape = csv_process(df)

        elif data_type == 1:
            label1 = request.POST['label1']
            label2 = request.POST['label2']
            img_path = process_uploaded_zip(file)
            X_train,X_test,y_train,y_test,input_shape = img_process(img_path,label1,label2)

        i = 1
        model_inputs = []
        if ensemble_type == 0:
            # 加载模型并构建输入输出
            for uploaded_file in uploaded_files:
                model_n = uploaded_file.name
                model_file_path = os.path.join('app/model_temp/',model_n)

                # 将上传的模型文件保存到临时路径
                with open(model_file_path,'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                loaded_model = tf.keras.models.load_model(model_file_path)
                if i == 1:
                    model1 = loaded_model
                else:
                    model2 = loaded_model
                i += 1

            input_shape = model1.input.shape
            input = Input(shape=input_shape[1:])

            output1_layer = model1.layers[-2]
            new_model1 = Model(inputs=model1.input,outputs=output1_layer.output)
            output2_layer = model2.layers[-2]
            new_model2 = Model(inputs=model2.input,outputs=output2_layer.output)
            model_1 = new_model1(input)
            model_2 = new_model2(input)
            output = Concatenate()([model_1,model_2])
            predictions = Dense(1,activation="sigmoid")(output)
            model = Model(inputs=input,outputs=predictions,name=model_name)
            model.summary()
            model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

        elif ensemble_type == 1:
            global weight_log
            num = request.POST['num']  # 获取名为 'num' 的表单数
            weight = []
            weight1 = float(request.POST['file1']) / 100.0
            weight2 = float(request.POST['file2']) / 100.0
            weight3 = request.POST['file3']
            weight4 = request.POST['file4']
            weight5 = request.POST['file5']
            weight.append(weight1)
            weight.append(weight2)
            if weight3:
                weight3 = float(weight3) / 100.0
                weight.append(weight3)
            if weight4:
                weight4 = float(weight4) / 100.0
                weight.append(weight4)
            if weight5:
                weight5 = float(weight5) / 100.0
                weight.append(weight5)
            print(weight)

            print(num)

            # 保存上传的文件到特定目录
            # 创建模型输入

            model_outputs = []
            models_list = []

            # 加载模型并构建输入输出
            for uploaded_file in uploaded_files:
                model_n = uploaded_file.name
                model_file_path = os.path.join('app/model_temp/',model_n)

                # 将上传的模型文件保存到临时路径
                with open(model_file_path,'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)

                loaded_model = tf.keras.models.load_model(model_file_path)
                models_list.append(loaded_model)
                loaded_model.summary()

            input = Input(shape=models_list[0].input_shape[1:])
            for i in range(len(models_list)):
                model_outputs.append(models_list[i](input))
            new_weight = weight.copy()
            weight_log = []
            weight_log.append(weight)
            print(weight_log)

            for i in range(len(X_train)):
                model_pred = []
                X_temp = np.expand_dims(X_train[i],axis=0)
                for model_temp in models_list:
                    model_pred.append((model_temp.predict(X_temp) > 0.5).astype("int32"))
                m = 0
                for k in range(len(weight)):
                    if model_pred[k] != y_train.iloc[i]:
                        m += 1
                for j in range(len(weight)):
                    if model_pred[j] != y_train.iloc[i]:
                        t = weight[j] / len(X_train)
                        new_weight[j] -= t
                        for k in range(len(weight)):
                            if k != j:
                                new_weight[k] += t / m
                                print(weight)
                print(new_weight)
                print(weight)
                print(weight_log)
                weight_log.append(new_weight.copy())
            print(weight_log)
            print(len(weight_log))
            # 创建加权和输出
            weighted_sum_output = WeightedSumLayer(custom_weights=weight)(model_outputs)

            # 构建模型
            model = Model(inputs=input,outputs=weighted_sum_output)
            model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

        class MyCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.start_time = None


            def on_epoch_begin(self,epoch,logs=None):
                self.start_time = time.time()


            def on_epoch_end(self,epoch,logs=None):
                global epoch_g
                global logs_g
                global estimate_time
                all_log = {'epoch':epoch+1}
                all_log.update(logs)
                logs_g.append(all_log)
                epoch_g = epoch+1
                end_time = time.time()
                estimate_time = (end_time-self.start_time) * (100-epoch_g)
                print(logs)

        callback = MyCallback()
        if data_type == -1:
            history = model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test),verbose=1,
                                callbacks=[callback])
        elif data_type == 1:
            batch_size = 16
            train_datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            test_datagen = ImageDataGenerator(rescale=1. / 255)
            train_generator = train_datagen.flow(X_train,y_train,batch_size=batch_size)
            test_generator = test_datagen.flow(X_test,y_test,batch_size=batch_size)
            history = model.fit(train_generator,epochs=epoch_input,validation_data=test_generator,
                                verbose=1,
                                callbacks=[callback],class_weight=class_weights)
        accuracy_g = history.history['accuracy']
        loss_g = history.history['loss']
        val_accuracy_g = history.history['val_accuracy']
        val_loss_g = history.history['val_loss']
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(y_pred)
        print(accuracy_g)
        accuracy_score_g = accuracy_score(y_test,y_pred)
        model_name_temp = f"{model_name}.keras"
        m_obj = MODELFile.objects.get(user=user,file_name=model_n)
        class_l = m_obj.classes
        path = os.path.join('app/model_temp/',model_name_temp)
        model.save(path)
        with open(path,'rb') as f:
            model_content = f.read()
        file_name = os.path.basename(path)
        model_file_obj = MODELFile.objects.create(user=user,file=ContentFile(model_content,name=file_name),
                                                  file_name=file_name,classes=class_l)

        return JsonResponse({'message':'Upload successful'},status=200)
    else:
        return render(request,'ensemble.html')


@csrf_exempt
def parameter_identification(request):
    if request.method == 'POST':
        global df,formula,output_formula,final_parameters
        file = request.FILES['data_File']
        user = request.user
        file_name = file.name
        y = request.POST['label']
        df = process_uploaded_csv(file)
        csv_file_obj = CSVFile.objects.create(user=user,file=file,file_name=file_name)
        column_names = df.columns[:-1]
        # 分离参数部分
        parameters = [col.split('*')[0] for col in column_names]
        variables = [col.split('*')[1] for col in column_names]
        # 列名和数据列表
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

        class MyCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                self.start_time = None


            def on_epoch_begin(self,epoch,logs=None):
                self.start_time = time.time()


            def on_epoch_end(self,epoch,logs=None):
                global epoch_g
                global estimate_time
                epoch_g = epoch+1
                end_time = time.time()
                estimate_time = (end_time-self.start_time) * (100-epoch_g)
                print(logs)

        callback = MyCallback()
        history = model.fit(X_TRAIN,Y_TRAIN,epochs=100,callbacks=[callback])
        # 获取模型参数
        weights,biases = model.layers[0].get_weights()
        weights = np.around(weights,decimals=2)

        print(weights)
        output_formula = y+' = '
        for i in range(len(weights)):
            output_formula += (str(weights[i])[1:-1]+' * '+variables[i])
            if i < len(weights)-1:
                output_formula += ' + '
        final_parameters = ''
        for i in range(len(weights)):
            final_parameters += (parameters[i]+': '+str(weights[i])[1:-1])
            if i < len(weights)-1:
                final_parameters += ', '
        return JsonResponse({'message':'Upload successful'},status=200)
    else:
        return render(request,'parameter.html')


@csrf_exempt
def parameter_identification_getdata(request):
    if request.method == 'POST':
        global formula,output_formula,final_parameters
        global epoch_g,epoch_input
        global estimate_time
        global logs_g
        rounded_time = round(estimate_time,2)
        data = {"epoch":epoch_g,"time":[rounded_time],"formula":formula,
                "output_formula":output_formula,"final_parameters":final_parameters}
        return JsonResponse(data)


@csrf_exempt
def motivation_system_fitting(request):
    if request.method == 'POST':
        return HttpResponse('motivation_system_fitting')
    else:
        return render(request,'fitting.html')


def motivation_system_getdata(request):
    if request.method == 'POST' and request.FILES['data_File']:
        data_file = request.FILES['data_File']
        y = request.POST['label']
        # 处理上传的文件，例如保存到服务器的某个地方
        # 这里假设你保存到某个特定的路径
        with open('H:\\DL\\app\\upload_testFile\\data_test_7.npz','wb+') as destination:
            for chunk in data_file.chunks():
                destination.write(chunk)
    show1(request,y)
    # return JsonResponse({'plot_image': image_str,'result':result})
    return HttpResponse('motivation_system_fitting')


def get_file_icon(file_name):
    if file_name.endswith('.csv'):
        return 'bi bi-file-earmark-text'
    elif file_name.endswith('.keras'):
        return 'bi bi-file-earmark-spreadsheet'
    else:
        return 'bi bi-file-earmark-binary'


def self(request):
    # 获取当前登录的用户
    global login_history
    current_user = request.user
    if not current_user.is_authenticated and login_history == 0:
        print(current_user)
        print('1')
        return redirect('index')  # 如果用户未登录，则重定向到登录页面
    elif not current_user.is_authenticated and login_history == 1:
        print(current_user)
        print('1')
        return redirect('warning')  # 如果用户未登录，则重定向到登录页面
    # 获取当前用户上传的 CSV 文件和 Model 文件
    csv_files = CSVFile.objects.filter(user=current_user)
    model_files = MODELFile.objects.filter(user=current_user)
    img_files = IMGFile.objects.filter(user=current_user)

    # 计算文件大小并将其添加到文件对象中
    for file_obj in csv_files:
        file_obj.file_size = os.path.getsize(file_obj.file.path)
        file_obj.save()
        file_obj.icon_class = get_file_icon(file_obj.file.name)
    for file_obj in model_files:
        file_obj.file_size = os.path.getsize(file_obj.file.path)
        file_obj.save()
        file_obj.icon_class = get_file_icon(file_obj.file.name)
    for file_obj in img_files:
        file_obj.file_size = os.path.getsize(file_obj.file.path)
        file_obj.save()
        file_obj.icon_class = get_file_icon(file_obj.file.name)

    csv_files = reversed(csv_files)
    model_files = reversed(model_files)
    img_files = reversed(img_files)

    # 从数据库中检索所有文件对象
    return render(request,'self.html',
                  {'csv_files':csv_files,'model_files':model_files,'img_files':img_files})


def download_csv(request,file_id):
    # 通过文件ID获取文件对象
    file = get_object_or_404(CSVFile,pk=file_id)

    # 打开文件并将其内容读取到HttpResponse中
    with open(file.file.path,'rb') as f:
        response = HttpResponse(f.read(),content_type='application/octet-stream')

    # 设置响应的文件名（可选）
    response['Content-Disposition'] = f'attachment; filename="{file.file_name}"'

    return response


def download_keras(request,file_id):
    # 通过文件ID获取文件对象
    file = get_object_or_404(MODELFile,pk=file_id)
    print(file.file.path)
    # 打开文件并将其内容读取到HttpResponse中
    with open(file.file.path,'rb') as f:
        response = HttpResponse(f.read(),content_type='application/octet-stream')

    # 设置响应的文件名（可选）
    response['Content-Disposition'] = f'attachment; filename="{file.file_name}"'

    return response


def download_img(request,file_id):
    # 通过文件ID获取文件对象
    file = get_object_or_404(IMGFile,pk=file_id)

    # 打开文件并将其内容读取到HttpResponse中
    with open(file.file.path,'rb') as f:
        response = HttpResponse(f.read(),content_type='application/octet-stream')

    # 设置响应的文件名（可选）
    response['Content-Disposition'] = f'attachment; filename="{file.file_name}"'

    return response


def warning(request):
    return render(request,'warning.html')


def navigation(request):
    global df,img_path,epoch_g,epoch_input,optimizer,l_r,n_classes,n_layers,n_filters,k_size,step,d_r,l_a,d_a,split,model_name,path,logs_g,estimate_time
    global accuracy_g,loss_g,val_accuracy_g,val_loss_g,accuracy_score_g,label,label1,label2,data_type,model_selection,classes,class_weights
    global output_formula,formula,final_parameters,weight_log
    df = pd.DataFrame()
    img_path = ''
    epoch_g = 0
    epoch_input = 0
    optimizer = 0
    l_r = 0
    n_classes = 0
    n_layers = 0
    n_filters = 0
    k_size = 0
    step = 0
    d_r = 0
    l_a = ''
    d_a = ''
    split = 0.0
    model_name = ''
    path = ''
    logs_g = []
    estimate_time = 0
    accuracy_g = {}
    loss_g = {}
    val_accuracy_g = {}
    val_loss_g = {}
    accuracy_score_g = 0
    label = ''
    label1 = ''
    label2 = ''
    data_type = 0
    model_selection = 0
    classes = []
    class_weights = {}
    output_formula = ''
    formula = ''
    final_parameters = ''
    weight_log = []

    return render(request,'navigation.html')


def get_username(request):
    global username
    data = {"username":username}
    return JsonResponse(data)


def ensemble_data(request):
    global weight_log
    weight_range = list(range(1,len(weight_log)))
    data = {"weight_log":weight_log,"weight_range":weight_range}
    return JsonResponse(data)
