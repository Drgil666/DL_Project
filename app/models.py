import os

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


def get_upload_path(instance,filename):
    # 获取上传文件的扩展名
    ext = filename.split('.')[-1]
    # 构建新的文件名，包括上传时间
    file_name_without_extension,extension = os.path.splitext(instance.file_name)
    new_filename = f"{file_name_without_extension}_{timezone.now().strftime('%Y%m%d%H%M')}.{ext}"
    return os.path.join('app/data_set/',new_filename)


def get_upload_model_path(instance,filename):
    # 获取上传文件的扩展名
    ext = filename.split('.')[-1]
    # 构建新的文件名，包括上传时间
    file_name_without_extension,extension = os.path.splitext(instance.file_name)
    new_filename = f"{file_name_without_extension}_{timezone.now().strftime('%Y%m%d%H%M')}.{ext}"
    return os.path.join('app/models/',new_filename)


def get_upload_img_path(instance,filename):
    # 获取上传文件的扩展名
    ext = filename.split('.')[-1]
    # 构建新的文件名，包括上传时间
    file_name_without_extension,extension = os.path.splitext(instance.file_name)
    new_filename = f"{file_name_without_extension}_{timezone.now().strftime('%Y%m%d%H%M')}.{ext}"
    return os.path.join('app/imgs_set/',new_filename)


class RegistrationForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['username','email','password1','password2']


class CSVFile(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    file = models.FileField(upload_to=get_upload_path)  # 使用函数作为 upload_to 参数
    file_name = models.CharField(max_length=255)  # 添加文件名字段
    file_size = models.IntegerField(default=0)  # 添加文件大小字段
    uploaded_at = models.DateTimeField(auto_now_add=True)


class IMGFile(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    file = models.FileField(upload_to=get_upload_img_path)  # 使用函数作为 upload_to 参数
    file_name = models.CharField(max_length=255)  # 添加文件名字段
    file_size = models.IntegerField(default=0)  # 添加文件大小字段
    uploaded_at = models.DateTimeField(auto_now_add=True)


class MODELFile(models.Model):
    user = models.ForeignKey(User,on_delete=models.CASCADE)
    file = models.FileField(upload_to=get_upload_model_path)  # 使用函数作为 upload_to 参数
    file_name = models.CharField(max_length=255)  # 添加文件名字段
    file_size = models.IntegerField(default=0)  # 添加文件大小字段
    uploaded_at = models.DateTimeField(auto_now_add=True)
    classes = models.TextField(default='')  # 存储类别列表的字段
