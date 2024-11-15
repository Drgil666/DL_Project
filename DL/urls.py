"""
URL configuration for DL project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path

from app import views

urlpatterns = [
    path('',views.index,name='index'),
    path('register/',views.register,name='register'),
    path('upload/',views.upload,name='upload'),
    path('train/',views.train),
    path('show/',views.show),
    path('load/',views.load),
    path('ensemble/',views.ensemble,name='ensemble'),
    path('self/',views.self),
    path('get_data/',views.get_data),
    path('get_parameter/',views.get_parameter),
    path('model_data/',views.model_data),
    path('ensemble_data/',views.ensemble_data,name='ensemble_data'),
    path('download_model/',views.download_model),
    path('get_text/',views.get_text),
    path('model_recommendation/',views.model_recommendation),
    path('get_result/',views.get_result),
    path('download_csv/<int:file_id>/',views.download_csv,name='download_csv'),
    path('download_keras/<int:file_id>/',views.download_keras,name='download_keras'),
    path('download_img/<int:file_id>/',views.download_img,name='download_img'),
    path('warning/',views.warning,name='warning'),
    path('navigation/',views.navigation,name='navigation'),
    path('parameter_identification/',views.parameter_identification,name='parameter_identification'),
    path('parameter_identification_getdata/',views.parameter_identification_getdata,
         name='parameter_identification_getdata'),
    path('motivation_system_fitting/',views.motivation_system_fitting,name='motivation_system_fitting'),
    path('motivation_system_getdata/',views.motivation_system_getdata,name='motivation_system_getdata'),
    path('get_username/',views.get_username,name='get_username')

]
urlpatterns += staticfiles_urlpatterns()
