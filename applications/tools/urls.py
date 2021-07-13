from django.contrib import admin
from django.urls import path, re_path,include
from .views import correr
from applications.tools import views

# from . import views
app_name = "tools_app"

urlpatterns = [
    path('correr/',views.correr, name='datos')
]