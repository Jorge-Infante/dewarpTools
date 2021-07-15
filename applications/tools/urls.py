from django.contrib import admin
from django.urls import path, re_path,include
from .views import entrada,salida
from applications.tools import views

# from . import views
app_name = "tools_app"

urlpatterns = [
    path('entrada/<str:name>/',views.entrada, name='datos'),
    path('salida/<str:nombre>/',views.salida, name='salida'),
]