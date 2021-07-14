#Python
import requests
import urllib.request
import ftplib
from ftplib import FTP
import getpass

#Django
from django.http import HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string 
#Funciones
from . dewarp import ejecutar 

import shutil
import urllib.request as request2
from contextlib import closing



def correr(request):

    # url_link = "https://ingenioitc.com/mipg/34.mp4"
    # urllib.request.urlretrieve(url_link, 'static/in/1.mp4') 
    # ejecutar()
    HOSTNAME = "ftp.ingenioitc.com"
    USERNAME = "ftpdocker@docker.ingenioitc.com"
    PASSWORD = "c4nd4d0$"

    ftp = ftplib.FTP(HOSTNAME)
    ftp.login(USERNAME,PASSWORD)
    ftp.set_pasv(False)  
    remotefile='/in/entra1.mp4'
    download='static/in/entrada1.mp4'
    with open(download, 'wb') as file:
        ftp.retrbinary('RETR %s' %remotefile, file.write )

    

    # with closing(request2.urlopen('ftp://tpdocker@docker.ingenioitc.com:c4nd4d0$@ftp.ingenioitc.com/in/', 'file')) as r:
    #     with open('file', 'wb') as f:
    #         shutil.copyfileobj(r, f)
    return HttpResponse('<h1>Video renderizado correctamente </h1>')

def out(request):

    HOSTNAME = "ftp.ingenioitc.com"
    USERNAME = "ftpdocker@docker.ingenioitc.com"
    PASSWORD = "c4nd4d0$"

    


    return HttpResponse('<h1>Video subido correctamente</h1>')



#Desde html
# def correr(request):
#     template_name = 'tools/video.html'
#     if request.method == 'POST':
#         url_link=str(request.POST.get('d1'))
#         url_link = "https://ingenioitc.com/mipg/34.mp4"
#         urllib.request.urlretrieve(url_link, 'static/in/1.mp4') 
#         ejecutar()

#     return render(request, template_name)
