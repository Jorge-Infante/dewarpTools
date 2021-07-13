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





def correr(request):

    # url_link = "https://ingenioitc.com/mipg/34.mp4"
    # urllib.request.urlretrieve(url_link, 'static/in/1.mp4') 
    # ejecutar()
    

    try:
        passwd ='c4nd4d0$'
        if not passwd:
            quit()
        try:
            s = ftplib.FTP('ftp.ingenioitc.com','ftpdocker@docker.ingenioitc.com', passwd)
            f = open('static/in/1.mp4', 'rb')
            s.storbinary('STOR https://docker.ingenioitc.com/out/1.mp4', f)
 
            f.close()
            s.quit()
            print (": )")
        except:
            quit()
 
    except:
        quit()
  
    return HttpResponse('<h1>Video renderizado correctamente </h1>')





#Desde html
# def correr(request):
#     template_name = 'tools/video.html'
#     if request.method == 'POST':
#         url_link=str(request.POST.get('d1'))
#         url_link = "https://ingenioitc.com/mipg/34.mp4"
#         urllib.request.urlretrieve(url_link, 'static/in/1.mp4') 
#         ejecutar()

#     return render(request, template_name)
