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
from django.shortcuts import redirect
from django.http import JsonResponse

#Json
from django.http import JsonResponse

#Funciones
from . dewarp import ejecutar 




def entrada(request, name):
    #Descarga normal
    # url_link = "https://ingenioitc.com/mipg/34.mp4"
    # urllib.request.urlretrieve(url_link, 'static/in/1.mp4') 

    #Descarga ftp
    if request.method == 'GET':
        nombre=name

        HOSTNAME = "ftp.ingenioitc.com"
        USERNAME = "ftpdocker@docker.ingenioitc.com"
        PASSWORD = "c4nd4d0$"

        ftp = ftplib.FTP(HOSTNAME)
        ftp.login(USERNAME,PASSWORD)
        ftp.set_pasv(False)  
        remotefile='/in/'+nombre+'.mp4'
        download='static/in/'+nombre+'.mp4'
        with open(download, 'wb') as file:
            ftp.retrbinary('RETR %s' %remotefile, file.write )


        # ejecutar()
        data=salida(request,nombre) 
    return JsonResponse(data)

def salida(request,nombre):
    print(nombre)

    HOSTNAME = "ftp.ingenioitc.com"
    USERNAME = "ftpdocker@docker.ingenioitc.com"
    PASSWORD = "c4nd4d0$"

    ftp = ftplib.FTP(HOSTNAME)
    ftp.login(USERNAME,PASSWORD)
    ftp.set_pasv(False)
    filename=nombre+'-d.mp4'
    localfile='static/out/'
    remotefile='/out/'
    ftp.cwd(remotefile)
    ftp.storbinary("STOR %s" %filename, open("%s%s" % (localfile,filename),'rb'))


    responseData = {
        'url': 'https://docker.ingenioitc.com/out/'+filename
    }

    return responseData



#Desde html
# def correr(request):
#     template_name = 'tools/video.html'
#     if request.method == 'POST':
#         url_link=str(request.POST.get('d1'))
#         url_link = "https://ingenioitc.com/mipg/34.mp4"
#         urllib.request.urlretrieve(url_link, 'static/in/1.mp4') 
#         ejecutar()

#     return render(request, template_name)
