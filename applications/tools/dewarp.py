#!/usr/bin/env python
import sys  
import cv2
import os
import numpy as np
import numpy.core._dtype_ctypes
from imageio import get_reader as get_reader
#import time
from . import dewarpf   # actual dewarp functions
#import remap
from . import Audio
import time
from . import bar
#import argparse
from multiprocessing import Pool
from multiprocessing import freeze_support

'''
dewarp a video file of circular Secure 360 video files
will output a new mp4 file with dewarped video or a series of jpgs
This is a standalone utility program.
For functions in a module to call, see dewarpf.py

'''
assert sys.version_info >= (3,5)
def replace_filename(pathname, new_dir=None, ext=None, add=''):
    #returns a pathname changed.  Where present:
    #Directory becomes new_dir, e.g. './out'
    #extension becomes ext, e.g. 'jpg'
    #add is added to filename before ext, e.g. '-v2'
    #ext should be e.g. 'jpg' ,'mp4', etc.
    d, name=os.path.split(pathname)
    base,ex=os.path.splitext(name)   # ex has the . in it already
    if ext:
        ex='.'+ext
    if new_dir:
        d=new_dir
    return os.path.join(d, base+add+ex)



def dewarp_to_jpg(infiles, dw_func, outdir, out_video_shape, supsamp, interp, period=1, stack=1, clip=False, jpg=False, skip=False, facedown=0, watermark=0):
    # Directories check
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    
    num_files=len(infiles)
    
    for i, imfile in enumerate(infiles):
        
        # Video information
        vid=get_reader(imfile, 'ffmpeg')
        nframes = vid.count_frames()
        print("\n{}/{} Dewarp {}'s {} frames to JPG ".format(i+1, num_files, imfile, nframes))

        # Process each frame
        draw_bar = bar.ProgressBar(max_value=int(nframes), name=imfile)
        for idx in range(0, nframes, period):
            try:
                img=vid.get_data(idx)
            except RuntimeError as ex:
                print('got error {}, skipping file'.format(ex))
                break
            draw_bar.update()          
            img=img[:,:,::-1].copy()
            
            # dewarp the frame using specified method
            dw = dw_func(img, supsamp=supsamp, interp=interp, stack=stack, facedown=facedown, watermark=watermark)     

            # save to file
            dw = dw.astype(np.float32)
            outfile=replace_filename(imfile, new_dir=outdir, ext='jpg', add='-{:04d}'.format(idx))
            outfile = outfile.replace('\\', '/')
            cv2.imwrite(outfile, dw)
        vid.close()
        
result = []
def call_back(args):
    global result
    result.append(args)
    
def dewarp_help(frame,supsamp,interp,stack,facedown,watermark,dw_func):
    
    dw = dw_func(frame, supsamp=supsamp, interp=interp, stack=stack, facedown=facedown, watermark=watermark) 
    dw=dw.astype(np.uint8)
    return dw
    

def dewarp_to_video(infiles, dw_func, outdir, out_video_shape, supsamp, interp, period=1, stack=1, clip=False, jpg=False, skip=False, facedown=0, watermark=0, multiprocess=0, audio=0):
    # Directories check
    
    outdir_tmp = outdir + '/temp' if audio==1 else outdir
    try:
        os.makedirs(outdir_tmp)
    except FileExistsError:
        pass    
    if os.path.exists(outdir_tmp): 
        delList = os.listdir(outdir_tmp)
        for f in delList:
            del_file = os.path.join( outdir_tmp, f )
            del_file = del_file.replace('\\', '/')
            os.remove(del_file)

    for i, imfile in enumerate(infiles):
        start = time.time
        print('{}/{} file: {} '.format(i+1, len(infiles), imfile), end=" ")
        outfile = replace_filename(imfile.split('/')[-1], new_dir=outdir_tmp, ext='mp4', add='-d_temp') if audio==1 else replace_filename(imfile, new_dir=outdir_tmp, ext='mp4', add='-d')     
        outfile = outfile.replace('\\', '/')   #./out/temp/input-d_temp.mp4  or #./out/input-d.mp4    
        # Video information
        cap = cv2.VideoCapture(imfile)   
        if cap.isOpened():
            fps = cap.get(5)
            FrameNumber = cap.get(7)
            duration = FrameNumber/fps     
            print('frames:{:.0f}  duration:{:.2f}s  fps:{:.0f}'.format(FrameNumber, duration, fps))
        else:
            print('unable to read video file: {}'.format(imfile.split('/')[-1]))
            return
        
        # Create video stream object by cv2
#        print(outfile)
        out_video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, out_video_shape) 
        _, mp4_name = os.path.split(outfile)
        
# -------------miltiprocessing & apply_async-------- 
        if multiprocess==1:
            draw_bar = bar.ProgressBar(max_value=int(FrameNumber), name=imfile.split('/')[-1])
            framelst = []  
            while(1): 
                ret, frame = cap.read()
                if ret == False:
                    break
                framelst.append(frame)
    #        print("time cost:", round(time.clock()-start,1), 's')
            pool = Pool()
            for frame in framelst:
                pool.apply_async(dewarp_help, args = (frame,supsamp,interp,stack,facedown,watermark,dw_func,), callback=call_back)       
            pool.close()
    #        pool.join()# 主进程阻塞 
            tmp = 0      
            while(len(result)<=FrameNumber):
                if tmp != len(result):
                    tmp+=1
                    draw_bar.update()
                if len(result)==FrameNumber:
                    break
                time.sleep(0.5)
            _ = [draw_bar.update() for i in range(len(result)-tmp)]   
            for warpedframe in result:
                out_video.write(warpedframe)
          
# ---------------------------------        

# --------------multi processing---      
#        framelst = []  
#        while(1): 
#            ret, frame = cap.read()
#            if ret == False:
#                break
#            framelst.append((frame,supsamp,interp,stack,facedown,watermark,dw_func))
#        warped = []
#        with Pool() as p:
#            warped = p.starmap(dewarp_help, framelst)
#        print(len(warped))    
#        for warpedframe in warped:
#            out_video.write(warpedframe)                
# ------------------------------------   
          
# --------------single processing---
        if multiprocess==0:
            draw_bar = bar.ProgressBar(max_value=int(FrameNumber), name=imfile.split('/')[-1])
            # Process each frame
            while(1): 
                draw_bar.update()        
                ret, frame = cap.read()
                if ret == False:
                    break
                dw = dw_func(frame, supsamp=supsamp, interp=interp, stack=stack, facedown=facedown, watermark=watermark) 
                dw=dw.astype(np.uint8)
                out_video.write(dw)
# ------------------------------------            
            
        # Release streams     
        cap.release()  
        out_video.release()
        cv2.destroyAllWindows()
        end = time.clock()
#        print("time cost:", round(end-start,1), 's')
        
        # Add audio
        if audio==1:
            print("Adding audio...")  
            mute_file = outdir_tmp + '/' + mp4_name
            Audio.get_aud(imfile, mute_file, outdir)
#        print("time cost:", round(time.clock()-end,1), 's')
        print()

# Not available
def homog(raw, supsamp=None, interp=None, stack=None, facedown=0):
    # quadrilaterals in input and output space to map img/
    pts_src = np.array([[626, 828],[679, 792],[666, 879],[727, 834]])
    pts_dst = np.array([[123, 318],[1248, 300],[159, 1830],[1218, 1860]])

    # Calculate homographic matrix
    hom, stat = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(raw, hom, (1536,2048))
     
    ret = cv2.resize(im_out, (0,0), fx=0.25, fy=0.25)
    return ret

def no_dewarp(raw, supsamp=None, interp=None, stack=None, facedown=0):
    #This does no dewarping
    return raw
        

def ejecutar():
        freeze_support()
        dewarp_methods = {"homog": homog, 
                        "linpol": dewarpf.dewarp_linpol,
                        "linpol2": dewarpf.dewarp_linpol2,
                        "logpol": None, 
                        "remap": dewarpf.dewarp_remap,
                        "none": no_dewarp}
        
        interp_methods = {"linear": cv2.INTER_LINEAR, 
                        "cubic": cv2.INTER_CUBIC, 
                        "lanczos": cv2.INTER_LANCZOS4}
        
        # Select mp4 shape
        out_shapes = {"linpol": (1920, 1080),
                    "linpol2": (1920, 1080),
                    "remap": (1952, 1952),
                    "homog":(384, 512)}           
        out_shapes_half = {"linpol": (1920, 540),
                    "linpol2": (1920, 540),
                    "remap": (1952, 976),
                    "homog":(384, 256)}   
        
        
        print("Waylens Dewarp Tool, version: v1.1.6")
    
        infiles = []
        method = 'linpol'
        outdir = './out'
        clip = 0
        supsamp = 2
        interp = 'linear'
        period = 1
        stack = 0
        jpg = 0
        skip = 0
        facedown = 1
        watermark = 0
        multiprocess = 0
        audio = 1

        # 从settings.txt中读入
        for line in open("static/config/settings.txt","r"):
            if line == '\n':
                break
            line = line.split()
            if line[0] == '--infilesdir':
                for i in os.listdir(line[1]):
                    infiles.append((os.path.join(line[1],i)).replace('\\', '/'))
                    
            if line[0] == '--method':
                method = line[1]
            elif line[0] == '--outdir':
                outdir = line[1]
            elif line[0] == '--clip':
                if line[1] == 1:
                    clip = True
            elif line[0] == '--supsamp':
                supsamp = int(line[1])
            elif line[0] == '--interp':
                interp = line[1]
            elif line[0] == '--period':
                period = int(line[1])    
            elif line[0] == '--stack':
                stack = int(line[1])
            elif line[0] == '--jpg':
                if line[1] == 1:
                    jpg = True
            elif line[0] == '--skip':
                if line[1] == 1:
                    skip = True      
            elif line[0] == '--facedown':
                facedown = int(line[1])
            elif line[0] == '--watermark':
                watermark = int(line[1])  
            elif line[0] == '--multiprocess':
                multiprocess = int(line[1])
            elif line[0] == '--audio':
                audio = int(line[1])
                
        dewarp_to_video(infiles, dewarp_methods[method], outdir, out_shapes[method],
                    supsamp, interp_methods[interp], period,
                    stack, clip, jpg, skip, facedown, watermark, multiprocess, audio)
        
