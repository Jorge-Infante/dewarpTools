# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:00:50 2020

@author: Weiqiang
"""
#
from moviepy.editor import VideoFileClip
#import time
def get_aud(infiles,  mute_file, out_dir):   

#    mp4_file: path for save mp4 file
    
#    time1 = time.clock() 
    target_mp4 = ''.join((infiles.split('/')[-1]).split(".")[:-1]) + '-d.mp4' 
#    delList = os.listdir(out_dir)
#    for f in delList:
#        if f == target_mp4:
#            del_file = os.path.join(out_dir+'/', f )
#            os.remove(del_file)    


    video_input = VideoFileClip(infiles)
#    time2 = time.clock()
#    print("time1 cost:", round(time2-time1,1), 's')           
    audio = video_input.audio
    #audio.write_audiofile(mp3_file, logger=None)
#    time3 = time.clock()
#    print("time2 cost:", round(time3-time2,1), 's')               
    video_mute = VideoFileClip(mute_file)
#    time4 = time.clock()
#    print("time3 cost:", round(time4-time3,1), 's')   
    video_out = video_mute.set_audio(audio) 
#    time5 = time.clock()
#    print("time4 cost:", round(time5-time4,1), 's')   
    video_out.write_videofile(out_dir+'/'+target_mp4, logger=None) 
#    time6 = time.clock()
#    print("time5 cost:", round(time6-time5,1), 's')   
    print(target_mp4+" saved in "+out_dir+'/')
    

if __name__ == '__main__':
    # only for test
    get_aud(infiles='static/in/32.mp4', mute_file='static/out/temp/32-d_temp.mp4', out_dir='static/out')

    