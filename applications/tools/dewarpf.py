#!/usr/bin/env python
import cv2

import numpy as np
import numpy.core._dtype_ctypes
import time
import sys

assert sys.version_info >= (3,5)
'''
dewarp utility functions that take images and dewarp (no file IO)
all function take in raw: single image (nxmx3 array)
interp: is one of cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
stack methods:
0=normal, front view on top, driver view on bottom
1=just front view
2=just driver view
Run as main to test operation on single JPGs
For a utility to convert video files using these functions, see dewarp.py
'''
interp_methods = {"linear": cv2.INTER_LINEAR, 
                "cubic": cv2.INTER_CUBIC, 
                "lanczos": cv2.INTER_LANCZOS4}
#turn this on to save out images
debug=False

# The actual dewarp functions should take an image, and keyword args:
#supsamp=1 (supersampling amount, improves quality for some methods)
# interp=cv2.<constant>
# stack: 0=road/driver), 1=road, 2=driver

def imwrite(file, image):
    #write out an RGB image
    #cv2 expects BGR so use this function instead of cv2.imwrite
    if len(image.shape)>2:
        image=cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(file,image)


def dewarp_homog(raw, supsamp, interp, stack=0):
    #raw: input image
    #
    # quadrilaterals in input and output space to map img/
    pts_src = np.array([[626, 828],[679, 792],[666, 879],[727, 834]])
    pts_dst = np.array([[123, 318],[1248, 300],[159, 1830],[1218, 1860]])

    # Calculate homographic matrix
    hom, stat = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(raw, hom, (1536,2048))
     
    ret = cv2.resize(im_out, (0,0), fx=0.25, fy=0.25)
    return ret

def dewarp_linpol(raw, interp=cv2.INTER_LINEAR, supsamp=2, stack=0, facedown=0, watermark=0):
    #raw is 255-based image in RGB format (uint8)
    # scale up for supersampling
    #cv2.INTER_CUBIC gave some bad pixels (investigate later)
    #output should be as wide as input image, height is either one-half or same as input height,
    #depending on stack value
    #NOTE: supsamp=2 is much better than 1, 3 is slightly better than 2
#    debug = 1
    (rinput,cinput,_)=raw.shape
    info = raw[:rinput//40, :cinput//5]
    t1=time.time()
    if debug:
        imwrite('raw.jpg', raw)
    big = cv2.resize(raw, (0,0), fx=supsamp, fy=supsamp, interpolation=interp)
    if debug:
        imwrite('raw2.jpg', big)
    #print('after supersamp, shape={}'.format(big.shape))
    # apply linear depolar distort
    r,c = big.shape[:2]
    M = min(r,c)/2

    img64_float = big.astype(np.float64)
    if debug:
        imwrite('float64.jpg', big)
    t2=time.time()   
    rect = cv2.linearPolar(img64_float, (c/2, r/2), M, cv2.WARP_FILL_OUTLIERS | interp)
    t3=time.time()
    #print('after linearPolar {}'.format(rect.shape))
    if debug:
        imwrite('linpol.jpg', rect)
    # crop to regions of interest and stack them
    road_rect = rect[4*r//8:, c//2:c]          # road-facing view
    driver_rect = rect[:4*r//8, c//2:c]        # driver-facing view

    if facedown == 1:
        road_rect = road_rect[::-1, ::-1]  
        driver_rect = driver_rect[::-1, ::-1]  
    if stack==0:
        #print('stacking')
        rect = np.hstack((road_rect, driver_rect))
        #(rout,cout)=rinput,cinput
        (rout,cout)=1080,1920 #default:1080P
    elif stack==1:
        rect = road_rect
        #(rout,cout)=rinput//2,cinput
        (rout,cout)=1080//2, 1920 #default:1080P
    elif stack==2:
        rect = driver_rect
        #(rout,cout)=rinput//2,cinput
        (rout,cout)=1080//2,1920 #default:1080P

    if debug:
        imwrite('stacked.jpg', rect)
    # rotate and scale
    #print('stacked {}'.format(rect.shape))
    #M = cv2.getRotationMatrix2D((c//8,r//8), -90, 1)
    #rect = cv2.warpAffine(rect, M, (c, r))
    #rotate(-90) is transpose and then flip around y
    rect=cv2.transpose(rect)
    rect=cv2.flip(rect,1)
    #shrink by supsamp size, don't know size since may have stacked, etc.
    #print('rect shape {}'.format(rect.shape))
    r1,c1,_=rect.shape
    ret=cv2.resize(rect, (cout,rout), interpolation=interp)
    t4=time.time()
    if debug:
        imwrite('resized.jpg', ret)
    ret8=ret.astype(np.uint8)
    if watermark == 1:
        ret8[:rinput//40, :cinput//5] = info
    if debug:
        imwrite('resized8.jpg', ret)
    #print('resize {:8.3f} linpol {:8.3f} resize {:8.3f}'.format(t2-t1,t3-t2,t4-t3))
    return ret8

def dewarp_linpol2(raw, supsamp, interp, stack=1, facedown=0):
    #Use warpPerspective in chunks to achieve LinearPolar as a test
    #do not use supsamp
    print('input raw {}'.format(raw.shape))
    r,c = raw.shape[:2]
    imwrite('raw.jpg', raw)
    radius = min(r,c)/2
    N=100     # number of pieces to break circle into
    R1=0.51  # inner radius (fraction)
    R2=1.00  # outer radius (fraction)
    rect=np.zeros((4*r, c, 3))  # output size
    vstep=4*r//N
    center=np.array([radius, radius])
    angle_step=2*np.pi/N
    #NOTE: we do not have to do all the angles since we are cropping in afterward
    for i in range(N):
        hsize=c   #dst rectangle
        a1=angle_step*i
        a2=a1+angle_step
        y=vstep*i
        #coordinates are (x,y), or (column, row), be careful
        p1=center+R1*radius*np.array([np.cos(a1), np.sin(a1)])
        p2=center+R2*radius*np.array([np.cos(a1), np.sin(a1)])
        p3=center+R1*radius*np.array([np.cos(a2), np.sin(a2)])
        p4=center+R2*radius*np.array([np.cos(a2), np.sin(a2)])
        src=np.vstack(( p1, p2, p3, p4))
        dst=np.array([ [0,0], [hsize,0], [0, vstep], [hsize, vstep]])
        src=src.astype(np.float32)
        dst=dst.astype(np.float32)
        #print(src, src.shape, src.dtype)
        #print(dst, dst.shape, dst.dtype)
        M=cv2.getPerspectiveTransform(src, dst)
        #warpPerspective dsize args are (cols, rows) instead of normal
        output=cv2.warpPerspective(raw, M, (hsize, vstep))
        #print('output shape {}'.format(output.shape))
        #copy this into the output rect image
        rect[y:y+vstep,0:hsize, :]=output
    imwrite('built.jpg', rect)
    r1,c1,_=rect.shape
    Fclip=0.1   # fract of image to clip off the ends of driver and road
    ht=int((1-2*Fclip)*r1/2)
    bt=int(Fclip*r1)
    road_rect = rect[r1//2+bt:r1//2+bt+ht, :]        # road-facing view
    
    driver_rect = rect[bt:bt+ht, :]      # driver-facing view
    print('ht {} {} {}'.format(ht, road_rect.shape, driver_rect.shape))
    if stack==0:
        print('stacking')
        rect = np.hstack((road_rect, driver_rect))
    else:
        rect = road_rect
    # rotate and scale
    print('stacked {}'.format(rect.shape))
    #M = cv2.getRotationMatrix2D((c//8,r//8), -90, 1)
    #rect = cv2.warpAffine(rect, M, (c, r))
    #rotate(-90) is transpose and then flip around y
    rect=cv2.transpose(rect)
    rect=cv2.flip(rect,1)
    imwrite('rotated.jpg', rect)
    r1=r
    c1=c
    if stack!=0:
        r1=r1//2
    print('now {} resizing to {}'.format(rect.shape, (r, c)))
    return cv2.resize(rect, (c1, r1), interpolation=interp)/255

remapobj1=None   # global variable holding Remap object
remapobj2=None   # global variable holding Remap object

def dewarp_remap(raw, supsamp=None, interp='cubic', stack=1, facedown=0):
    #Dewarp using the Remap class (can use directly instead)
    #Note: This uses 2 global vars to hold the Remap() object.  Should use that class directly
    #if this causes problems.
    import remap
    assert stack in [0,1,2]
    global remapobj1
    global remapobj2
    #create the global objects
    if stack==0 or stack==1:
        if remapobj1 is None:
            #road facing
            remapobj1=remap.Remap(interp=interp)
    if stack==0 or stack==2:
        if remapobj2 is None:
            #driving facing
            remapobj2=remap.Remap(interp=interp, yfocal=1835)
    if stack==0:
        #both
        top=remapobj1.map(raw)
        bottom=remapobj2.map(raw)    
        return np.vstack((top, bottom))
    elif stack==1:
        #road only
        return remapobj1.map(raw)
    elif stack==2:
        #driver only
        return remapobj2.map(raw)    

def dewarp_jpg(infile, outfile, supsamp=None, interp=cv2.INTER_LINEAR, stack=0, dwfunc=dewarp_linpol):
    #dewarp a single JPG file
    image=cv2.imread(infile)
    #img=img[:,:,::-1].copy()   #convert to BGR (reverse index on last dimension)
    dw = dwfunc(image, supsamp=supsamp, interp=interp, stack=stack)
    imwrite(outfile, dw)
  
if __name__=='__main__':
    dewarp_methods = {"homog": None, 
                      "linpol": dewarp_linpol,
                      "linpol2": dewarp_linpol2,
                      "logpol": None, 
                      "remap": dewarp_remap}
    interp_methods = {"linear": cv2.INTER_LINEAR, 
                    "cubic": cv2.INTER_CUBIC, 
                    "lanczos": cv2.INTER_LANCZOS4}

    import argparse
    parser = argparse.ArgumentParser(description='convert video circular JPG into output JPG')
    parser.add_argument("infile", type=str, help='input jpg name')
    parser.add_argument("outfile", type=str, help='output jpg name')
    parser.add_argument("--method", type=str, help='dewarp method: linpol, remap', default='linpol')
    parser.add_argument("--clip", action='store_true', help='clip to bottom center 600x600')
    parser.add_argument("--supsamp", nargs=1, type=int, default=[2],
                        help="what factor to scale images for supersampling")
    parser.add_argument("--interp", choices=["linear", "cubic", "lanczos"], 
                        default="linear", help="interpolation method for opencv calls")
    parser.add_argument('--period', type=float, default=1.0,
                        help='approximate time (sec) between frames to export')
    parser.add_argument('--stack', type=int, help='how to combine images 0=stack, 1=road only, 2=driver only', default=0)
    options=parser.parse_args()
    #base, ext=os.path.splitext(os.path.basename(options.infile))
    #outfile=base+'-dewarped.jpg'
    dewarp_jpg(options.infile, options.outfile, supsamp=options.supsamp[0], interp=interp_methods[options.interp],
               stack=options.stack, dwfunc=dewarp_methods[options.method])

