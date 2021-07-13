#!/usr/bin/env python
import cv2
import numpy as np
import numpy.core._dtype_ctypes
import sys
import os
from pickle import load as load
from pickle import dump as dump

'''
uses remap to correctly dewarp Secure 360 circular image file to pinhole camera model
Note that parameters for direction and step are hard-coded
Putting some parameters as globals variables here - need better way to calibrate by camera,
as centers, etc. can vary somewhat

'''
#these are camera parameters
fov_deg=107         # degrees of full phi (vertical=0)
fov_radius=964      # pixel radius that corresponds to this fov_deg
mask = None

'''
Class to perform remap function.  Need to create class first to handle generation of mapping
tables, which may be cached in a disk pickle file (in current directory)

Usage:

remap=Remap()

new_image=remap.map(image)
'''
assert sys.version_info >= (3,5)

def cache_dir():
    #return the cache directory to use and make sure it exists
    #uses env var WCACHE or ~/.wcache
    cache=os.environ.get('WCACHE', os.path.expanduser('~/.wcache'))
    try:
        os.mkdir(cache)
    except FileExistsError:   # Python 3
        pass
    return cache

class Remap():
    '''
    class to handle remapping.  Does write a pickle file in current directory to cache
    the computation of the map
    '''
    interp_dict = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
    #interp can be the text string or the cv2 constant
    def __init__(self, shape=(976,1952),
                 fov_deg=fov_deg, fov_radius=fov_radius,
                 xcenter=960,
                 ycenter=960,
                 xfocal=960,
                 yfocal=85,
                 step=0.0008,
                 interp='cubic'):
        cs=[cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
        assert interp in Remap.interp_dict or interp in cs
        self.shape=shape
        self.fov_deg=fov_deg
        self.fov_radius=fov_radius
        self.xcenter=xcenter
        self.ycenter=ycenter
        self.xfocal=xfocal
        self.yfocal=yfocal
        self.step=step
        if interp in cs:
            self.interp=interp
        else:
            self.interp=Remap.interp_dict[interp]
        #cache and/or generate the mapping data
        filename='map-{}-{}-{}-{}-{}-{}-{}.pickle'.format(shape[0],shape[1],xcenter,ycenter,xfocal,yfocal,int(step*10000))
        filename=os.path.join(cache_dir(), filename)
        if os.path.isfile(filename):
#            print('reading map file')
            with open(filename, 'rb') as fp:
                maps=load(fp)
                mapx=maps['mapx']
                mapy=maps['mapy']
#            print('Done')
        else:
            print('creating map...')
            big_map=make_map(self.shape, (self.xfocal,self.yfocal), (self.xcenter,self.ycenter), self.step)
            print('created map')
            mapx, mapy = cv2.convertMaps(big_map, None,  dstmap1type=cv2.CV_16SC2)
            print('converted map')
            #save the maps
            maps={'mapx':mapx,'mapy':mapy}
            with open(filename,'wb') as fp:
                dump(maps, fp)
        self.mapx=mapx
        self.mapy=mapy
        
    def map(self,img):
        return cv2.remap(img, self.mapx, self.mapy, interpolation=self.interp)
    
def lensalign(img):
    global mask
    h,w,_ = img.shape
    
    if mask is None:
        _, mask = cv2.threshold(
                    cv2.circle(
                        cv2.rectangle(
                            cv2.circle(
                                np.zeros((h,w), dtype=np.uint8),
                            (w//2,h//2),w//2+50,[255,255,255],-1),
                        (0,0),(480,120),[0,0,0],-1),
                    (w//2,h//2),w//2-50,[0,0,0],-1), 
                127, 255, cv2.THRESH_BINARY)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 127, 256)
    canny = cv2.bitwise_or(canny, canny, mask=mask)
    
    # find contours as points
    im, cnt, hier = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    # generate enclosing circle and update running average
    points = np.array([(c[0][0][0],c[0][0][1]) for c in cnt])
    return cv2.minEnclosingCircle(points)


def unit(v):
    if np.any(v):
        return v / np.linalg.norm(v)
    return 0

def to_object_coords(a,b,h,k):
    #(a,b) is center of circular image
    #(h,k) is point in circular image we want camera pointing at
    #returns p vector (center of output image and u,v unit vectors.
    #u,v are the normal coordinate system for output map
    #  5.27521424e-08 8.35108324e-04 4.31154361e-03
    print(a,b,h,k)
    def invF_lens2(r):
        #for testing, linear 110 deg over 960 px
        return r/fov_radius*fov_deg*np.pi/180
    
    def invF_lens(r):
        return .004312 + .0008351*r + .0000000528*r**2
    
    # convert to 2d aligned to center
    x = a-h
    y = b-k
    r = np.sqrt(x**2+y**2)

    # convert to 3d spherical coords
    th = np.arctan2(y,x)
    ph = invF_lens2(r)
    R = 1

    # convert to object coord system
    p = R*np.array([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)])
    v = R*np.array([-np.cos(ph)*np.cos(th), -np.cos(ph)*np.sin(th), np.sin(ph)])
    u = np.cross(p,v)
    return p,unit(u),unit(v)

def to_image_coords(qx,qy,qz,h,k):
    #convert x,y,z point into image coordinates (circular image)
    # -17.96335853   34.15082381  -55.82450804 1151.18053144    3.33807318
    def F_lens2(ph):
        #linear for testing
        return ph/np.pi*180/fov_deg*fov_radius

    def F_lens(ph):
        return 3.3381 + 1151.1805*ph - 55.8245*ph**2 + 34.1508*ph**3 - 17.9634*ph**4
    
    # convert to 3d spherical coords
    R = np.sqrt(qx*qx + qy*qy + qz*qz)
    ph = np.arccos(qz / R)
    th = np.arctan2(qy, qx)
    
    # convert to 2d aligned to center
    r = F_lens2(ph)
    y = r*np.sin(th)
    x = r*np.cos(th)

    # convert to image coord system
    b = y+k
    a = x+h
    return a,b

def x(a):
    return 2*a

def make_map(dim,foc,ctr,eps):
    print('make map',dim,foc,ctr,eps)
    j,i = dim
    p,u,v=to_object_coords(*foc,    *ctr)
    mapxy = np.empty(dim, dtype=(np.float32,2))
    
    for n in range(j):
        for m in range(i):
            # scan from the top left corner across each row then down
            y = -n + j//2
            x = m - i//2
            
            # compute pointing vector and image projection
            q = p + (u*x + v*y)*eps
            a, b = to_image_coords(*q, *ctr)
            mapxy[n][m] = (a, b)
    return mapxy

if __name__=="__main__":
    import argparse, glob
    parser = argparse.ArgumentParser()
    parser.add_argument("indir")
    parser.add_argument("outdir")
    parser.add_argument("--interp", "-i", nargs=1, default="cubic")    # linear, cubic, lanczos
    args = parser.parse_args()
    indir, outdir = args.indir, args.outdir
    #center of image (h,k) (x,y)
    remap=Remap(interp=args.interp)
    sourceimgs = sorted(glob.glob("./" + indir + "/*"))
    for i, imfile in enumerate(sourceimgs):
        img = cv2.imread(imfile)
        out = remap.map(img)
        cv2.imwrite("./" + outdir + "/%05d.jpg" % i, out)
        print('did {}'.format(imfile))



