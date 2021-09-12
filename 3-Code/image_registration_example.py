# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 18:07:57 2021

@author: talha
"""




from skimage import io
from skimage.transform import resize

#pip install image_registration

from image_registration import chi2_shift
from matplotlib import pyplot as plt
from scipy.ndimage import shift


bad_image = resize(io.imread("../2-Data/bad.png"),(299,299))
good_image = resize(io.imread("../2-Data/good.png"),(299,299))

def show_imgs(referance ,original , converted, axis = None,title = None ,save="test.jpg"):
    fig = plt.figure(figsize=(5,3),dpi=250)
    fig.suptitle(title,fontsize=20)

    fig.add_subplot(1,3,1)
    plt.axis(axis)
    plt.title('Reference Image')
    plt.imshow(good_image, cmap='gray')
    fig.add_subplot(1,3,2)
    plt.axis(axis)
    plt.title('Offset Image')
    plt.imshow(original, cmap='gray')
    fig.add_subplot(1,3,3)
    plt.axis(axis)
    plt.title('Registered Image')
    plt.imshow(converted, cmap='gray')
    plt.show()
    fig.savefig("../4-Results/"+save,dpi = 250) #dpi --> high resolution



# chi2_shift
#Find the offsets between image 1 and image 2 using the DFT 
#https://image-registration.readthedocs.io/en/latest/api/image_registration.chi2_shifts.chi2_shift.html
xoff, yoff, exoff, eyoff = chi2_shift(good_image, bad_image, upsample_factor='auto')

corrected_image = shift(bad_image, shift=(xoff, yoff), mode='constant')

show_imgs(good_image,bad_image,corrected_image,title="Chi2 Shift",axis=("off"),save="Chi2 Shift.jpg")



# cross_correlation_shifts
from image_registration import cross_correlation_shifts

xoff, yoff = cross_correlation_shifts(good_image, bad_image)

corrected_image = shift(bad_image, shift=(xoff,yoff), mode='constant')

show_imgs(good_image,bad_image,corrected_image,title="Cross Corelation Shift",axis=("off"),save="Cross Corelation Shift.jpg")


# register_translation
from skimage.feature import register_translation
shifted, error, diffphase = register_translation(good_image, bad_image, 100)
xoff = -shifted[1]
yoff = -shifted[0]

corrected_image = shift(bad_image, shift=(xoff,yoff), mode='constant')
show_imgs(good_image,bad_image,corrected_image,title="Register Translation",axis=("off"),save="Register Translation.jpg")



# Optical flow based shift
from skimage import registration

flow = registration.optical_flow_tvl1(good_image, bad_image)

import numpy as np
flow_x = flow[1, :, :]
flow_y = flow[0, :, :]
xoff = np.mean(flow_x)
yoff = np.mean(flow_y)
corrected_image = shift(bad_image, shift=(xoff,yoff), mode='constant')

show_imgs(good_image,bad_image,corrected_image,title="Optical Flow Shift",axis=("off"),save="Optical Flow Shift.jpg")


# Rigid Body transformation

from pystackreg import StackReg
sr = StackReg(StackReg.RIGID_BODY)
out_rigid = sr.register_transform(good_image, bad_image)
show_imgs(good_image,bad_image,out_rigid,title="Rigid Body Transformation",axis=("off"),save="Rigid Body Transformation.jpg")

# Affine transformation
sr = StackReg(StackReg.AFFINE)
out_aff = sr.register_transform(good_image, bad_image)
show_imgs(good_image,bad_image,out_aff,title="Affine Transformation",axis=("off"),save="Affine Transformation.jpg")

#Scaled Rotation transformation
sr = StackReg(StackReg.SCALED_ROTATION)
out_rot= sr.register_transform(good_image, bad_image)
show_imgs(good_image,bad_image,out_rot,title="Scaled Rotation Transformation",axis=("off"),save="Scaled Rotation Transformation.jpg")

#Bilinear transformation
sr = StackReg(StackReg.BILINEAR)
out_bil= sr.register_transform(good_image, bad_image)
show_imgs(good_image,bad_image,out_bil,title="Bilinear Transformation",axis=("off"),save="Bilinear Transformation.jpg")


