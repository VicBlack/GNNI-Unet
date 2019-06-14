import pydicom
import numpy as np
import skimage.io as io
from skimage import color
from matplotlib import pyplot as plt
from keras.preprocessing.image import *
import random

def img_crop(data, cropshape=(256, 256)):
    cropwidth = cropshape[0]
    cropheight = cropshape[1]
    data = np.array(data)
    croped = np.zeros((cropwidth, cropheight), dtype=np.uint16)
    wpos = int(abs((cropwidth - data.shape[0])/2))
    hpos = int(abs((cropheight - data.shape[1])/2))
    if(data.shape[0] < cropwidth):
        if(data.shape[1] < cropheight):
            croped[wpos: wpos + data.shape[0], hpos: hpos + data.shape[1]] = data
        else:
            croped[wpos: wpos + data.shape[0], :] = data[:, hpos: hpos + cropheight]
    else:
        if (data.shape[1] < cropheight):
            croped[:, hpos: hpos + data.shape[1]] = data[wpos:wpos + cropwidth, :]
        else:
            croped = data[wpos:wpos + cropwidth, hpos: hpos + cropheight]
    return croped

def img_load(item, shape=None, norm=True):
    img = pydicom.dcmread(item).pixel_array
    if(shape):
        img = img_crop(img, shape)
    if(norm):
        img = img_norm(img)
    return img

def lab_load(item, shape=None, norm=False, binary=True):
    lab = io.imread(item, as_gray=True)
    if (shape):
        lab = img_crop(lab, shape)
    if (norm):
        lab = img_norm(lab)
    if (binary):
        lab[lab.nonzero()] = 1
        # lab = lab.astype(np.bool)
    return lab

def img_norm(data):
    data = np.array(data)
    max = np.max(data)
    min = np.min(data)
    if max == min:
        return np.zeros(data.shape)
    data = (data - min)/(max - min)
    return data

if __name__=='__main__':
    file_path = 'E:/DATA/DCMS/dcm/DET0000101_SA2_ph5.dcm'
    label_path = 'E:/DATA/DCMS/masks/DET0000101_SA2_ph5.png'
    # datagen = ImageDataGenerator(
    #     rotation_range=10,
    #     width_shift_range=0.05,
    #     height_shift_range=0.05,
    #     shear_range=0.05,
    #     zoom_range=0.05,
    #     fill_mode='nearest',
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     dtype=np.float64)
    img = img_load(file_path, shape=(256, 256))
    lab = lab_load(label_path, shape=(256, 256))
    # img = color.gray2rgb(img)
    # imgdata = img * 255
    # labdata = lab * 255

    image_label_overlay = color.label2rgb(lab, image=img, colors=[(1, 0, 0)], alpha=0.3, bg_label=0)
    io.imshow(image_label_overlay)
    # img = img.reshape(*img.shape, 1)
    # lab = lab.reshape(*lab.shape, 1)
    # seed = random.randint(1, 100000)
    # img = datagen.random_transform(img, seed)
    # lab = datagen.random_transform(lab, seed)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img[:, :, 0], 'gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(lab[:, :, 0], 'gray')
    plt.show()




