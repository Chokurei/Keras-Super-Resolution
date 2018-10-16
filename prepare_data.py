# -*- coding: utf-8 -*-
import os
import cv2
import h5py
import numpy

Random_Crop = 30
Patch_size = 32
label_size = 20
conv_side = 6

PATCH_SIZE = 32
STRIDE = 16
def prepare_train_data(_path, scale):
    """
    Prepare training data: get crops from each training patches with stride = STRIDE
    data size(lr): 32 x 32
    label size(hr): 20 x 20
    """
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = []
    label = []

    for i in range(nums):   
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]
        shape = hr_img.shape

        # two resize operation to produce training data and labels
        lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        width_num = (shape[0] - (PATCH_SIZE - STRIDE) * 2) // STRIDE
        height_num = (shape[1] - (PATCH_SIZE - STRIDE) * 2) // STRIDE
        for k in range(width_num):
            for j in range(height_num):
                x = k * STRIDE
                y = j * STRIDE
                hr_patch = hr_img[x: x + PATCH_SIZE, y: y + PATCH_SIZE]
                lr_patch = lr_img[x: x + PATCH_SIZE, y: y + PATCH_SIZE]

                lr_patch = lr_patch.astype(float) / 255.
                hr_patch = hr_patch.astype(float) / 255.

                lr = numpy.zeros((1, Patch_size, Patch_size), dtype=numpy.double)
                hr = numpy.zeros((1, label_size, label_size), dtype=numpy.double)

                lr[0, :, :] = lr_patch
                hr[0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]

                data.append(lr)
                label.append(hr)

    data = numpy.array(data, dtype=float)
    label = numpy.array(label, dtype=float)
    return data, label

def prepare_test_data(_path, scale):
    """
    Prepare testing data: randomly get Random_Crop crops from each testing image
        nums(images) * Random_Crop in total
    data size(lr): 32 x 32
    label size(hr): 20 x 20
    """
    names = os.listdir(_path)
    names = sorted(names)
    nums = names.__len__()

    data = numpy.zeros((nums * Random_Crop, 1, Patch_size, Patch_size), dtype=numpy.double)
    label = numpy.zeros((nums * Random_Crop, 1, label_size, label_size), dtype=numpy.double)

    for i in range(nums):
        name = _path + names[i]
        hr_img = cv2.imread(name, cv2.IMREAD_COLOR)
        shape = hr_img.shape

        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_img = hr_img[:, :, 0]

        # two resize operation to produce training data and labels
        # resolution becomes 1/2
        lr_img = cv2.resize(hr_img, (shape[1] // scale, shape[0] // scale))
        lr_img = cv2.resize(lr_img, (shape[1], shape[0]))

        # produce Random_Crop random coordinate to crop training img
        Points_x = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)
        Points_y = numpy.random.randint(0, min(shape[0], shape[1]) - Patch_size, Random_Crop)

        for j in range(Random_Crop):
            lr_patch = lr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]
            hr_patch = hr_img[Points_x[j]: Points_x[j] + Patch_size, Points_y[j]: Points_y[j] + Patch_size]

            lr_patch = lr_patch.astype(float) / 255.
            hr_patch = hr_patch.astype(float) / 255.

            data[i * Random_Crop + j, 0, :, :] = lr_patch
            label[i * Random_Crop + j, 0, :, :] = hr_patch[conv_side: -conv_side, conv_side: -conv_side]    
    return data, label



def write_hdf5(data, labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    x = data.astype(numpy.float32)
    y = labels.astype(numpy.float32)

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)
        # h.create_dataset()


def read_training_data(file):
    with h5py.File(file, 'r') as hf:
        data = numpy.array(hf.get('data'))
        label = numpy.array(hf.get('label'))
        train_data = numpy.transpose(data, (0, 2, 3, 1))
        train_label = numpy.transpose(label, (0, 2, 3, 1))
        return train_data, train_label

if __name__ == "__main__":
    #train_data_path = '/media/kaku/HDCL-UT/remote sensing/village mapping(Laos Thai Kenya)/training/Laos/image/'
    train_data_path = "./data/train/"
    test_data_path =  "./data/test/"
    data_cache = "./logs/data_cache/"
    # upscale and downscale factor
    scale = 2

    train_data, train_label = prepare_train_data(train_data_path, scale)
    write_hdf5(train_data, train_label, os.path.join(data_cache,"train_data.h5"))
    print(train_data.shape)
    print(train_label.shape)
    test_data, test_label = prepare_test_data(test_data_path, scale)
    print(test_data.shape)
    print(test_label.shape)
    write_hdf5(test_data, test_label, os.path.join(data_cache,"test_data.h5"))
#    # _, _a = read_training_data("train.h5")
    # _, _a = read_training_data("test.h5")