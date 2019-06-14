import sys
sys.path.append('../')
import os
import random
import skimage.io as io
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from unet2d_model import *
from data_construct import travel_testfiles
from data_preprocess import img_load


def testGenerator(test_path, num_image=50, target_size=(256, 256), result_path='test_result/'):
    filelist = travel_testfiles(test_path)
    files = random.sample(filelist, num_image)
    for i, item in enumerate(files):
        img_array = img_load(os.path.join(test_path, item), shape=target_size, norm=True)
        io.imsave(os.path.join(result_path, "%d_source.png" % i), img_array)
        img_array = np.reshape(img_array, img_array.shape + (1,))
        img_array = np.reshape(img_array, (1,) + img_array.shape)
        yield img_array


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        io.imsave(os.path.join(save_path, "%d_predict.png"%i), img)


def test(file_path, model_path, netconf, target_size=(256, 256), test_num=50, result_path='test_result/'):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    test_generator = testGenerator(file_path, num_image=test_num, target_size=target_size, result_path=result_path)
    model = unet_bn_t(**netconf)
    model.load_weights(model_path)
    test_results = model.predict_generator(test_generator, test_num, verbose=1)
    saveResult(result_path, test_results)


if __name__ == '__main__':
    net_conf = {'pretrained_weights': None,
               'input_size': (256, 256, 1),
               'depth': 4,
               'n_base_filters': 64,
               'optimizer': Adam,
               'activation': LeakyReLU,
               'batch_normalization': True,
               'initial_learning_rate': 5e-4,
               'loss_function': dice_coefficient_loss,
               'multi_gpu_num': 0}

    test_conf = {'file_path': '/data/data/Validation/dcm/',
                'model_path': 'train_result/weights/unet_bn_t_2d-25-0.99848.hdf5',
                'netconf': net_conf,
                'target_size': (256, 256),
                'test_num': 50,
                'result_path': 'test_result/'}

    test(**test_conf)
