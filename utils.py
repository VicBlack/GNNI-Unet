import matplotlib.pyplot as plt
import os
import skimage.io as io
from skimage import color
import numpy as np
import sys
import glob
import json

def plothistory(history, figure_path):
    # 绘制训练 & 验证的准确率值
    for metrictype in history:
        if 'val_' == metrictype[0:4]:
            continue
        fig = plt.figure()
        plt.plot(history[metrictype], label='train ' + metrictype)
        plt.plot(history['val_' + metrictype], label='val ' + metrictype)
        plt.title('Model ' + metrictype)
        plt.ylabel(metrictype)
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.legend(loc='upper left')
        fig.savefig(os.path.join(figure_path, metrictype + '.png'), dpi=300)


def plot_acc_loss(history, figure_path):
    fig = plt.figure()
    # 绘制训练 & 验证的acc & loss值
    plt.plot(history['acc'], 'r', label='train acc')
    # loss
    plt.plot(history['loss'], 'g', label='train loss')
    # val_acc
    plt.plot(history['val_acc'], 'b', label='val acc')
    # val_loss
    plt.plot(history['val_loss'], 'k', label='val loss')
    plt.title('Model Acc Loss')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc='upper left')
    fig.savefig(os.path.join(figure_path, 'model acc_loss.png'), dpi=300)


def saveResult(save_path, npyfile):
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        # img_nonzero = np.zeros(img.shape)
        # img_half = np.zeros(img.shape)
        # img_nonzero[img.nonzero()] = 1
        # img_half[img > 0.5] = 1
        imgdata = img * 255
        # imgdata_nonzero = img_nonzero * 255
        # imgdata_half = img_half * 255
        io.imsave(os.path.join(save_path, str(i) + "_predict.png"), imgdata.clip(0, 255, imgdata).astype(np.uint8))
        # io.imsave(os.path.join(save_path, str(i) + "_predict_nonzero.png"), imgdata_nonzero.clip(0, 255, imgdata_nonzero).astype(np.uint8))
        # io.imsave(os.path.join(save_path, str(i) + "_predict_half.png"), imgdata_half.clip(0, 255, imgdata_half).astype(np.uint8))


class Logger(object):
    def __init__(self, filename='train.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def saveOverlay(path):
    files_count = len(glob.glob(pathname=os.path.join(path, '*_predict.png')))
    for i in range(files_count):
        predict_files = glob.glob(pathname=os.path.join(path, str(i) + '_predict.png'))
        source_files = glob.glob(pathname=os.path.join(path, str(i) + '_source_img_*.png'))
        label_files = glob.glob(pathname=os.path.join(path, str(i) + '_source_label_*.png'))
        predict_image = io.imread(predict_files[0], as_gray=True)
        source_image = io.imread(source_files[0], as_gray=True)
        label_image = io.imread(label_files[0], as_gray=True)
        predict = np.zeros(predict_image.shape)
        label = np.zeros(label_image.shape)

        predict[predict_image >= 5] = 3
        label[label_image.nonzero()] = 2
        overlay = predict + label
        overlay[overlay == 5] = 1
        # color.label2rgb 会按label的值依次取color，这里需要注意有类别不存在的情况下，颜色顺序会有问题。
        image_label_overlay = color.label2rgb(overlay, image=source_image, colors=[(0, 1, 0), (1, 0, 0), (0, 0, 1)],
                                              alpha=0.4, bg_label=0, image_alpha=1.0)
        image_label_overlay = image_label_overlay * 255
        io.imsave(os.path.join(path, str(i) + "_result.png"), image_label_overlay.clip(0, 255, image_label_overlay).astype(np.uint8))


if __name__=='__main__':
    # conf_path = 'E:/WorkSpace/PYSpace/Heart/Unet2d/vicpc/train_result/configures/unet_gn_upsampling_2d_B2_SGD_ReLU_drop0.3_da-20190522-165901'
    # figure_path = 'E:/WorkSpace/PYSpace/Heart/Unet2d/vicpc/train_result/figures/unet_gn_upsampling_2d_B2_SGD_ReLU_drop0.3_da-20190522-165901'
    # with open(os.path.join(conf_path, "history.json"), "r", encoding='utf-8') as f:
    #     history = json.load(f)
    #     plothistory(history, figure_path)
    # path = 'E:/WorkSpace/PYSpace/Heart/TEMP/test_result/unet_bn_upsampling_2d_B12_SGD_LeakyReLU-20190505-175304'
    # path = 'E:/WorkSpace/PYSpace/Heart/TEMP/test_result/unet_bn_upsampling_2d_bs_24-20190427-083109'
    path = 'E:/WorkSpace/PYSpace/Heart/TEMP/test_result/unet_bn_upsampling_2d_B12_SGD_ReLU-20190516-084553'
    saveOverlay(path)

