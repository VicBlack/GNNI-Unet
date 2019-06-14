import shutil
import sys
sys.path.append('../')
from data_construct import travel_files, data_set_split
from data_generator import *
from unet2d_model import *
from utils import *
from configure import GetConfigure
from keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.utils import plot_model
import os
import time
import json
import warnings
warnings.filterwarnings('ignore')


def main():
    # ## load configure
    file_path, epochs, chosen_file_percent, predict_percent, params, net_conf, cudas, model_type, datagen = GetConfigure()
    os.environ["CUDA_VISIBLE_DEVICES"] = cudas
    model_name = model_type + '_B' + str(params['batch_size']) + '_' + str(type(net_conf['optimizer'])).split('\'')[1].split('.')[-1] + '_'\
                 + str(net_conf['activation']).split('\'')[1].split('.')[-1]
    model_name = model_name + '_drop' + str(net_conf['dropout']) if net_conf['dropout'] else model_name
    model_name = model_name + '_da' if datagen else model_name
    model_name += '-' + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    conf_path = 'train_result/configures/{}'.format(model_name)
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)
    weight_path = 'train_result/weights/{}'.format(model_name)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    figure_path = 'train_result/figures/{}'.format(model_name)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    test_result_path = 'test_result/{}'.format(model_name)
    if not os.path.exists(test_result_path):
        os.makedirs(test_result_path)
    sys.stdout = Logger(os.path.join(conf_path, 'train.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(conf_path, 'train_error.log'), sys.stderr)
    # ## configure dataset
    file_items = travel_files(file_path)
    partition = data_set_split(file_items, chosen_file_percent)
    training_generator = DataGenerator(partition['train'], **params, datagen=datagen)
    validation_generator = DataGenerator(partition['validate'], **params, datagen=datagen)
    test_generator = DataGenerator(partition['test'], **params, shuffle=False)
    predicting_generator = predictGenerator(partition['test'], **params,
                                            percent=predict_percent, save_path=test_result_path)

    model = GetNet(model_type, net_conf)
    early_stoping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(weight_path, model_type + '-{epoch:02d}-{val_loss:.5f}.hdf5'),
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
        mode='auto', period=1)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

    # ## save configures
    shutil.copy('configure.py', conf_path)
    with open(os.path.join(conf_path, "filelist.json"), "w", encoding='utf-8') as f:
        json.dump(partition, f, indent=4)
    plot_model(model, to_file=os.path.join(conf_path, model_type + ".png"), show_shapes=True)
    # ## training on train_dataset and validate_dataset
    print('>> Start Training')
    results = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=epochs,
                                  callbacks=[model_checkpoint, early_stoping, tensorboard])
    print('Validation_Accuracy: ', np.mean(results.history['val_acc']))
    # ## predict on test_dataset
    print('>> Start Predicting')
    p_test = model.predict_generator(predicting_generator, steps=int(np.ceil(len(partition['test']) * predict_percent)), verbose=1)
    saveResult(test_result_path, p_test)

    # ## evaluate on test_dataset
    print('>> Start Evaluating')
    eva = model.evaluate_generator(test_generator, verbose=1)
    metricsnames = model.metrics_names
    for i in range(len(metricsnames)):
        print(">> Testing dataset " + metricsnames[i] + " = {:.8f}".format(eva[i]))
    print('>> Run Model Completed !')

    # ## save figures
    plothistory(results, figure_path, metricsnames)
    plot_acc_loss(results, figure_path)


if __name__ == '__main__':
    main()


