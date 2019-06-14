import os
import random
import pydicom


def travel_files(file_path):
    file_items = []
    for root, dirs, files in os.walk(file_path, topdown=True):
        for name in files:
            if name.find('SA') > 0 and name.find('dcm') > 0:
                file_items.append((os.path.join(root, name), os.path.join(file_path, 'masks', name[:-3]+'png')))
    random.shuffle(file_items)
    return file_items


def travel_testfiles(file_path):
    file_items = []
    for root, dirs, files in os.walk(file_path, topdown=True):
        for name in files:
            if name.find('SA') > 0 and name.find('dcm') > 0:
                file_items.append(os.path.join(root, name))
    random.shuffle(file_items)
    return file_items


def data_set_split(file_items, full_percentage=1.0, test_percentage = 0.15, train_percentage = 0.8):
    partition = {}

    chosen_file_items = file_items[0: int(len(file_items)*full_percentage)]

    partition['test'] = chosen_file_items[0: int(len(chosen_file_items)*test_percentage)]
    train_list = chosen_file_items[int(len(chosen_file_items)*test_percentage): len(chosen_file_items)]


    partition['train'] = train_list[0:int(len(train_list)*train_percentage)]
    partition['validate'] = train_list[int(len(train_list)*train_percentage): len(train_list)]

    # print(partition)
    return partition


def count_data_size(file_path, outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for root, dirs, files in os.walk(file_path, topdown=True):
        count = 0
        for name in files:
            if name.find('SA') > 0 and name.find('dcm') > 0:
                im = pydicom.dcmread(os.path.join(root, name)).pixel_array
                filename = os.path.join(outpath, '%d' % im.shape[0] + 'x' + '%d' % im.shape[1] + '.txt')
                fp = open(filename, 'a+')
                fp.write(name + '\n')
                count += 1
                fp.close()
        print(count)
    outfp = open(os.path.join(outpath, 'sizecount.txt'), 'a+')
    for DataSize in os.listdir(outpath):  # DataSize -> width x height.txt
        if DataSize[:-4].find('x') > 0:
            fp = open(os.path.join(outpath, DataSize), "r")
            count = 0
            for i in fp:
                count = count + 1
            fp.close()
            print(DataSize[:-4] + " " + str(count))
            outfp.write(DataSize[:-4] + " " + str(count) + "\n")
    outfp.close()


if __name__=='__main__':
    file_path = 'E:/DATA/DCMS/'
    lvscpath = 'E:/DATA/DCMS/dcm/'
    scdpath = 'E:/DATA/SCD/SCD_DCMs/'
    lvscouterpath = 'E:/DATA/DCMS/sizecount/'
    scdoutpath = 'E:/DATA/SCD/sizecount/'
    # file_items = travel_files(file_path)
    # data_set_split(file_items, 0.2314, 0.2435, 0.5431)
    count_data_size(scdpath, scdoutpath)




