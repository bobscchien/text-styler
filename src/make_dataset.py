import glob
import argparse
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from utils import configuration
from utils.configuration import *
from utils.preprocessor import *

changeToColabPath(configuration.colab)
createDirectory()

def preprocess_pair_data(from_file, lang, min_inp_len=4, max_inp_len=256, min_tar_len=2, max_tar_len=256, 
                         size=None, split=0.2, plot=False, random_state=24601):

    print(f"\nStart process raw data...\n")
    
    # cctc
    print('  Start processing CCTC dataset...')

    dataset = []
    for dirpath, dirnames, filenames in tqdm.tqdm(os.walk(from_file)):
        if (len(filenames) > 0) & ('ccpm' not in dirpath):
            for filename in filenames:
                book = dirpath[dirpath.rfind('/')+1:]
                data = pd.read_json(os.path.join(dirpath, filename))
                for title, contents in data.values:
                    data_tmp = pd.DataFrame(contents).assign(title=title, book=book)
                    dataset.append(data_tmp)
    data1 = pd.concat(dataset)
    data1 = data1.rename(columns={'target':'source', 'source':'target'})      

    # ccpm
    print('  Start processing CCPM dataset...')

    file_tmp = []
    for file in glob.glob(os.path.join(from_file, 'ccpm', '*')):
        with open(file) as f:
            file_tmp += f.readlines()

    dataset = []
    for text in tqdm.tqdm(file_tmp):
        data_tmp = pd.read_json(text)
        data_tmp = data_tmp.iloc[data_tmp.answer.unique()[0], :2].to_frame().T
        dataset.append(data_tmp)
    data2 = pd.concat(dataset)
    data2 = data2.rename(columns={'translation':'source', 'choices':'target'})      

    # combine
    data = pd.concat([data1, data2])[['source', 'target']]
    data = data.assign(source=data.source.str.replace('\(.*?\)|（.*?）', '', regex=True), 
                       target=data.target.str.replace('\(.*?\)|（.*?）', '', regex=True))

    # specify the length range for datasets
    print(' Before clean, data size is:, ', len(data))
    data = data.assign(length_inp=data.source.apply(len))
    data = data[(min_inp_len<=data.length_inp)&(data.length_inp<=max_inp_len)]
    data = data.assign(length_tar=data.target.apply(len))
    data = data[(min_tar_len<=data.length_tar)&(data.length_tar<=max_tar_len)]
    print(' After clean, data size is:, ', len(data))

    # sample part of data
    if size:
        size = min(size, len(data))
        data = data.sample(n=size, replace=False, random_state=random_state)

    # show size & plot distribution
    print("  Data Size: ", data.shape[0])
    for col_name, col in zip(['source', 'target'], ['length_inp', 'length_tar']):
        plt.hist(data[col], bins=100)
        if plot:
            print(f"  Plot length distribution of {col_name}")
            plt.show()
        else:
            plt.savefig(f'../reports/figures/data-{col_name}.jpg')
        data = data.drop(col, axis=1)

    # preprocess texts & labels
    print("  Preprocess & Transfer from Simplified Chinese to Tranditional Chinese...")
    data = data.applymap(lambda text:preprocessors[lang](cc.convert(text), py_function=True)[0])
    data = data.dropna().reset_index(drop=True)
    
    # split dataset & output
    data_train, data_valid = train_test_split(data, test_size=split, random_state=random_state)
    data_valid, data_test = train_test_split(data_valid, test_size=0.5, random_state=random_state)

    return data_train, data_valid, data_test

config = configparser.ConfigParser()
config.read('../config/model.cfg')

### read configurations

seed = config['basic'].getint('seed')

lang = config['data']['lang']
train_file = config['data']['train_file']

min_inp_len = config['data'].getint('min_inp_len')    
max_inp_len = config['data'].getint('max_inp_len')
min_tar_len = config['data'].getint('min_tar_len')    
max_tar_len = config['data'].getint('max_tar_len')

if __name__ == '__main__':

    # setup size
    train_size = config['data'].getint('train_size')
    test_size = config['data'].getint('test_size')  

    # setup path
    train_from_path = os.path.join(configuration.DIR_DATA, train_file)

    to_path = os.path.join(configuration.DIR_INTERMIN, train_file)
    if not os.path.isdir(to_path):
        os.makedirs(to_path, exist_ok=True)
    train_to_path = os.path.join(to_path, 'train.zip')
    valid_to_path = os.path.join(to_path, 'valid.zip')
    test_to_path = os.path.join(to_path, 'test.zip')

    # preprocess 
    if 'pair' in train_file:
        datasets = preprocess_pair_data(train_from_path, lang, min_inp_len, max_inp_len, min_tar_len, max_tar_len,
                                        size=train_size, split=test_size, random_state=seed)

    train, valid, test = datasets
    
    # save to intermin directory
    train.to_csv(train_to_path, sep=',', index=None)
    valid.to_csv(valid_to_path, sep=',', index=None)
    test.to_csv(test_to_path, sep=',', index=None)
