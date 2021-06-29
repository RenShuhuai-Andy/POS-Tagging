import re
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import random
import os


def data_analysis(file_name):
    label_set = set()
    label_dict = defaultdict(int)
    seq_len = []
    label_pattern = re.compile(r'[A-Za-z]+')
    with open(file_name, 'r', encoding='utf-16') as f:
        for line in f:
            line = line.split('  ')
            seq_len.append(len(line))
            for word in line:
                label = label_pattern.findall(word)
                if len(label) > 0:
                    label = label[0]
                    label_set.add(label)
                    label_dict[label] += 1
    print('for file %s' % file_name)
    print('average sequence length: %d' % (sum(seq_len) / len(seq_len)))
    print('label num: %d' % len(label_set))
    print('label dict:')
    data_type = file_name.split('/')[1].split('_')[0]
    data_path = os.path.join('data', data_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, f'labels.txt'), 'w', encoding='utf-8') as f:
        for key, value in label_dict.items():
            print(key, ': ', value)
            f.write(key + '\n')


def data_split(file_name):
    total_data = []
    with open(file_name, 'r', encoding='utf-16') as f:
        for line in f:
            if line != '\n':
                total_data.append(line)
    random.shuffle(total_data)
    total_len = len(total_data)
    train_len = int(total_len * 0.7)
    dev_len = int(total_len * 0.1)
    test_len = total_len - train_len - dev_len
    print('for file %s' % file_name)
    print('total_len: %d, train_len: %d, dev_len: %d, test_len: %d' % (total_len, train_len, dev_len, test_len))
    train_data, dev_data, test_data = total_data[:train_len], total_data[train_len:train_len + dev_len], total_data[
                                                                                                         train_len + dev_len:]
    data_dict = {'train': train_data, 'dev': dev_data, 'test': test_data}
    data_type = file_name.split('/')[1].split('_')[0]
    data_path = os.path.join('data', data_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    assert len(train_data) == train_len and len(dev_data) == dev_len and len(test_data) == test_len
    for mode in ['train', 'dev', 'test']:
        with open(os.path.join(data_path, f'%s.txt' % mode), 'w', encoding='utf-16') as f:
            data = data_dict[mode]
            for line in data:
                f.write(line)


if __name__ == '__main__':
    for file_name in [r'data/simplified_train_utf16.tag', r"data/traditional_train_utf16.tag"]:
        # data_analysis(file_name)
        data_split(file_name)
