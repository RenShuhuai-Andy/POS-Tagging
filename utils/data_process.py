import re
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit, KFold


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
    for key, value in label_dict.items():
        print(key, ': ', value)


def data_split(file_name):
    total_data = []
    with open(file_name, 'r', encoding='utf-16') as f:
        for line in f:
            if line != '\n':
                total_data.append(line)



if __name__ == '__main__':
    for file_name in ['..\data\simplified_train_utf16.tag', r"..\data\traditional_train_utf16.tag"]:
        data_analysis(file_name)