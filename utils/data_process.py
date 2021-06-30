import re
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit, KFold
import random
import os


def data_analysis(file_name):
    label_set = set()
    label_dict = defaultdict(int)
    seq_len = []
    with open(file_name, 'r', encoding='utf-16') as f:
        for line in f:
            if line == '\n':
                continue
            line = line.strip().split('  ')
            sl = 0
            # seq_len.append(len(line))
            for word in line:
                tokens_label = word.split('/')
                tokens = tokens_label[0]
                label = tokens_label[1]
                label_set.add('B-' + label)
                label_dict['B-' + label] += 1
                sl += len(list(tokens))
                if len(list(tokens)) > 1:
                    label_set.add('I-' + label)
                    label_dict['I-' + label] += len(list(tokens)) - 1
            seq_len.append(sl)
    print('for file %s' % file_name)
    print('average sequence length: %d' % (sum(seq_len) / len(seq_len)))
    print('max sequence length: %d' % max(seq_len))
    print('label num: %d' % len(label_set))
    print('label dict:')
    data_type = file_name.split('/')[1].split('_')[0]
    data_path = os.path.join('data/final_eval', data_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(os.path.join(data_path, f'labels.txt'), 'w', encoding='utf-8') as f:
        for key, value in label_dict.items():
            print(key, ': ', value)
            f.write(key + '\n')


def test_data_process(file_name):
    def n_split(tmp_list, n):
        for i in range(0, len(tmp_list), n):
            yield tmp_list[i:i + n]

    data_type = file_name.split('/')[1].split('_')[0]
    data_path = os.path.join('data/final_eval', data_type)
    seq_len = []
    with open(file_name, 'r', encoding='utf-16') as f, \
            open(os.path.join(data_path, f'test.txt'), 'w', encoding='utf-16') as f1, \
            open(os.path.join(data_path, f'test_split_info.txt'), 'w') as f2:
        line_idx = 0
        for line in f:
            if line == '\n':
                continue
            tokenized_line = line.strip().split('  ')
            sl = 0
            for word in tokenized_line:
                sl += len(list(word))
            seq_len.append(sl)
            if sl >= 128-3:
                split_lines = list(n_split(tokenized_line, 40))
                for split_line in split_lines:
                    f1.write('  '.join(split_line) + '\n')
                line_idx_tmp = list(range(line_idx, line_idx + len(split_lines)))
                line_idx_tmp = [str(x) for x in line_idx_tmp]
                f2.write(' '.join(line_idx_tmp) + '\n')  # idx of lines that belongs to a long original sentence
                line_idx += len(split_lines)
            else:
                f1.write(line)
                line_idx += 1

    print('average sequence length: %d' % (sum(seq_len) / len(seq_len)))
    print('max sequence length: %d' % max(seq_len))


def data_split(file_name):
    total_data = []
    with open(file_name, 'r', encoding='utf-16') as f:
        for line in f:
            if line != '\n':
                total_data.append(line)
    random.shuffle(total_data)
    total_len = len(total_data)
    train_len = int(total_len * 0.9)
    dev_len = total_len - train_len
    print('for file %s' % file_name)
    print('total_len: %d, train_len: %d, dev_len: %d' % (total_len, train_len, dev_len))
    train_data, dev_data = total_data[:train_len], total_data[train_len:]

    data_dict = {'train': train_data, 'dev': dev_data}
    data_type = file_name.split('/')[1].split('_')[0]
    data_path = os.path.join('data/final_eval', data_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    assert len(train_data) == train_len and len(dev_data) == dev_len

    for mode in ['train', 'dev']:
        with open(os.path.join(data_path, f'%s.txt' % mode), 'w', encoding='utf-16') as f:
            data = data_dict[mode]
            for line in data:
                f.write(line)


if __name__ == '__main__':
    # for file_name in [r'data/simplified_train_utf16.tag', r"data/traditional_train_utf16.tag"]:
    #     data_analysis(file_name)
        # data_split(file_name)
    for file_name in [r'data/simplified_test_utf16.tag', r"data/traditional_test_utf16.tag"]:
        test_data_process(file_name)
