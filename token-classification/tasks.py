import logging
import os
from typing import List, TextIO, Union

from utils_pos import InputExample, Split, TokenClassificationTask


logger = logging.getLogger(__name__)


class NER(TokenClassificationTask):
    def __init__(self, label_idx=-1):
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-16") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[self.label_idx].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class Chunk(NER):
    def __init__(self):
        # in CONLL2003 dataset chunk column is second-to-last
        super().__init__(label_idx=-2)

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return [
                "O",
                "B-ADVP",
                "B-INTJ",
                "B-LST",
                "B-PRT",
                "B-NP",
                "B-SBAR",
                "B-VP",
                "B-ADJP",
                "B-CONJP",
                "B-PP",
                "I-ADVP",
                "I-INTJ",
                "I-LST",
                "I-PRT",
                "I-NP",
                "I-SBAR",
                "I-VP",
                "I-ADJP",
                "I-CONJP",
                "I-PP",
            ]


class POS(TokenClassificationTask):
    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        if mode == 'test':
            with open(os.path.join(data_dir, f"labels.txt")) as f:
                tmp_label = f.readlines()[0].strip()

        with open(file_path, encoding="utf-16") as f:
            for sentence in f:
                tokens = []
                labels = []
                for words in sentence.strip().split('  '):
                    if mode == 'test':
                        tmp_tokens = list(words)
                        tmp_labels = [tmp_label] * len(tmp_tokens)
                        tokens.extend(tmp_tokens)
                        labels.extend(tmp_labels)
                    else:
                        tokens_labels = words.split('/')
                        tmp_tokens = list(tokens_labels[0])
                        tmp_labels = ['B-' + tokens_labels[1]] + ['I-' + tokens_labels[1]] * (len(tmp_tokens) - 1)
                        tokens.extend(tmp_tokens)
                        labels.extend(tmp_labels)
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=tokens, labels=labels))
                guid_index += 1
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List, test_examples, test_split_infos):
        infos_idx = 0
        for i, sentence in enumerate(test_input_reader):
            labels = []
            tokenized_sentence = test_examples[i].words
            predicted_labels = preds_list[i]
            assert len(tokenized_sentence) == len(predicted_labels)
            for words in sentence.strip().split('  '):
                gt_tokens_len = len(list(words))
                label = [label.strip('B-').strip('I-') for label in predicted_labels[:gt_tokens_len]]
                labels.append(max(label, key=label.count))
                predicted_labels = predicted_labels[gt_tokens_len:]  # remove previous gt_tokens_len label
            assert len(sentence.strip().split('  ')) == len(labels)
            out = '  '.join([words + '/' + label for words, label in zip(sentence.strip().split('  '), labels)])
            writer.write(out)
            if infos_idx < len(test_split_infos) and str(i) in test_split_infos[infos_idx]:
                if str(i) != test_split_infos[infos_idx][-1]:
                    writer.write('  ')
                else:
                    infos_idx += 1
                    writer.write('\n')
            else:
                writer.write('\n')

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                return f.read().splitlines()
        else:
            return [
                "ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SCONJ",
                "SYM",
                "VERB",
                "X",
            ]