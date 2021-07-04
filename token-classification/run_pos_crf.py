# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
import torch
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from losses import DiceLoss, FocalLoss, LabelSmoothingCrossEntropy, CourageLoss
from torch.nn import CrossEntropyLoss
from models import BertCrfForPos
import collections
import warnings
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import (
    PredictionOutput,
    set_seed,
)
from torch.utils.data.dataloader import DataLoader
from transformers.file_utils import is_torch_tpu_available
from transformers.optimization import AdamW
from utils_pos import Split, TokenClassificationDataset, TokenClassificationTask
from evaluation_util import *
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    nested_concat,
    nested_detach,
)
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
logger = logging.getLogger(__name__)


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    tags: np.ndarray


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="POS", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    loss_type: Optional[str] = field(
        default='CrossEntropyLoss', metadata={"help": "The loss used during training."}
    )
    loss_gamma: Optional[float] = field(
        default=2, metadata={"help": "The gamma of Focal loss etc."}
    )
    crf_learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for crf."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


class TrainerWithSpecifiedLoss(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, compute_metrics, loss_type, loss_gamma, optimizers):
        Trainer.__init__(self, model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                         compute_metrics=compute_metrics, optimizers=optimizers)
        if loss_type == 'DiceLoss':
            self.loss_fct = DiceLoss()
        elif loss_type == 'FocalLoss':
            self.loss_fct = FocalLoss(gamma=loss_gamma)
        elif loss_type == 'LabelSmoothingCrossEntropy':
            self.loss_fct = LabelSmoothingCrossEntropy()
        elif loss_type == 'CrossEntropyLoss':
            self.loss_fct = CrossEntropyLoss()
        elif loss_type == 'CourageLoss':
            self.loss_fct = CourageLoss(gamma=loss_gamma)
        else:
            raise ValueError("Doesn't support such loss type")

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     outputs = model(**inputs)
    #     logits = outputs[1]  # [bsz, max_token_len, class_num]
    #     labels = inputs['labels']  # [bsz, max_token_len]
    #     attention_mask = inputs['attention_mask']  # [bsz, max_token_len]
    #     loss = None
    #     if labels is not None:
    #         num_labels = model.module.num_labels if isinstance(model, torch.nn.DataParallel) else model.num_labels
    #         # Only keep active parts of the loss
    #         if attention_mask is not None:
    #             active_loss = attention_mask.view(-1) == 1
    #             active_logits = logits.view(-1, num_labels)  # [bsz * max_token_len, class_num]
    #             active_labels = torch.where(
    #                 active_loss, labels.view(-1), torch.tensor(self.loss_fct.ignore_index).type_as(labels)
    #             )  # [bsz * max_token_len]
    #             loss = self.loss_fct(active_logits, active_labels)
    #         else:
    #             loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if hasattr(self, "_prediction_loop"):
            warnings.warn(
                "The `_prediction_loop` method is deprecated and won't be called in a future version, define `prediction_loop` in your subclass.",
                FutureWarning,
            )
            return self._prediction_loop(dataloader, description, prediction_loss_only=prediction_loss_only)

        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        tags_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
        labels_gatherer = DistributedTensorGatherer(world_size, num_examples)
        tags_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels, tags = self.prediction_step(model, inputs, prediction_loss_only)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, dim=0)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, dim=0)
            if tags is not None:
                tags_host = tags if tags_host is None else nested_concat(tags_host, tags, dim=0)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
                tags_gatherer.add_arrays(self._gather_and_numpify(tags_host, "eval_tags"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, tags_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
        labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))
        tags_gatherer.add_arrays(self._gather_and_numpify(tags_host, "eval_tags"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize()
        label_ids = labels_gatherer.finalize()
        tags = tags_gatherer.finalize()

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids, tags=tags))
        else:
            metrics = {}

        if eval_loss is not None:
            metrics["eval_loss"] = eval_loss.mean().item()

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            if has_labels:
                loss = outputs[0].mean().detach()
                logits = outputs[1:]
            else:
                loss = None
                # Slicing so we get a tuple even if `outputs` is a `ModelOutput`.
                logits = outputs[:]
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]
                # Remove the past from the logits.
                logits = logits[: self.args.past_index - 1] + logits[self.args.past_index :]

        if prediction_loss_only:
            return (loss, None, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        crf = model.module.crf if isinstance(model, torch.nn.DataParallel) else model.crf
        tags = crf.decode(logits, inputs['attention_mask']).squeeze(0)

        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return (loss, logits, labels, tags)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)
    logger.info("num labels %d", num_labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = BertCrfForPos.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': training_args.weight_decay, 'lr': training_args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': training_args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': training_args.weight_decay, 'lr': model_args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': model_args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': training_args.weight_decay, 'lr': model_args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': model_args.crf_learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon)

    # Get datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(tags: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = tags  # actually return tags in bert_crf

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.tags, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list) * 100,
            "precision": precision_score(out_label_list, preds_list) * 100,
            "recall": recall_score(out_label_list, preds_list) * 100,
            "f1": f1_score(out_label_list, preds_list) * 100,
        }

    # Initialize our Trainer
    trainer = TrainerWithSpecifiedLoss(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        loss_type=model_args.loss_type,
        loss_gamma=model_args.loss_gamma,
        optimizers=[optimizer, None]
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %.2f", key, value)
                    writer.write("%s = %.2f\n" % (key, value))

            results.update(result)

        # log config
        with open(output_eval_file, "a") as writer:
            hyper_params = str(
                torch.load(open(os.path.join(training_args.output_dir, "training_args.bin"), 'rb'))).split(',')
            for i, para in enumerate(hyper_params):
                if i == 0:
                    para = para.replace("TrainingArguments(", '')
                if i == len(hyper_params) - 1:
                    para = para.replace(")", '')
                para = para.replace(' ', '')
                writer.write(para + '\n')

        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        result = trainer.evaluate(test_dataset)

        if trainer.is_world_master():
            logger.info("***** Test results *****")
            for key, value in result.items():
                logger.info("  %s = %.2f", key, value)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()