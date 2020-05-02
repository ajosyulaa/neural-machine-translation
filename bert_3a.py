import argparse
import csv
import logging
import os
import random
import sys
import pandas as pd

import numpy as np
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig, BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

############################################### Creating training file ########################################################
with open("dev.de", "r") as ins:
    dev_de = []
    for line in ins:
        dev_de.append(line.strip('\n'))

with open("dev.en", "r") as ins:
    dev_en = []
    for line in ins:
        dev_en.append(line.strip('\n'))

with open("dev_generated.en", "r") as ins:
    devg_en = []
    for line in ins:
        devg_en.append(line.strip('\n'))

src_lines = dev_de + dev_de
        
lines = []
labels = []
gid = 0
guids = []
all_guids = [i for i in range(2*len(dev_en))]

for line in dev_en:
  lines.append(line)
  labels.append('human')
  gind = random.randint(0,len(all_guids) - 1)
  gid = all_guids[gind]
  del all_guids[gind]
  guids.append(gid)

for line in devg_en:
  lines.append(line)
  labels.append('machine')
  gind = random.randint(0,len(all_guids) - 1)
  gid = all_guids[gind]
  del all_guids[gind]
  guids.append(gid)

assert(len(src_lines) == len(lines))
dev_data = {'guid' : guids,'text_a': src_lines ,'text_b': lines,'label':labels}
df_train = pd.DataFrame(dev_data, columns=["guid", "text_a","text_b","label"])
df_train = df_train.sort_values(by=['guid'])
df_train.to_csv("dev_bert_src.csv", index=False)

############################################### Logger ########################################################

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

############################################### Args ########################################################

args = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": 'data',
    "data_dir": 'data',
    "cache_dir" : 'cache',
    "task_name": "toxic_multilabel",
    "no_cuda": False,
    "bert_model": 'bert-base-multilingual-cased',
    "output_dir": 'output',
    "max_seq_length": 150,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": False,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 4.0,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}

############################################### Bert Class ########################################################

class DoubleBertForSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(DoubleBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(2*config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids1, input_ids2, token_type_ids1=None, token_type_ids2=None, attention_mask1=None, attention_mask2=None, labels=None):
        _, pooled_output1 = self.bert(input_ids1, token_type_ids1, attention_mask1, output_all_encoded_layers=False)
        _, pooled_output2 = self.bert(input_ids2, token_type_ids2, attention_mask2, output_all_encoded_layers=False)
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)
        pooled_output = torch.cat((pooled_output1,pooled_output2),dim = -1)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

############################################################# Data prep #######################################################

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids1, input_ids2,input_mask1,input_mask2, segment_ids1,segment_ids2, label_id):
        self.input_ids1 = input_ids1
        self.input_ids2 = input_ids2
        self.input_mask1 = input_mask1
        self.input_mask2 = input_mask2
        self.segment_ids1 = segment_ids1
        self.segment_ids2 = segment_ids2
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

############################################################# 
class MultiLabelTextProcessor(DataProcessor):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None
    
    
    def get_train_examples(self, data_dir, size=-1):
        filename = 'dev_bert_src.csv'
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
        if size == -1:
            data_df = pd.read_csv(os.path.join(data_dir, filename), na_filter = False)
            return self._create_examples(data_df, "train")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename), na_filter = False)
            return self._create_examples(data_df.sample(size), "train")
        
    def get_dev_examples(self, data_dir, size=-1):
        """See base class."""
        filename = 'dev_bert_src.csv'
        if size == -1:
            data_df = pd.read_csv(os.path.join(data_dir, filename), na_filter = False)
            return self._create_examples(data_df, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename), na_filter = False)
            return self._create_examples(data_df.sample(size), "dev")

    def get_labels(self):
        """See base class."""
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(self.data_dir, "classes.txt"),header=None, na_filter = False)[0].values)
        return self.labels

    def _create_examples(self, df, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[1]
            text_b = row[2]
            if labels_available:
                labels = row[3]
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b = text_b, label=labels))
        return examples


########################################################################################

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
#         print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[:(max_seq_length - 2)]
        
        if len(tokens_b) > max_seq_length - 2:
          tokens_b = tokens_b[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens1 = ["[CLS]"] + tokens_a + ["[SEP]"]
        tokens2 = ["[CLS]"] + tokens_b + ["[SEP]"]
        segment_ids1 = [0] * len(tokens1)
        segment_ids2 = [0] * len(tokens2)

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)
        input_mask2 = [1] * len(input_ids2)

        # Zero-pad up to the sequence length.
        padding1 = [0] * (max_seq_length - len(input_ids1))
        input_ids1 += padding1
        input_mask1 += padding1
        segment_ids1 += padding1
        
        # Zero-pad up to the sequence length.
        padding2 = [0] * (max_seq_length - len(input_ids2))
        input_ids2 += padding2
        input_mask2 += padding2
        segment_ids2 += padding2

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length
        assert len(segment_ids1) == max_seq_length
        
        assert len(input_ids2) == max_seq_length
        assert len(input_mask2) == max_seq_length
        assert len(segment_ids2) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids1=input_ids1,input_ids2=input_ids2,
                              input_mask1=input_mask1,input_mask2=input_mask2,
                              segment_ids1=segment_ids1,segment_ids2=segment_ids2,
                              label_id=label_id))
    return features


#########################################################################################

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(task_name, preds, labels):
  return {"acc": simple_accuracy(preds, labels)}

############################################################################################

processors = {
    "toxic_multilabel": MultiLabelTextProcessor
}

# Setup GPU parameters

if args["local_rank"] == -1 or args["no_cuda"]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    n_gpu = torch.cuda.device_count()
#     n_gpu = 1
else:
    torch.cuda.set_device(args['local_rank'])
    device = torch.device("cuda", args['local_rank'])
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))

#############################################################################################

args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

task_name = args['task_name'].lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name](args['data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)

#############################################################################################

tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])

train_examples = None
num_train_optimization_steps = None
if args['do_train']:
    train_examples = processor.get_train_examples(args['full_data_dir'], size=args['train_size'])
#     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_optimization_steps = int(
        len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])


cache_dir = args['cache_dir']
model = DoubleBertForSequenceClassification.from_pretrained(args["bert_model"],
              cache_dir=cache_dir,
              num_labels=num_labels)

if args['fp16']:
    model.half()
model.to(device)

if args['local_rank'] != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)


#######################################################################################################

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
t_total = num_train_optimization_steps

optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         t_total=t_total)

##########################################################################################################

global_step = 0
nb_tr_steps = 0
tr_loss = 0
output_mode = "classification"
if args["do_train"]:
  train_features = convert_examples_to_features(
            train_examples, label_list, args["max_seq_length"], tokenizer)
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_examples))
  logger.info("  Batch size = %d", args["train_batch_size"])
  logger.info("  Num steps = %d", num_train_optimization_steps)
  
  all_input_ids1 = torch.tensor([f.input_ids1 for f in train_features], dtype=torch.long)
  all_input_mask1 = torch.tensor([f.input_mask1 for f in train_features], dtype=torch.long)
  all_segment_ids1 = torch.tensor([f.segment_ids1 for f in train_features], dtype=torch.long)
  
  all_input_ids2 = torch.tensor([f.input_ids2 for f in train_features], dtype=torch.long)
  all_input_mask2 = torch.tensor([f.input_mask2 for f in train_features], dtype=torch.long)
  all_segment_ids2 = torch.tensor([f.segment_ids2 for f in train_features], dtype=torch.long)
  
  all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
  train_data = TensorDataset(all_input_ids1,all_input_ids2, all_input_mask1,all_input_mask2, all_segment_ids1, all_segment_ids2, all_label_ids)
  
  if args["local_rank"] == -1:
    train_sampler = RandomSampler(train_data)
  else:
    train_sampler = DistributedSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args["train_batch_size"])

  model.train()
  for _ in trange(int(args["num_train_epochs"]), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
      batch = tuple(t.to(device) for t in batch)
      input_ids1, input_ids2, input_mask1, input_mask2, segment_ids1, segment_ids2, label_ids = batch
      
      # define a new function to compute loss values for both output_modes
      logits = model(input_ids1, input_ids2, segment_ids1, segment_ids2, input_mask1, input_mask2, labels=None)
      
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

      if n_gpu > 1:
        loss = loss.mean() # mean() to average on multi-gpu.
      if args["gradient_accumulation_steps"] > 1:
        loss = loss / args["gradient_accumulation_steps"]


      loss.backward()

      tr_loss += loss.item()
      nb_tr_examples += input_ids1.size(0)
      nb_tr_steps += 1
      if (step + 1) % args["gradient_accumulation_steps"] == 0:
        if args["fp16"]:
          # modify learning rate with special warm up BERT uses
          # if args["fp16"] is False, BertAdam is used that handles this automatically
          lr_this_step = args["learning_rate"] * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                                                                                 args["warmup_proportion"])
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_step
          
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1

if args["do_train"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
  # Save a trained model, configuration and tokenizer
  model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
  
  # If we save using the predefined names, we can load using `from_pretrained`
  output_model_file = os.path.join(args["output_dir"], WEIGHTS_NAME)
  output_config_file = os.path.join(args["output_dir"], CONFIG_NAME)
  
  torch.save(model_to_save.state_dict(), output_model_file)
  model_to_save.config.to_json_file(output_config_file)
  tokenizer.save_vocabulary(args["output_dir"])
  
  # Load a trained model and vocabulary that you have fine-tuned
#   model = BertForSequenceClassification.from_pretrained(args["output_dir"], num_labels=num_labels)
#   tokenizer = BertTokenizer.from_pretrained(args["output_dir"], do_lower_case=args["do_lower_case"])
else:
  model = DoubleBertForSequenceClassification.from_pretrained(args["bert_model"], num_labels=num_labels)
  model.to(device)

########################################################################################################

if args["do_eval"] and (args["local_rank"] == -1 or torch.distributed.get_rank() == 0):
  eval_examples = processor.get_dev_examples(args["data_dir"])
  eval_features = convert_examples_to_features(
      eval_examples, label_list, args["max_seq_length"], tokenizer)
  logger.info("***** Running evaluation *****")
  logger.info("  Num examples = %d", len(eval_examples))
  logger.info("  Batch size = %d", args["eval_batch_size"])
  all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

  all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  
  # Run prediction for full data
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args["eval_batch_size"])
  
  model.eval()
  eval_loss = 0
  nb_eval_steps = 0
  preds = []
  
  for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)
    
    with torch.no_grad():
      logits = model(input_ids, segment_ids, input_mask, labels=None)
    
    # create eval loss and other metric required by the task
    loss_fct = CrossEntropyLoss()
    tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    
    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
      preds.append(logits.detach().cpu().numpy())
    else:
      preds[0] = np.append(
          preds[0], logits.detach().cpu().numpy(), axis=0)
  
  eval_loss = eval_loss / nb_eval_steps
  preds = preds[0]
  preds = np.argmax(preds, axis=1)
  
  result = compute_metrics(task_name, preds, all_label_ids.numpy())
  loss = tr_loss/global_step if args["do_train"] else None
  
  result['eval_loss'] = eval_loss
  result['global_step'] = global_step
  result['loss'] = loss
  
  output_eval_file = os.path.join(args["output_dir"], "eval_results.txt")
  with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
      logger.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))