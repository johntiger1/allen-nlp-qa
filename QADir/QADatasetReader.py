'''

For the PubMedQA dataset.

'''

import tempfile
from typing import Dict, Iterable, List, Tuple
from overrides import overrides

import torch

import allennlp
from allennlp.data import DataLoader, DatasetReader, Instance, Vocabulary
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Auc
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate

import pandas as pd
import os
import gc

# for the pubmed qa dataset
import json
import re



'''
We should throw out the X, where X is not good
'''

'''
It is true that we could have attention and such with our regular framework. 
however, allen-nlp may be better for attention, and already implemeneted seq2vec interfaces

new directive: each dataset reader will be responsible for ingesting just one data source?
or: we could leverage the predictors framework for comprehensively?
i.e. do validation on a full set of indices

'''

@DatasetReader.register("PubMedQADatasetReader-json")
class PubMedQADatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 512,
                 listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 train_file: str= None,
                 test_file:    str = None,
                 num_classes: int = 2

    ):
        super().__init__(lazy, max_instances=1000)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()} # is it possible the tokens are no longer identical?
        self.max_tokens = max_tokens
        self.listfile = listfile
        self.notes_dir = notes_dir
        self.neg_samp_prob = 0.2

        '''replicating the subsampling procedure, except iterating through a json now'''
        '''
        
        '''
        self.limit_examples = None
        self.sampler_type = "balanced"
        self.mode = "train"
        self.sampled_idx = {}
        self.train_file = train_file
        self.test_file = test_file

        self.set_sampled_idx("train", self.train_file)
        self.set_sampled_idx("test", self.test_file)

        self.num_classes = num_classes
        self.labels_vocab_mapping = {

            "yes": 0,
            "no": 1,
            "maybe": 2
        }


    # Populates the sampled idx for the given mode specified by name
    def set_sampled_idx(self, name, file_path):
        labels = []

        assert self.num_classes == len(self.labels_vocab_mapping)
        class_counts = np.zeros(self.num_classes) # a bug now!

        self.json_obj = json.load(open(file_path))

        for key,datum in self.json_obj.items():
            ans = datum['final_decision']
            labels.append(ans)
            class_counts[self.labels_vocab_mapping[ans]] += 1


        # now, we assign the weights to ALL the class labels
        class_weights = 1 / class_counts
        # essentially, assign the weights as the ratios, from the self.stats stuff

        all_label_weights = class_weights[
            labels].squeeze()  # produce an array of size labels, but looking up the value in class weights each time
        num_samples = self.limit_examples if self.limit_examples else len(all_label_weights)
        num_samples = min(num_samples, len(all_label_weights))

        if self.args.sampler_type == "balanced":
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=all_label_weights,
                                                                     num_samples=num_samples,
                                                                     replacement=False)
        elif self.args.sampler_type == "random":
            sampler = torch.utils.data.sampler.SubsetRandomSampler(indices=[i for i in range(len(all_label_weights))])
        else:
            # logger.critical("Weird sampler specified \n")
            sampler = None

        self.sampled_idx[name] = list(sampler)[:num_samples]




    # def get_sampler(self, listfile: str = ""):
    #
    # def get_stats(self, file_path: str):
    #     '''
    #
    #     '''
    #     # get stats on the dataset listed at _path_
    #     from collections import defaultdict
    #     self.stats = defaultdict(int)
    #
    #     with open(file_path, "r") as file:
    #         file.readline() # could also pandas readcsv and ignore first line
    #         for line in file:
    #             info_filename, label = line.split(",")
    #             self.stats[int(label)] +=1
    #     return self.stats


    '''
    Internally, the public facing read() method will call this internal _read() method.
    Ensure that instances are only yielded according to the indices sampled. 
    '''
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        '''Expect: JSON array'''


        self.json_obj = json.load(open(file_path))

        for idx,(key,datum) in enumerate(self.json_obj.items()):
            if self.mode == "test" or idx in self.sampled_idx[self.mode]:
                '''preprocessing'''
                ques = datum["QUESTION"]
                ques = re.sub("[^a-zA-Z0-9\s]", "", ques)
                ans = datum['final_decision']
                ans = re.sub("[^a-zA-Z0-9\s]", "", ans)

                long_ans = datum["LONG_ANSWER"]
                long_ans = re.sub("[^a-zA-Z0-9\s]", "", long_ans)


                tokenized_ques = self.tokenizer.tokenize(ques)
                tokenized_ans = self.tokenizer.tokenize(ans)
                tokenized_long_ans = self.tokenizer.tokenize(long_ans)

                text_field = TextField(tokenized_ques, self.token_indexers)

                # answer_vocab = {"yes": 0, "no": 1, "maybe": 2}
                label_field = LabelField(str(tokenized_ans))

                long_answer = TextField(tokenized_long_ans, self.token_indexers)

                # label_field = TextField(tokenized_ans, self.token_indexers)

                fields = {'text': text_field, 'label': label_field, "long_answer": long_answer}

                # '''we also need to pass in whether we are training or predicting here!'''
                # if "No" in tokenized_ans:
                yield Instance(fields)



# setting the lazy flag will help a lot with memory issues
pmqad_reader = PubMedQADatasetReader(lazy=True)


instances = pmqad_reader.read("/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqal.json")

for elt in instances:
    print(elt)

'''i wanna introduce a dataset, at some point'''