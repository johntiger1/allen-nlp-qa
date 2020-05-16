'''

For the PubMedQA dataset.

Don't feel like I'm working enough!

Practice some eigenvalue and such problems

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
'''


class PubMedQADatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = 512,
                 listfile: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/in-hospital-mortality/train/listfile.csv",
                 notes_dir: str = "/scratch/gobi1/johnchen/new_git_stuff/multimodal_fairness/data/extracted_notes",
                 ):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.listfile = listfile
        self.notes_dir = notes_dir

    def get_stats(self, file_path: str):
        '''

        '''
        # get stats on the dataset listed at _path_
        from collections import defaultdict
        self.stats = defaultdict(int)

        with open(file_path, "r") as file:
            file.readline() # could also pandas readcsv and ignore first line
            for line in file:
                info_filename, label = line.split(",")
                self.stats[int(label)] +=1
        return self.stats


    '''
    Internally, the public facing read() method will call this internal _read() method
    '''
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        '''Expect: JSON array'''


        self.json_obj = json.load(open(file_path))

        for key,datum in self.json_obj.items():

            '''preprocessing'''
            ques = datum["QUESTION"]
            ques = re.sub("[^a-zA-Z0-9\s]", "", ques)
            ans = datum['final_decision']
            ans = re.sub("[^a-zA-Z0-9\s]", "", ans)


            tokenized_ques = self.tokenizer.tokenize(ques)
            tokenized_ans = self.tokenizer.tokenize(ans)
            text_field = TextField(tokenized_ques, self.token_indexers)

            answer_vocab = {"yes": 0, "no": 1, "maybe": 2}
            label_field = LabelField(str(tokenized_ans))
            # label_field = TextField(tokenized_ans, self.token_indexers)

            fields = {'text': text_field, 'label': label_field}
            yield Instance(fields)

# setting the lazy flag will help a lot with memory issues
pmqad_reader = PubMedQADatasetReader(lazy=True)


instances = pmqad_reader.read("/scratch/gobi1/johnchen/new_git_stuff/lxmert/standalone_seq2seq/data/ori_pqal.json")

for elt in instances:
    print(elt)

'''i wanna introduce a dataset, at some point'''