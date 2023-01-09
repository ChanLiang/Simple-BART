# Copyright 2021 Haoyu Song
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import os

from argparse import ArgumentParser
from xlibs import BartTokenizer

from dataloader import read_convai2_split
from dataloader import read_ecdt2019_split


def preprocess(args):
    train_persona, train_query, train_response = read_convai2_split(args.trainset, multi_turn=args.multi_turn, permutaion_id=args.permutation) if args.dataset_type=='convai2' else read_ecdt2019_split(args.trainset, split_type='train')

    assert len(train_persona) == len(train_query) == len(train_response)
    # print (train_persona[3])
    # print (train_query[3])
    # print (train_response[3])

    tokenizer = BartTokenizer.from_pretrained(args.encoder_model_name_or_path)

    
    train_persona_tokenized = tokenizer(train_persona,
                                       truncation=True,
                                       padding=True,
                                       max_length=args.max_source_length)
    train_persona_tokenized = {
            key: val
            for key, val in train_persona_tokenized.items()
        }

    
    train_query_tokenized = tokenizer(train_query,
                                     truncation=True,
                                     padding=True,
                                     max_length=args.max_source_length)
    train_query_tokenized = {
            key: val
            for key, val in train_query_tokenized.items()
        }

    
    train_response_tokenized = tokenizer(train_response,
                                        truncation=True,
                                        padding=True,
                                        max_length=args.max_target_length)
    train_response_tokenized = {
            key: val
            for key, val in train_response_tokenized.items()
        }

        
    if args.dataset_type=='convai2':
        path = './data/ConvAI2/convai2_tokenized/' 
        if args.multi_turn != -1:
            path = './data/ConvAI2/convai2_tokenized_multi_turn_segtoken/'
        if args.permutation != -1:
            path = f'./data/ConvAI2/permutations/convai2_tokenized_multi_turn_segtoken_permutation_{args.permutation}/'
    else:
        path = './data/ECDT2019/ecdt2019_tokenized/'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    # print(f"Saving tokenized dict at {path}")
    
    with open(path+'train_persona.json','w') as train_persona:
        # print(len(train_persona_tokenized['input_ids']))
        json.dump(train_persona_tokenized, train_persona)

    with open(path+'train_query.json','w') as train_query:
        # print(len(train_query_tokenized['input_ids']))
        json.dump(train_query_tokenized, train_query)
   
    with open(path+'train_response.json','w') as train_response:
        # print(len(train_response_tokenized['input_ids']))
        json.dump(train_response_tokenized, train_response)
    
if __name__ == "__main__":
    parser = ArgumentParser("Transformers EncoderDecoderModel Preprocessing")

    parser.add_argument(
        "--trainset",
        type=str)
    

    parser.add_argument("--train_valid_split", type=float, default=0.1)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--multi_turn", type=int, default=-1)
    parser.add_argument("--permutation", type=int, default=-1)
    parser.add_argument("--encoder_model_name_or_path", type=str)

    parser.add_argument("--dataset_type",
                        type=str,
                        default='convai2',
                        required=True)  # convai2, ecdt2019

    args = parser.parse_args()

    preprocess(args)
