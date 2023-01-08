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


import torch
import random as rd
import json

from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader

from xlibs import AdamW
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import math

from xlibs import BartForConditionalGeneration
from xlibs import BartTokenizer

from dataloader import ConvAI2Dataset
from dataloader import ECDT2019Dataset
from dataloader import NLIDataset
from evaluations import eval_distinct

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    CUDA_AVAILABLE = True
    print("CUDA IS AVAILABLE")
else:
    print("CUDA NOT AVAILABLE")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_tokenier_and_model(tokenizer, model):
    ########## set special tokens
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = 32
    model.config.min_length = 3
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 1.0
    model.config.num_beams = 1
    model.config.temperature = 0.95
    model.config.output_hidden_states = True
    return tokenizer, model


def prepare_data_batch(batch):
    # print (batch['persona'].keys()) # ['input_ids', 'attention_mask']
    persona_input_ids = batch['persona']['input_ids']
    persona_attention_mask = batch['persona']['attention_mask']
    query_input_ids = batch['query']['input_ids']
    query_attention_mask = batch['query']['attention_mask']

    input_ids = torch.cat([persona_input_ids, query_input_ids], -1)
    attention_mask = torch.cat([persona_attention_mask, query_attention_mask], -1)

    decoder_input_ids = batch['response']['input_ids']
    decoder_attention_mask = batch['response']['attention_mask']

    mask_flag = torch.Tensor.bool(1 - decoder_attention_mask)
    lables = decoder_input_ids.masked_fill(mask_flag, -100)

    # print ('=' * 50)
    # # input_ids: torch.Size([32, 126])，persona和context都做了padding，所以中间有一堆0...
    # # 「BART字典有点反直觉」：bart的padding是0，bos是2，padding是1？？？
    # print ('input_ids = ', input_ids.shape, input_ids[0]) 
    # # attention_mask: 标注了中间，最后哪些位置是无效的（0无效，1有效）
    # print ('attention_mask = ', attention_mask.shape, attention_mask[0])
    # # type_ids: （只有0,1）1表示persona，0表示context；（包括padding位置）
    # # print ('type_ids = ', type_ids.shape, type_ids[0])

    # # decoder_input_ids: torch.Size([32, 32])，padding在右边；
    # print ('decoder_input_ids = ', decoder_input_ids.shape, decoder_input_ids[0])
    # # （0无效，1有效）
    # print ('decoder_attention_mask = ', decoder_attention_mask.shape, decoder_attention_mask[0])
    # # mask_flag: attention_mask=0的位置为True;
    # print ('mask_flag = ', mask_flag.shape, mask_flag[0])
    # # 和 decoder_input_ids 一模一样，没有错位；
    # # label的无效位置弄成-100，BART能识别吗？
    # print ('lables = ', lables.shape, lables[0])

    return input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids



def train(args):
    # Model
    print("\nInitializing Model...\n")
    if args.load_checkpoint:
        model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    else:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print ('more than one gpu...')
    #     device = torch.device(f'cuda')
    #     model = model.to(device)
    #     model = torch.nn.parallel.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model.train()
    print (model)
    print ('the number of trainable parameters: ', str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    print("Load tokenized data...\n")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    # Tokenize & Batchify
    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        try:
            print(f"Load tokenized train & val dataset from {args.dumped_token}.")
            path = args.dumped_token

            with open(path + 'train_persona.json') as train_persona, open(path + 'val_persona.json') as val_persona:
                tmp = train_persona.readline()
                train_persona_tokenized = json.loads(tmp)
                tmp = val_persona.readline()
                val_persona_tokenized = json.loads(tmp)

            with open(path + 'train_query.json') as train_query, open(path + 'val_query.json') as val_query:
                tmp = train_query.readline()
                train_query_tokenized = json.loads(tmp)
                tmp = val_query.readline()
                val_query_tokenized = json.loads(tmp)

            with open(path + 'train_response.json') as train_response, open(path + 'val_response.json') as val_response:
                tmp = train_response.readline()
                train_response_tokenized = json.loads(tmp)
                tmp = val_response.readline()
                val_response_tokenized = json.loads(tmp)

        except FileNotFoundError:
            print(f"Sorry! The files in {args.dumped_token} can't be found.")
            raise ValueError

    tokenizer, model = set_tokenier_and_model(tokenizer, model)

    train_dataset = ConvAI2Dataset(train_persona_tokenized,
                                   train_query_tokenized,
                                   train_response_tokenized,
                                   device) if args.dataset_type == 'convai2' else ECDT2019Dataset(
        train_persona_tokenized, train_query_tokenized, train_response_tokenized, device)
    val_dataset = ConvAI2Dataset(val_persona_tokenized,
                                 val_query_tokenized,
                                 val_response_tokenized,
                                 device) if args.dataset_type == 'convai2' else ECDT2019Dataset(val_persona_tokenized,
                                                                                                val_query_tokenized,
                                                                                                val_response_tokenized,
                                                                                                device)

    # Training
    print("\nStart Training...")
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)


    # optim_warmup = AdamW(model.parameters(), lr=args.warm_up_learning_rate)
    # optim = AdamW(model.parameters(), lr=args.learning_rate)
    args.total_optim_steps = (len(train_dataset) // args.batch_size) * args.total_epochs
    print ('total_optim_steps = ', args.total_optim_steps)
    optim = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warm_up_steps, num_training_steps=args.total_optim_steps)
    optim.zero_grad()

    step = 0
    start_epoch = int(args.checkpoint.split("_")[-1]) if args.load_checkpoint else 0
    for epoch in range(start_epoch, args.total_epochs):
        print('\nTRAINING EPOCH %d' % epoch)
        batch_n = 0

        for batch in train_loader:
            batch_n += 1
            step += 1
            # optim_warmup.zero_grad()
            optim.zero_grad()


            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                batch)

            outputs = model(input_ids=input_ids, # 包括 persona_input_ids 吗？
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=lables,
                            return_dict=True,
                            )
            loss = outputs.loss
            ppl = math.exp(loss.item())
           
            loss_prt = loss.cpu().detach().numpy() if CUDA_AVAILABLE else loss.detach().numpy()
            loss_prt = round(float(loss_prt),3)
            ppl_prt = round(float(ppl),4)
            lr = optim.param_groups[0]['lr']

            if step <= args.warm_up_steps:
                if step % 100 == 0:
                    print(f"warm up step {step}\tlr: {lr}\tloss: {loss_prt}\tppl: {ppl_prt}")
            else:
                if step % 100 == 0:
                    print(f"train step {step}\tlr: {lr}\tloss: {loss_prt}\tppl: {ppl_prt}")

            loss.backward()
            scheduler.step()
            optim.step()


            if step % args.print_frequency == 0 and not step <= args.warm_up_steps and not args.print_frequency == -1:
                print('Sampling (not final results) ...')
                model.eval()
                for val_batch in val_loader:

                    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                        val_batch)

                    generated = model.generate(input_ids,
                                               attention_mask=attention_mask,
                                               per_input_ids=persona_input_ids)
                    
                    generated_token = tokenizer.batch_decode(
                        generated, skip_special_tokens=True)[-5:]
                   
                    query_token = tokenizer.batch_decode(
                        query_input_ids, skip_special_tokens=True)[-5:]

                    gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                        skip_special_tokens=True)[-5:]
                    persona_token = tokenizer.batch_decode(
                        persona_input_ids, skip_special_tokens=True)[-5:]

                    if rd.random() < 0.6:
                        for p, q, g, j, k in zip(persona_token, query_token, gold_token, generated_token):
                            print(
                                f"persona: {p[:150]}\nquery: {q[:100]}\ngold: {g[:100]}\nresponse from D1: {j[:100]}\n")
                        break
                print('\nTRAINING EPOCH %d\n' % epoch)
                model.train()

        if not step <= args.warm_up_steps:
            print(f'Saving model at epoch {epoch} step {step}')
            model.save_pretrained(f"{args.save_model_path}_%d" % epoch)


def predict(args):
    print("Load tokenized data...\n")
    tokenizer = BartTokenizer.from_pretrained(args.encoder_model)

    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            print(f"Load tokenized dataset from {args.dumped_token}.")

            # Loading testset
            with open(path + 'test_persona.json') as test_persona:
                print("Load test_persona")
                tmp = test_persona.readline()
                test_persona_tokenized = json.loads(tmp)
            with open(path + 'test_query.json') as test_query:
                print("Load test_query")
                tmp = test_query.readline()
                test_query_tokenized = json.loads(tmp)
            with open(path + 'test_response.json') as test_response:
                print("Load test_response")
                tmp = test_response.readline()
                test_response_tokenized = json.loads(tmp)

        except FileNotFoundError:
            print(f"Sorry! The files in {args.dumped_token} can't be found.")

    print ()
    test_dataset = ConvAI2Dataset(test_persona_tokenized,
                                  test_query_tokenized,
                                  test_response_tokenized,
                                  device) if args.dataset_type == 'convai2' else ECDT2019Dataset(test_persona_tokenized,
                                                                                                 test_query_tokenized,
                                                                                                 test_response_tokenized,                                                                                             device)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Loading Model
    if args.dataset_type == 'convai2':
        model_path = f"./checkpoints/ConvAI2/bertoverbert_{args.eval_epoch}"
        # model_path = f"./checkpoints/ConvAI2_lex/bertoverbert_{args.eval_epoch}"
    elif args.dataset_type == 'ecdt2019':
        model_path = f"./checkpoints/ECDT2019/bertoverbert_ecdt_{args.eval_epoch}"
    else:
        print(f"Invalid dataset_type {args.dataset_type}")
        raise (ValueError)
    print("Loading Model from %s" % model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    tokenizer, model = set_tokenier_and_model(tokenizer, model)

    print(f"Writing generated results to {args.save_result_path}...")

    with open(args.save_result_path, "w", encoding="utf-8") as outf:
        for test_batch in tqdm(test_loader):
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                test_batch)

            generated = model.generate(input_ids,
                                       attention_mask=attention_mask,
                                       num_beams=args.beam_size,
                                       length_penalty=args.length_penalty,
                                       min_length=args.min_length,
                                       no_repeat_ngram_size=args.no_repeat_ngram_size)
            # print ('generated')
            # print (generated)

            # for resp_ids in generated_2:
            #     print ('resp_ids = ', resp_ids)
            #     print ('decoded = ', tokenizer.decode(resp_ids))

            generated_token = tokenizer.batch_decode(
                generated, skip_special_tokens=True)
            
            query_token = tokenizer.batch_decode(
                query_input_ids, skip_special_tokens=True)
                
            gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                skip_special_tokens=True)
            persona_token = tokenizer.batch_decode(
                persona_input_ids, skip_special_tokens=True)

            
            for p, q, g, r, r2 in zip(persona_token, query_token, gold_token, generated_token):
                outf.write(f"persona:{p}\tquery:{q}\tgold:{g}\tresponse_from_d1:{r}\n")


def evaluation(args):
    print("Load tokenized data...\n")
    tokenizer = BartForConditionalGeneration.from_pretrained(args.encoder_model)

    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            print(f"Load tokenized dataset from {args.dumped_token}.")
            with open(path + 'test_persona.json') as test_persona:
                print("Load test_persona")
                tmp = test_persona.readline()
                test_persona_tokenized = json.loads(tmp)
            with open(path + 'test_query.json') as test_query:
                print("Load test_query")
                tmp = test_query.readline()
                test_query_tokenized = json.loads(tmp)
            with open(path + 'test_response.json') as test_response:
                print("Load test_response")
                tmp = test_response.readline()
                test_response_tokenized = json.loads(tmp)

        except FileNotFoundError:
            print(f"Sorry! The file {args.dumped_token} can't be found.")

    test_dataset = ConvAI2Dataset(test_persona_tokenized,
                                  test_query_tokenized,
                                  test_response_tokenized,
                                  device) if args.dataset_type == 'convai2' else ECDT2019Dataset(test_persona_tokenized,
                                                                                                 test_query_tokenized,
                                                                                                 test_response_tokenized,
                                                                                                 device)

    ppl_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Loading Model
    if args.dataset_type == 'convai2':
        # model_path = f"./checkpoints/ConvAI2/bertoverbert_{args.eval_epoch}"
        model_path = f"./checkpoints/ConvAI2_lex/bertoverbert_{args.eval_epoch}"

    elif args.dataset_type == 'ecdt2019':
        model_path = f"./checkpoints/ECDT2019/bertoverbert_ecdt_{args.eval_epoch}"
    else:
        print(f"Invalid dataset_type {args.dataset_type}")
        raise (ValueError)

    print("Loading Model from %s" % model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer, model = set_tokenier_and_model(tokenizer, model)
    model.to(device)
    model.eval()

    print('Evaluate perplexity...')
    loss_1 = []
    ntokens = []
    n_samples = 0
    for ppl_batch in tqdm(ppl_test_loader):
        input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(ppl_batch)

        with torch.no_grad():
            outputs_1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=lables,
                return_dict=True,
                )

        if args.ppl_type == 'tokens':
            trg_len = decoder_attention_mask.sum()
            log_likelihood_1 = outputs_1.loss * trg_len
            ntokens.append(trg_len)
            loss_1.append(log_likelihood_1)

        elif args.ppl_type == 'sents':
            n_samples += 1
            loss_1.append(torch.exp(outputs_1.loss))
        else:
            print(f"Invalid ppl type {args.ppl_type}")
            raise (ValueError)

    if args.ppl_type == 'tokens':
        ppl_1 = torch.exp(torch.stack(loss_1).sum() / torch.stack(ntokens).sum())
    elif args.ppl_type == 'sents':
        ppl_1 = torch.stack(loss_1).sum() / n_samples
    else:
        raise (ValueError)

    print(f"Perplexity on test set is {round(float(ppl_1.cpu().numpy()),3)} ."
          ) if CUDA_AVAILABLE else (
        f"Perplexity on test set is {round(float(ppl_1.numpy()),3)} .")

    if args.word_stat:
        print('Generating...')
        generated_token = []
        generated2_token = []
        gold_token = []
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        with open('evaluations/hyp.txt', 'w') as hyp, open('evaluations/hyp2.txt', 'w') as hyp2, open(
                'evaluations/ref.txt', 'w') as ref:
            for test_batch in tqdm(test_loader):
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                    test_batch)
                generated = model.generate(input_ids,
                                           attention_mask=attention_mask,
                                           num_beams=args.beam_size,
                                           length_penalty=args.length_penalty,
                                           min_length=args.min_length,
                                           no_repeat_ngram_size=args.no_repeat_ngram_size)
                                        #    per_input_ids=persona_input_ids)

                
                generated_token += tokenizer.batch_decode(generated,
                                                          skip_special_tokens=True)
               
                gold_token += tokenizer.batch_decode(decoder_input_ids,
                                                     skip_special_tokens=True)
            for g, r in zip(gold_token, generated_token):
                ref.write(f"{g}\n")
                hyp.write(f"{r}\n")

        hyp_d1, hyp_d2 = eval_distinct(generated_token)
        ref_d1, ref_d2 = eval_distinct(gold_token)
        print(f"Distinct-1 (hypothesis, reference): {round(hyp_d1,4)}, {round(ref_d1,4)}")
        print(f"Distinct-2 (hypothesis, reference): {round(hyp_d2,4)}, {round(ref_d2,4)}")



if __name__ == "__main__":
    parser = ArgumentParser("Transformers EncoderDecoderModel")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--word_stat", action="store_true")
    parser.add_argument("--use_decoder2", action="store_true") # 区分两次decode；

    parser.add_argument("--train_valid_split", type=float, default=0.1)

    parser.add_argument(
        "--encoder_model",
        type=str,
        default="./pretrained_models/bert/bert-base-uncased/")
    parser.add_argument(
        "--decoder_model",
        type=str,
        default="./pretrained_models/bert/bert-base-uncased/")
    
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/bertoverbert_epoch_5")

    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)

    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--eval_epoch", type=int, default=7)
    parser.add_argument("--print_frequency", type=int, default=-1)
    parser.add_argument("--warm_up_steps", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    parser.add_argument("--save_model_path",
                        type=str,
                        default="checkpoints/bertoverbert")
    parser.add_argument("--save_result_path",
                        type=str,
                        default="test_result.tsv")
    parser.add_argument("--dataset_type",
                        type=str,
                        default='convai2')  # convai2, ecdt2019
    parser.add_argument("--ppl_type",
                        type=str,
                        default='sents')  # sents, tokens

    parser.add_argument("--local_rank", type=int, default=0)
    
    '''
    dumped_token
        convai2:    ./data/ConvAI2/convai2_tokenized/
        ecdt2019:   ./data/ECDT2019/ecdt2019_tokenized/
    '''
    parser.add_argument("--dumped_token",
                        type=str,
                        default=None,
                        required=True)
    args = parser.parse_args()

    if args.do_train:
        train(args)
    if args.do_predict:
        predict(args)
    if args.do_evaluation:
        evaluation(args)

