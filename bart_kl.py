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
from torch.utils.data import DataLoader, Sampler, RandomSampler
import math
import sys

# from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

from transformers import BartTokenizer # unmodified
from modeling_bart import BartForConditionalGeneration # modified

from dataloader import ConvAI2Dataset, WrapperDataset


torch.manual_seed(42)

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    CUDA_AVAILABLE = True
else:
    print("CUDA NOT AVAILABLE")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def prepare_data_batch(batch):
    # print (batch['persona'].keys()) # ['input_ids', 'attention_mask'] 
    # BartTokenizer 不会返回 token_type_ids，需要自己make; RoBERTa会返回是要做sentence pair分类任务；
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
    model.train()
    
    print (model)
    print ('the number of trainable parameters: ', str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    # Tokenize & Batchify
    print(f"Load tokenized train & val dataset from {args.dumped_token}.")
    path = args.dumped_token
    shuffle_path = args.dumped_token_shuffle + '1/'

    def get_data_loader(path, shuffle_path):
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

        with open(shuffle_path + 'train_persona.json') as train_persona:
            tmp = train_persona.readline()
            shuffle_train_persona_tokenized = json.loads(tmp)

        with open(shuffle_path + 'train_query.json') as train_query:
            tmp = train_query.readline()
            shuffle_train_query_tokenized = json.loads(tmp)

        with open(shuffle_path + 'train_response.json') as train_response:
            tmp = train_response.readline()
            shuffle_train_response_tokenized = json.loads(tmp)

    
        train_dataset = ConvAI2Dataset(train_persona_tokenized,
                                    train_query_tokenized,
                                    train_response_tokenized,
                                    device) 

        val_dataset = ConvAI2Dataset(val_persona_tokenized,
                                    val_query_tokenized,
                                    val_response_tokenized,
                                    device) 

        shuffle_train_dataset = ConvAI2Dataset(shuffle_train_persona_tokenized,
                                    shuffle_train_query_tokenized,
                                    shuffle_train_response_tokenized,
                                    device) 


        wrapper_train_dataset = WrapperDataset(train_dataset, shuffle_train_dataset)
        print (len(train_dataset), len(shuffle_train_dataset), len(wrapper_train_dataset), args.batch_size)

        train_loader = DataLoader(wrapper_train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)

        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)
        
        return train_loader, val_loader, train_dataset

    train_loader, val_loader, train_dataset = get_data_loader(path, shuffle_path)

    # optim_warmup = AdamW(model.parameters(), lr=args.warm_up_learning_rate)
    # optim = AdamW(model.parameters(), lr=args.learning_rate)
    args.total_optim_steps = (len(train_dataset) // args.batch_size) * args.total_epochs
    print ('train_dataset, bz, total_optim_steps = ', len(train_dataset), args.batch_size, args.total_optim_steps)
    optim = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warm_up_steps, num_training_steps=args.total_optim_steps)
    optim.zero_grad()


    print("\nStart Training...")
    step = 0
    best_dev = 10000
    start_epoch = int(args.checkpoint.split("_")[-1]) if args.load_checkpoint else 0
    for epoch in range(start_epoch, args.total_epochs):
        print('\nTRAINING EPOCH %d' % epoch)
        batch_n = 0
        for batch in train_loader:
            batch_n += 1
            step += 1
            # optim_warmup.zero_grad()
            optim.zero_grad()

            batch1, batch2 = batch
            input_ids1, attention_mask1, decoder_input_ids1, decoder_attention_mask1, lables1, query_input_ids1, persona_input_ids1 = prepare_data_batch(batch1)
            input_ids2, attention_mask2, decoder_input_ids2, decoder_attention_mask2, lables2, query_input_ids2, persona_input_ids2 = prepare_data_batch(batch2)
            # context and response are the same
            assert torch.equal(query_input_ids1, query_input_ids2), (query_input_ids1.shape, query_input_ids2.shape)
            assert torch.equal(decoder_input_ids1, decoder_input_ids2), (decoder_input_ids1, decoder_input_ids2)
            assert torch.equal(lables1, lables2), (lables1[0], lables2[0])

            input_ids = torch.cat([input_ids1, input_ids2], dim=0)
            attention_mask = torch.cat([attention_mask1, attention_mask2], dim=0)
            decoder_input_ids = torch.cat([decoder_input_ids1, decoder_input_ids2], dim=0)
            decoder_attention_mask = torch.cat([decoder_attention_mask1, decoder_attention_mask2], dim=0)
            labels = torch.cat([lables1, lables2], dim=0)

            # regularize response:
            kl_mask = decoder_attention_mask.eq(0) # （0无效，1有效）

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask,
                            labels=labels,
                            return_dict=True,
                            kl=True,
                            kl_mask=kl_mask,
                            split_loss=args.split_loss,
                            left_one=args.left_one,
                            fine_grain_kl=args.fine_grain_kl)

            lm_loss = outputs.loss[0]
            kl_loss = outputs.loss[1]
            loss = lm_loss + (args.alpha * step / args.total_optim_steps) * kl_loss

            ppl = math.exp(lm_loss.item())

            lm_loss_prt = lm_loss.cpu().detach().numpy() if CUDA_AVAILABLE else lm_loss.detach().numpy()
            lm_loss_prt = round(float(lm_loss_prt),3)
            kl_loss_prt = kl_loss.cpu().detach().numpy() if CUDA_AVAILABLE else kl_loss.detach().numpy()
            kl_loss_prt = round(float(kl_loss_prt),3)
            loss_prt = loss.cpu().detach().numpy() if CUDA_AVAILABLE else loss.detach().numpy()
            loss_prt = round(float(loss_prt),3)
            ppl_prt = round(float(ppl),4)
            lr = optim.param_groups[0]['lr']

            if step <= args.warm_up_steps:
                if step % args.log_step == 0:
                    print(f"warm up step {step}  lr: {lr}  loss: {loss_prt}  kl_loss: {kl_loss_prt}  lm_loss: {lm_loss_prt}  ppl: {ppl_prt}")
            else:
                if step % args.log_step == 0:
                    print(f"train step {step}  lr: {lr}  loss: {loss_prt}  kl_loss: {kl_loss_prt}  lm_loss: {lm_loss_prt}  ppl: {ppl_prt}")
            
            loss.backward()
            optim.step()
            scheduler.step()

            if step % args.print_frequency == 0 and not args.print_frequency == -1:
                print('Sampling (not final results) ...')
                model.eval()
                for val_batch in val_loader:

                    input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                        val_batch)

                    generated = model.generate(input_ids,
                                               attention_mask=attention_mask)
                    
                    generated_token = tokenizer.batch_decode(
                        generated, skip_special_tokens=True)[-5:]
                   
                    query_token = tokenizer.batch_decode(
                        query_input_ids, skip_special_tokens=True)[-5:]

                    gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                        skip_special_tokens=True)[-5:]
                    persona_token = tokenizer.batch_decode(
                        persona_input_ids, skip_special_tokens=True)[-5:]

                    if rd.random() < 0.6:
                        for p, q, g, j in zip(persona_token, query_token, gold_token, generated_token):
                            print(
                                f"persona: {p[:150]}\nquery: {q[:100]}\ngold: {g[:100]}\npredict response: {j[:100]}\n")
                        break
                print('\nTRAINING EPOCH %d\n' % epoch)
                model.train()

            if step == 1 or step % args.valid_frequency == 0:
                # print('validing ppl ...')
                model.eval()
                loss_1 = []
                ntokens = []
                n_samples = 0
                for ppl_batch in val_loader:
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
                        # print ('loss, ppl = ', outputs_1.loss, torch.exp(outputs_1.loss))
                    else:
                        print(f"Invalid ppl type {args.ppl_type}")
                        raise (ValueError)

                if args.ppl_type == 'tokens':
                    ppl_1 = torch.exp(torch.stack(loss_1).sum() / torch.stack(ntokens).sum())
                elif args.ppl_type == 'sents':
                    ppl_1 = torch.stack(loss_1).sum() / n_samples
                else:
                    raise (ValueError)

                print(f"Perplexity on valid set is {round(float(ppl_1.cpu().numpy()),3)} ."
                    ) if CUDA_AVAILABLE else (
                    f"Perplexity on valid set is {round(float(ppl_1.numpy()),3)} .")

                if best_dev > float(ppl_1.cpu().numpy()):
                    best_dev = float(ppl_1.cpu().numpy())
                    if epoch > 1:
                        model.save_pretrained(f"{args.save_model_path}/ckp_best")
                model.train()

            sys.stdout.flush()

        if not step <= args.warm_up_steps:
            print(f'Saving model at epoch {epoch} step {step}')
            model.save_pretrained(f"{args.save_model_path}/ckp_%d" % epoch)

        # use a new shuffle data
        cur_shuffle_path = args.dumped_token_shuffle + str(epoch + 2) + '/'
        train_loader, _, _ = get_data_loader(path, cur_shuffle_path)


def predict(args):
    tokenizer = BartTokenizer.from_pretrained(args.encoder_model)

    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            # print(f"Load tokenized dataset from {args.dumped_token}.")
            # Loading testset
            with open(path + 'test_persona.json') as test_persona:
                tmp = test_persona.readline()
                test_persona_tokenized = json.loads(tmp)
            with open(path + 'test_query.json') as test_query:
                tmp = test_query.readline()
                test_query_tokenized = json.loads(tmp)
            with open(path + 'test_response.json') as test_response:
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
    test_loader = DataLoader(test_dataset, batch_size=args.infer_batch_size, shuffle=False)

    # Loading Model
    if args.dataset_type == 'convai2':
        model_path = f"./checkpoints/ConvAI2/bart/{args.exp_name}/ckp_{args.eval_epoch}"
    elif args.dataset_type == 'ecdt2019':
        model_path = f"./checkpoints/ECDT2019/bart_ecdt_{args.eval_epoch}"
    else:
        print(f"Invalid dataset_type {args.dataset_type}")
        raise (ValueError)
    # print("Loading Model from %s" % model_path)

    # from transformers import BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()


    # print(f"Writing generated results to {args.save_result_path}...")

    with open(args.save_result_path + '_human_view', "w", encoding="utf-8") as outf, \
        open(args.save_result_path + '_pred_response', "w", encoding="utf-8") as outfr:
        for test_batch in tqdm(test_loader):
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, lables, query_input_ids, persona_input_ids = prepare_data_batch(
                test_batch)

            generated = None
            if args.decode_strategy == 'beam_search':
                # 1. beam search
                generated = model.generate(input_ids,
                                        attention_mask=attention_mask,
                                        num_beams=args.beam_size,
                                        max_new_tokens=40,
                                        length_penalty=args.length_penalty,
                                        min_length=args.min_length)

            elif args.decode_strategy == 'sampling':
                # 2. sampling
                # generated = model.generate(input_ids, 
                #     attention_mask=attention_mask,
                #     top_k=10,
                #     top_p=0.9,
                #     do_sample=True,
                #     temperature=0.8,
                #     max_new_tokens=40
                # )
            
                generated = model.generate(input_ids, 
                    attention_mask=attention_mask,
                    top_k=50,
                    top_p=0.8,
                    do_sample=True,
                    temperature=1.2,
                    max_new_tokens=40
                )

            else:
                # 3. greedy
                generated = model.generate(input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=40,
                )

            generated_token = tokenizer.batch_decode(
                generated, skip_special_tokens=True)
            
            query_token = tokenizer.batch_decode(
                query_input_ids, skip_special_tokens=True)
                
            gold_token = tokenizer.batch_decode(decoder_input_ids,
                                                skip_special_tokens=True)
            persona_token = tokenizer.batch_decode(
                persona_input_ids, skip_special_tokens=True)

            
            for p, q, g, r in zip(persona_token, query_token, gold_token, generated_token):
                r = r.strip()
                while r[0] in ['.', '!', ',']:
                    r = r[1:]
                if r.startswith('esome'):
                    r = 'aw' + r
                if r.startswith('ounds'):
                    r = 's' + r
                if r.startswith('aker 2:'):
                    r = r[7:]
                outf.write(f"persona:{p}\tquery:{q}\tgold:{g}\tresponse_from_d1:{r}\n")
                outfr.write(r.strip() + '\n')

def evaluation(args):
    tokenizer = BartTokenizer.from_pretrained(args.encoder_model)

    if args.dumped_token is None:
        print('Pre-tokenized files must be provided.')
        raise (ValueError)
    else:
        path = args.dumped_token
        try:
            # print(f"Load tokenized dataset from {args.dumped_token}.")
            with open(path + 'test_persona.json') as test_persona:
                tmp = test_persona.readline()
                test_persona_tokenized = json.loads(tmp)
            with open(path + 'test_query.json') as test_query:
                tmp = test_query.readline()
                test_query_tokenized = json.loads(tmp)
            with open(path + 'test_response.json') as test_response:
                tmp = test_response.readline()
                test_response_tokenized = json.loads(tmp)

        except FileNotFoundError:
            print(f"Sorry! The file {args.dumped_token} can't be found.")

    test_dataset = ConvAI2Dataset(test_persona_tokenized,
                                  test_query_tokenized,
                                  test_response_tokenized,
                                  device)  

    ppl_test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Loading Model
    if args.dataset_type == 'convai2':
        # model_path = f"./checkpoints/ConvAI2/bart_{args.eval_epoch}"
        model_path = f"./checkpoints/ConvAI2/bart/{args.exp_name}/ckp_{args.eval_epoch}"
    elif args.dataset_type == 'ecdt2019':
        model_path = f"./checkpoints/ECDT2019/bart_ecdt_{args.eval_epoch}"
    else:
        print(f"Invalid dataset_type {args.dataset_type}")
        raise (ValueError)

    # print("Loading Model from %s" % model_path)
    # from transformers import BartForConditionalGeneration
    model = BartForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # print('Evaluate perplexity...')
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
            # print ('loss, ppl = ', outputs_1.loss, torch.exp(outputs_1.loss))
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


if __name__ == "__main__":
    parser = ArgumentParser("Transformers EncoderDecoderModel")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_evaluation", action="store_true")
    parser.add_argument("--word_stat", action="store_true")
    
    parser.add_argument("--split_loss", action="store_true")
    parser.add_argument("--kl_loss", action="store_true")
    parser.add_argument("--left_one", action="store_true")
    parser.add_argument("--fine_grain_kl", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.3)

    parser.add_argument(
        "--encoder_model",
        type=str,
        default="facebook/bart-base")
    
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/bertoverbert_epoch_5")

    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=32)

    parser.add_argument("--total_epochs", type=int, default=10)
    parser.add_argument("--eval_epoch", type=int, default=7)
    parser.add_argument("--print_frequency", type=int, default=800)
    parser.add_argument("--valid_frequency", type=int, default=500)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--warm_up_steps", type=int, default=1000)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--infer_batch_size", type=int, default=128)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--min_length", type=int, default=3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)

    parser.add_argument("--save_model_path",
                        type=str,
                        default="checkpoints/bart")
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

    parser.add_argument("--exp_name", type=str, default="bart_base_baseline")

    parser.add_argument("--decode_strategy", type=str, default="beam_search")

    
    '''
    dumped_token
        convai2:    ./data/ConvAI2/convai2_tokenized/
        ecdt2019:   ./data/ECDT2019/ecdt2019_tokenized/
    '''
    parser.add_argument("--dumped_token",
                        type=str,
                        default='./data/ConvAI2/convai2_tokenized_multi_turn_segtoken',
                        required=True)

    parser.add_argument("--dumped_token_shuffle",
                        type=str,
                        default='./data/ConvAI2/shuffle/convai2_tokenized_multi_turn_segtoken_shuffle_',
                        required=True)

    args = parser.parse_args()

    if args.do_train:
        train(args)

    if args.do_predict:
        predict(args) # to decode response

    if args.do_evaluation:
        evaluation(args) # to find best checkpoint

