# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from re import L, X
from sys import implementation
from torch.utils.data import TensorDataset
import numpy as np
import logging
import os
import errno
import random
import torch
import time
from tqdm import tqdm
import pickle
import ast
import astor

logger = logging.getLogger(__name__)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 pj, name, ty, exact, **kwargs):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.pj = pj
        self.name = name
        self.type = ty
        self.exact = exact
        for k,v in kwargs.items():
            setattr(self, k, v)

class Example(object):
    """A single training/test example."""

    def __init__(self, idx, context, target, folder, name, type, **kwargs):
        self.idx = idx
        self.source = context
        self.target = target.strip()
        self.pj = folder
        self.name = name
        self.type = type
        for k,v in kwargs.items():
            setattr(self, k, v)

def check(x):
    for op in ["=", "+", "-", ".", "*", "/", "[", "]", ")", "("]:
        x = op.join([y.strip() for y in x.split(op)])
    x = ", ".join([y.strip() for y in x.split(",")])
    x = x.replace("\n", " ")
    x = " ".join(x.split())        
    return x

def remove_comment(text):
    lis = text.split("\n")
    for i in range(len(lis)):
        if i==0:
            body = "\n".join(lis)
            remain = ""
        else:
            body = "\n".join(lis[:-i])
            remain = "\n".join(lis[-i:])
        try:
            node = ast.parse(body)
        except Exception as e:
            continue 
        node = node.body[0]
        if isinstance(node.body[0], ast.Expr):
            if hasattr(node.body[0], 'value') and isinstance(node.body[0].value, ast.Str):
                node.body = node.body[1:]
        body = astor.to_source(node)
        if len(remain.strip()) > 0:
            return body + "\n" + remain
        else:
            return body
    return text

def generate_target(example, args, tokenizer, special_token, stage):
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        target_str = target_str.replace(tokenizer.eos_token, tokenizer.unk_token)
        if args.decoder_only:
            target_tokens = tokenizer.tokenize(target_str)[:args.max_target_length - 1]
            target_tokens = target_tokens + [tokenizer.sep_token]  
        elif args.model_type in ['roberta']:
            target_tokens = tokenizer.tokenize(target_str)[:args.max_target_length - 2]
            target_tokens = [special_token] + target_tokens + [tokenizer.sep_token]  
        elif args.model_type in ['codet5']:
            target_tokens = tokenizer.tokenize(target_str)[:args.max_target_length - 3]
            target_tokens = [tokenizer.cls_token, special_token] + target_tokens + [tokenizer.sep_token]   
        elif args.model_type in ['plbart']:
            target_tokens = tokenizer.tokenize(target_str)[:args.max_target_length - 3]
            target_tokens = ["python", tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]   
        else:
            raise NotImplementedError         
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        if args.decoder_only:
            pass
        else:
            padding_length = args.max_target_length - len(target_ids)
            if args.model_type in ['roberta']:
                target_ids += [tokenizer.pad_token_id]*padding_length
            elif args.model_type in ['codet5', 'plbart']:
                target_ids += [-100]*padding_length

        assert target_ids.count(tokenizer.sep_token_id) == 1

    return target_ids

def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.decoder_only:
        special_token = ""
    elif args.model_type in ['codet5']:
        special_token = "<extra_id_0>"
    elif args.model_type in ['roberta']:
        special_token = "<mask0>"
    elif args.model_type in ['plbart']:
        special_token = tokenizer.mask_token
    else:
        raise NotImplementedError

    signature = []
    cur_signature_length = 0

    if args.max_signature_length > 0:
        if args.use_implementation:
            implement = [x["body"].strip() for x in example.definition if len(x["body"].strip().split("\n"))>1]
        else:
            implement = []

        if len(implement)>0:
            implement = implement[0]
            if args.remove_comment:
                implement = remove_comment(implement)
            signature = tokenizer.tokenize(implement.replace(tokenizer.eos_token, tokenizer.unk_token))
            signature = signature[:args.max_signature_length-1]
        else:
            signature = [" ".join(x["label"].strip().split())+" "+x["documentation"]["value"].strip() for x in example.signature]
            signature = signature[0]
            signature = signature.replace(tokenizer.eos_token, tokenizer.unk_token)
            signature = tokenizer.tokenize(signature)[:args.max_signature_length-1]
        signature = [tokenizer.sep_token] + signature

    if args.max_global > 0:
        x = [y.replace(tokenizer.eos_token, tokenizer.unk_token) for y in getattr(example, "global")]
        x = "\n".join(x)
        x = tokenizer.tokenize(x)[-(args.max_global-1):]
        signature = [tokenizer.sep_token] + x + signature

    exact = []    
    if args.usages:
        exact = [-100 for i in range(args.max_references)]
        assert args.max_references > 0
        single_length = min(128, args.max_references_length // max(1, len(example.usages[:args.max_references])))
        for i, ref in enumerate(example.usages[:args.max_references]):
            if isinstance(ref, str):
                x = ref.replace(tokenizer.eos_token, tokenizer.unk_token)
                x = tokenizer.tokenize(x)
            else:
                x = ref[0] + "(" + ref[-1]
                x = x.replace(tokenizer.eos_token, tokenizer.unk_token)
                if args.remove_comment:
                    x = remove_comment(x)
                x = tokenizer.tokenize(x)
                y = ref[1].replace(tokenizer.eos_token, tokenizer.unk_token)
                y = tokenizer.tokenize(y)
                right = int((single_length - 1) * 0.25)
                if len(y) > right:
                    y = y[:right]
                x.extend(y)

                if check(ref[-1].strip())==check(example.target):
                    exact[i] = 1
                else:
                    exact[i] = 0

            if len(x) > single_length - 1:
                x = x[-(single_length-1):]
                assert len(x) == single_length - 1 
            x = [tokenizer.sep_token] + x
            signature.extend(x)
            
    cur_signature_length = len(signature)

    if args.task == "single":
        assert args.decoder_only!=True
        available_full = args.max_source_length - cur_signature_length - 2
        left_length = int(args.max_source_left_length / args.max_source_length * available_full)
        source_str0 = example.source[0].strip() + "("
        if args.remove_comment:
            source_str0 = remove_comment(source_str0)
        source_str0 = source_str0.replace(tokenizer.eos_token, tokenizer.unk_token)
        tokenized0 = tokenizer.tokenize(source_str0)
        tokenized0 += [special_token]
        if len(tokenized0) > left_length:
            tokenized0 = tokenized0[-left_length:]
            assert len(tokenized0) == left_length

        source_str1 = example.source[1].strip().replace(tokenizer.eos_token, tokenizer.unk_token)
        tokenized1 = tokenizer.tokenize(source_str1)
        tokenized = tokenized0 + tokenized1
        if args.model_name_or_path == "microsoft/unixcoder-base":
            tokenized = ["<encoder-decoder>", tokenizer.sep_token] + tokenized
        tokenized = tokenized[:available_full]
        if len(signature) > 0:
            tokenized = tokenized + signature

        if args.model_type == "plbart":
            tokenized += [tokenizer.sep_token] + ["python"]
        else:
            tokenized = [tokenizer.cls_token] + tokenized + [tokenizer.sep_token]

        source_ids = tokenizer.convert_tokens_to_ids(tokenized) 
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        assert len(source_ids) == args.max_source_length

    elif args.task == "single_left":
        source_str = example.source[0].strip() + "("
        if args.remove_comment:
            source_str = remove_comment(source_str)
        source_str = source_str.replace(tokenizer.eos_token, tokenizer.unk_token)
        tokenized = tokenizer.tokenize(source_str)
        if args.decoder_only:
            avail = args.max_source_length - cur_signature_length
            if args.model_name_or_path == "microsoft/unixcoder-base":
                avail -= 3
            if len(tokenized) > avail:
                over = len(tokenized) - avail
                tokenized = tokenized[over:]
                assert len(tokenized) == avail

            if len(signature) > 0:
                tokenized = signature[1:] + tokenized

            if args.model_name_or_path == "microsoft/unixcoder-base":
                tokenized = ["<s>","<decoder-only>","</s>"] + tokenized

            source_ids = tokenizer.convert_tokens_to_ids(tokenized) 
            lis = [i for i,x in enumerate(source_ids) if x is None]
            if len(lis)>0:
                print(tokenized[lis[0]], source_ids[lis[0]], tokenizer.unk_token, tokenizer.unk_token_id)
                raise ValueError
        
            assert len(source_ids) <= args.max_source_length
        else:
            tokenized += [special_token]
            if len(tokenized) > args.max_source_length - cur_signature_length - 2:
                over = len(tokenized) - (args.max_source_length - cur_signature_length - 2)
                tokenized = tokenized[over:]
                assert len(tokenized) == args.max_source_length - cur_signature_length - 2

            if len(signature) > 0:
                tokenized = tokenized + signature

            if args.model_name_or_path == "microsoft/unixcoder-base":
                if len(tokenized) == args.max_source_length - 2:
                    tokenized[0] = "<encoder-decoder>"
                    tokenized[1] = tokenizer.sep_token
                elif len(tokenized) == args.max_source_length - 3:
                    tokenized = ["<encoder-decoder>"] + tokenized
                    tokenized[1] = tokenizer.sep_token
                else:
                    tokenized = ["<encoder-decoder>", tokenizer.sep_token] + tokenized 

            if args.model_type == "plbart":
                tokenized += [tokenizer.sep_token] + ["python"]
            else:
                tokenized =[tokenizer.cls_token] + tokenized + [tokenizer.sep_token]

            source_ids = tokenizer.convert_tokens_to_ids(tokenized) 
            padding_length = args.max_source_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            assert len(source_ids) == args.max_source_length

    elif args.task == "multiple":
        raise NotImplementedError
    else:
        raise NotImplementedError

    target_ids = generate_target(example, args, tokenizer, special_token, stage)

    if args.decoder_only:
        mask = np.zeros(args.max_source_length + args.max_target_length)
        mask[len(source_ids):len(source_ids)+len(target_ids)] = 1
        source_ids += target_ids
        pad_len = args.max_source_length + args.max_target_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * pad_len
        assert len(source_ids) == args.max_source_length + args.max_target_length
        labels = np.array(source_ids)
        labels = labels * mask + (1-mask) * (-100)
        target_ids = labels.astype(int).tolist()

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        example.pj,
        example.name,
        example.type,
        exact       
    )

def read_examples(filename, data_num):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    
    examples = []
    ids = 0
    for info in data:
        examples.append(Example(idx=ids, **info))
        ids += 1
        if ids == data_num:
            break
            
    return examples

def load_and_cache_gen_data(args, pool, tokenizer, split_tag):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + data_tag)
    filename = os.path.join(args.data_dir, split_tag+".pkl")

    examples = read_examples(filename, args.data_num)

    if args.debug:
        examples = random.sample(examples, min(5000, len(examples)))
    #if split_tag == 'train':
    #    calc_stats(examples, tokenizer, is_tokenize=True)
    #else:
    #    calc_stats(examples)
    if os.path.exists(cache_fn) and not args.debug and not args.no_cache:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if args.debug:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        elif not args.no_cache:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids)

        if args.local_rank in [-1, 0] and not args.debug and not args.no_cache:
            torch.save(data, cache_fn)
    
    logger.info("Example left source: {}".format(examples[10].source[0]))
    if args.remove_comment:
        logger.info("Example left source (no comment): {}".format(remove_comment(examples[10].source[0])))
    logger.info("Example right source: {}".format(examples[10].source[1]))
    logger.info("Example target: {}".format(examples[10].target))
    if args.use_implementation:
        logger.info("Use implementation")
    logger.info("Example input: {}".format(" ".join(tokenizer.convert_ids_to_tokens(data[10][0]))))
    if split_tag != "test":
        logger.info("Example output: {}".format(" ".join(tokenizer.convert_ids_to_tokens([x if x!=-100 else tokenizer.pad_token_id for x in data[10][1]]))))
    logger.info("max source len: {}".format(max([len(x[0]) for x in data])))
    logger.info("max target len: {}".format(max([len(x[-1]) for x in data])))
    return examples, data

def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(" ".join(ex.source).split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(" ".join(ex.source))))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(" ".join(ex.source).split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)
