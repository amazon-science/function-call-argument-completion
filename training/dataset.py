# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import gc
import shutil
import json

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import pickle
from preprocess import process
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

class finetuneDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank==-1:
            local_rank=0
            world_size=1
        else:
            local_rank=args.local_rank
            world_size=torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size)+"_wordsize_%d"%(world_size)+"_rank_%d"%(local_rank))
        if os.path.exists(cached_file) and not args.overwrite_cache:
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []
            if args.new_process:
                datafile = os.path.join(args.data_dir, f"clean_{file_type}.pkl")
                if file_type == 'train':
                    logger.warning("Creating features from dataset file at %s", datafile)
                with open(datafile, "rb") as f:
                    data = pickle.load(f)
            else: 
                datafile = os.path.join(args.data_dir, f"{file_type}.txt")
                if file_type == 'train':
                    logger.warning("Creating features from dataset file at %s", datafile)
                with open(datafile) as f:
                    data = f.readlines()

            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if args.model_type == "codegen":
                    """
                    if x.startswith("<s>"):
                        x = x.strip("<s>").strip()
                    x = x.replace("</s>", tokenizer.bos_token)
                    x = x.replace("<EOL>", "\n")
                    """
                    if x.startswith("<s>") and x.endswith("</s>"):
                        pass
                    else:
                        x = "<s> " + x + " </s>"
                else:    
                    if x.startswith("<s>") and x.endswith("</s>"):
                        pass
                    else:
                        x = "<s> " + x + " </s>"
                if idx == 0:
                    print(x)

                try:
                    input_ids.extend(tokenizer.encode(x, add_special_tokens=False))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("Rank %d, load %d"%(local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids) // world_size
            logger.info(f"tokens: {length*world_size}")
            input_ids = input_ids[local_rank*length: (local_rank+1)*length]

            if args.model_type == "unixcoder":
                block_size -= 3
            for i in range(0, length-block_size, block_size):
                if args.model_type == "unixcoder":
                    self.inputs.append(tokenizer.convert_tokens_to_ids(["<s>","<decoder-only>","</s>"])+input_ids[i : i + block_size])            
                else:
                    self.inputs.append(input_ids[i : i + block_size])            
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples"%(local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])

class EvalDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(block_size))
        if file_type!="test" and os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []
            if args.new_process:
                datafile = os.path.join(args.data_dir, f"clean_{file_type}.pkl")
                with open(datafile, "rb") as f:
                    data = pickle.load(f)
            else:
                datafile = os.path.join(args.data_dir, f"{file_type}.txt")
                with open(datafile) as f:
                    data = f.readlines()

            length = len(data)
            logger.info("Data size: %d"%(length))
            input_ids = []
            for idx,x in enumerate(data):
                x = x.strip()
                if args.model_type == "codegen":
                    """
                    if x.startswith("<s>"):
                        x = x.strip("<s>").strip()
                    x = x.replace("</s>", tokenizer.bos_token)
                    x = x.replace("<EOL>", "\n")
                    """
                    if x.startswith("<s>") and x.endswith("</s>"):
                        pass
                    else:
                        x = "<s> " + x + " </s>"
                else:
                    if x.startswith("<s>") and x.endswith("</s>"):
                        pass
                    else:
                        x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x, add_special_tokens=False))
                except Exception:
                    pass
                if idx % (length//10) == 0:
                    percent = idx / (length//10) * 10
                    logger.warning("load %d"%(percent))
            del data
            gc.collect()

            logger.info(f"tokens: {len(input_ids)}")
            self.split(args, input_ids, tokenizer, logger, block_size=block_size)
            del input_ids
            gc.collect()
            if file_type!="test":
                with open(cached_file, 'wb') as handle:
                    pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def split(self, args, input_ids, tokenizer, logger, block_size=1024):
        sample = []
        i = 0
        while i < len(input_ids):
            if args.model_type == "unixcoder":
                block_size -= 3
            sample = input_ids[i: i+block_size]
            if len(sample) == block_size:
                for j in range(block_size):
                    if tokenizer.convert_ids_to_tokens(sample[block_size-1-j])[0] == '\u0120' or tokenizer.convert_ids_to_tokens(sample[block_size-1-j]).startswith("<NUM_LIT"):
                        break
                    if sample[block_size-1-j] in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id]:
                        if sample[block_size-1-j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size-1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size-1-j]
            # print(len(sample))
            if args.model_type == "unixcoder":
                block_size += 3
                sample = tokenizer.convert_tokens_to_ids(["<s>","<decoder-only>","</s>"]) + sample
            i += len(sample)
            pad_len = block_size-len(sample)
            sample += [tokenizer.pad_token_id]*pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"{len(self.inputs)} samples")


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])
        


class lineDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924):
        datafile = os.path.join(args.data_dir, f"{file_type}.json")
        with open(datafile) as f:
            datas = f.readlines()

        length = len(datas)
        logger.info("Data size: %d"%(length))
        self.inputs = []
        self.gts = []
        for data in datas:
            data = json.loads(data.strip())
            self.inputs.append(tokenizer.encode(data["input"])[-block_size:])
            self.gts.append(data["gt"])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item]), self.gts[item]

class CallDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='test', block_size=924):
        self.file_type = file_type

        cached_file = os.path.join(args.output_dir, file_type+"_blocksize_%d"%(args.block_size))
        if block_size is not None and args.model_type == "unixcoder":
            block_size -= 3
        if file_type!="test" and os.path.exists(cached_file) and not args.overwrite_cache:
            with open(cached_file, 'rb') as handle:
                self.inputs, self.gts = pickle.load(handle)
        else:
            if len(args.do_eval_fc)>0:
                datafile = os.path.join(args.do_eval_fc, f"{file_type}.pkl")
            else:
                datafile = os.path.join(args.data_dir, f"{file_type}.pkl")
            datas = pickle.load(open(datafile, "rb"))
            length = len(datas)
            logger.info("Data size: %d"%(length))
            self.inputs = []
            self.gts = []
            for i , ex in tqdm(enumerate(datas)):
                inputs = ex["context"][0]+"("
                if args.not_process:
                    pass
                else:
                    inputs = process(inputs, ignore=False)
                    assert len(inputs)>0
                    inputs = " ".join(inputs)

                input_ids = tokenizer.encode(inputs, add_special_tokens=False)
                if file_type in ["train", "dev"]:
                    targets = ex["target"]
                    if args.not_process:
                        pass
                    else:
                        targets = process(targets, ignore=False)
                        assert len(targets)>0
                        targets = " ".join(targets)
                    target_ids = tokenizer.encode(targets, add_special_tokens=False)[:127]
                    target_ids.append(tokenizer.eos_token_id)
                    block_size = args.block_size - len(target_ids)

                if args.use_implementation:
                    implement = [x["body"].strip() for x in ex["definition"]]
                    if len(implement)>0:
                        implement = implement[0]
                    else:
                        implement = [" ".join(x["label"].strip().split()) for x in ex["signature"]]
                        implement = implement[0]
                    if args.not_process:
                        imp_ids = tokenizer.encode(implement, add_special_tokens=False)
                    else:
                        imp = process(implement, ignore=False)
                        imp = " ".join(imp)
                        imp_ids = tokenizer.encode(imp, add_special_tokens=False)
                    intend = block_size // 4
                    if len(imp_ids) > intend:
                        imp_ids = imp_ids[-intend:]
                else: 
                    imp_ids = []

                if args.usages>0:
                    ref_ids = []
                    for ref in ex["usages"][:args.usages][::-1]:
                        if isinstance(ref, str):
                            x = ref
                        else:
                            x = ref[0] + "(" + ref[-1]
                        if args.not_process:
                            pass
                        else:
                            x = process(x, ignore=False)
                            x = " ".join(x)
                        x_ids = tokenizer.encode(x, add_special_tokens=False)    
                        intend = block_size // (len(ex["usages"][:args.usages]) + 1 + args.use_implementation)
                        if len(x_ids) > intend:
                            x_ids = x_ids[-intend:]

                        if len(ref_ids) > 0:
                            ref_ids = ref_ids + tokenizer.convert_tokens_to_ids(["<EOL>", "<EOL>"])
                        ref_ids += x_ids
                else:
                    ref_ids = []

                if len(ref_ids) > 0 and len(imp_ids)>0:
                    tot = len(ref_ids) + 4 + len(imp_ids) + len(input_ids)
                    if tot > block_size:
                        if len(ref_ids) > block_size // 4:
                            over = min(len(ref_ids) - block_size // 4, (tot - block_size) // 3 + 1)
                            ref_ids = ref_ids[over:]
                        if len(imp_ids) > block_size // 4:
                            over = min(len(imp_ids) - block_size // 4, (len(ref_ids) + 4 + len(imp_ids) + len(input_ids) - block_size) // 2 + 1)
                            imp_ids = imp_ids[:-over]
                            
                        remain = len(ref_ids) + 4 + len(imp_ids) + len(input_ids) - block_size
                        input_ids = input_ids[remain:]
                        assert len(imp_ids) > 0 
                        assert len(ref_ids) > 0
                    input_ids = imp_ids + tokenizer.convert_tokens_to_ids(["<EOL>", "<EOL>"]) + ref_ids + tokenizer.convert_tokens_to_ids(["<EOL>", "<EOL>"]) + input_ids

                elif len(ref_ids) > 0:
                    tot = len(ref_ids) + 2 + len(input_ids)
                    if tot > block_size:
                        if len(ref_ids) < block_size // 4:
                            over = (tot - block_size) 
                            input_ids = input_ids[over:]
                        else:
                            over = min(len(ref_ids) - block_size // 4, (tot - block_size) // 2 + 1)
                            ref_ids = ref_ids[over:]
                            remain = len(ref_ids) + 2 + len(input_ids) - block_size
                            input_ids = input_ids[remain:]
                        
                    input_ids = ref_ids + tokenizer.convert_tokens_to_ids(["<EOL>", "<EOL>"]) + input_ids

                elif len(imp_ids)>0:
                    tot = len(imp_ids) + 2 + len(input_ids)
                    if tot > block_size:
                        if len(imp_ids) < block_size // 4:
                            over = (tot - block_size) 
                            input_ids = input_ids[over:]
                        else:
                            over = min(len(imp_ids) - block_size // 4, (tot - block_size) // 2 + 1)
                            imp_ids = imp_ids[:-over]
                            remain = len(imp_ids) + 2 + len(input_ids) - block_size
                            input_ids = input_ids[remain:]

                    input_ids = imp_ids + tokenizer.convert_tokens_to_ids(["<EOL>", "<EOL>"]) + input_ids
                
                else:
                    input_ids = input_ids[-block_size:]

                assert len(input_ids) <= block_size, (len(input_ids), len(ref_ids), len(imp_ids))

                if i==0:
                    print(tokenizer.convert_ids_to_tokens(input_ids))

                if args.model_type == "unixcoder":
                    input_ids = tokenizer.convert_tokens_to_ids(["<s>","<decoder-only>","</s>"]) + input_ids

                if file_type == "test":
                    targets = ex["target"]
                    self.gts.append(targets)
                else:
                    mask = np.zeros(args.block_size)
                    mask[len(input_ids):len(input_ids)+len(target_ids)] = 1
                    input_ids += target_ids
                    pad_len = args.block_size - len(input_ids)
                    input_ids += [tokenizer.pad_token_id] * pad_len
                    assert len(input_ids) == args.block_size
                    labels = np.array(input_ids)
                    labels = labels * mask + (1-mask) * (-100)
                    labels = labels.astype(int)
                    if i==0:
                        print(labels)
                    self.gts.append(labels.tolist())
                self.inputs.append(input_ids)
            if file_type!="test":
                with open(cached_file, 'wb') as handle:
                    pickle.dump((self.inputs, self.gts), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        if self.file_type == "test":
            return torch.tensor(self.inputs[item]), self.gts[item]
        else:
            return torch.tensor(self.inputs[item]), torch.tensor(self.gts[item]).long()

