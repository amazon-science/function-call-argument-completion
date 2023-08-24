# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time
from fuzzywuzzy import fuzz
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_elapse_time, load_and_cache_gen_data, make_sure_path_exists, check
from configs import add_args, set_seed, set_dist
import transformers

transformers.logging.set_verbosity_error()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids = batch[0]
        target_ids = batch[-1]
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(-100)

        with torch.no_grad():
            if args.decoder_only:
                outputs = model(source_ids, labels=target_ids)
                loss = outputs[0]
            elif args.model_type == 'roberta': # unicoder
                loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss

        if args.n_gpu > 1:
            loss = loss.mean()
            
        eval_loss += loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl

def transform(item):
    args, tokenizer, outputs = item
    if args.task.startswith("single"):
        res = tokenizer.decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if args.add_lang_ids and res.startswith("<python>"):
            res = res[8:]
        return res
    else:
        raise NotImplementedError

def eval_generate_epoch(args, pool, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running generate evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    eval_sampler = SequentialSampler(eval_data)
    batch_size = args.eval_batch_size
    if args.n_gpu > 1:
        batch_size = batch_size // args.n_gpu
    if args.decoder_only:
        batch_size = 1
    logger.info("  Batch size = %d", batch_size)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval generate for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta': # unicoder
                preds = model(source_ids=source_ids)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if hasattr(model, "module"):
                    gen_model = model.module
                else:
                    gen_model = model
                model_kwargs = {}
                if args.model_type == "plbart": 
                    start_id = tokenizer.lang_code_to_id["python"]
                else:
                    start_id = None
                if args.decoder_only:
                    length = source_mask.sum(1)[0]
                    preds = gen_model.generate(source_ids[:, :length],
                                    use_cache=True,
                                    num_beams=args.beam_size,
                                    early_stopping=False,
                                    eos_token_id=tokenizer.eos_token_id,
                                    max_new_tokens=args.max_target_length)
                    top_preds = list(preds.cpu().numpy()[:, length:])
                else:
                    preds = gen_model.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    decoder_start_token_id=start_id,
                                    num_beams=args.beam_size,
                                    early_stopping=False,
                                    max_length=args.max_target_length,**model_kwargs)
                    top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
        
    #pred_nls = [transform(args, tokenizer, outputs) for outputs in pred_ids]
    tuple_examples = [(args, tokenizer, outputs) for outputs in pred_ids]
    pred_nls = pool.map(transform, tqdm(tuple_examples, total=len(tuple_examples)))
    result = {}
    tys = list(set([x.type for x in eval_examples]))
    if args.do_eval_acc or split_tag == "test":
        dev_accs = []
        edit_sim = []
        detailed = {x:[] for x in tys}
        for pred_nl, gold in zip(pred_nls, eval_examples):
            a = check(pred_nl.strip()) 
            b = check(gold.target.strip())
            res = (a==b)
            dev_accs.append(res)
            edit_sim.append(fuzz.ratio(a, b))
            detailed[gold.type].append(res)
            
        result = {'acc': np.mean(dev_accs) * 100}
        logger.info(f"Detailed accuracy for {result['acc']}")
        result['edit'] = np.mean(edit_sim)
        for x in tys:
            acc = round(np.mean(detailed[x]) * 100, 2)
            logger.info(f"For type {x}, acc: {acc}")

    if args.do_eval_bleu or split_tag == "test":
        output_fn = os.path.join(args.output_dir, "{}_{}.output".format(split_tag, criteria))
        gold_fn = os.path.join(args.output_dir, "{}_{}.gold".format(split_tag, criteria))
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
            for pred_nl, gold in zip(pred_nls, eval_examples):                
                f.write(check(pred_nl.strip()) + '\n')
                f1.write(check(gold.target.strip()) + '\n')

        bleu = round(_bleu(gold_fn, output_fn), 2)
        codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)
        result['bleu'] = bleu
        result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logger.info(args)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    make_sure_path_exists(args.output_dir)
    make_sure_path_exists(args.cache_path)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'w')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/writer'.format(args.output_dir)
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, pool, tokenizer, 'train')
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)
        dev_dataset = {}
        if args.do_eval:
            eval_examples, eval_data = load_and_cache_gen_data(args, pool, tokenizer, 'dev')
            dev_dataset['dev_loss'] = eval_examples, eval_data
        # Prepare optimizer and schedule (linear warmup and decay)
        if args.fix_fuse:
            logger.info("Only training fuse layer")
            for name, p in model.named_parameters():
                if name.find("fuse")==-1 and not name.startswith("cls"):
                    p.requires_grad = False
                else:
                    logger.info(name)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size / args.gradient_accumulation_steps))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        global_step, best_edit, best_ppl, best_bleu = 0, -1, 1e6, 0
        not_loss_dec_cnt = 0
        not_bleu_em_inc_cnt = 0 if args.do_eval_bleu else 1e6
        not_edit_inc_cnt = 0 if args.do_eval_acc else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids = batch[0]
                target_ids = batch[-1]
                batch_size = source_ids.size(0)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(-100)

                if args.decoder_only:
                    outputs = model(source_ids, labels=target_ids)
                    loss = outputs[0]
                elif args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, target_ids=target_ids)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += batch_size
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                eval_examples, eval_data = dev_dataset['dev_loss']                    

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("Ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_edit_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, not_edit_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_edit_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        fa.write(early_stop_str)
                        break

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                
                if args.do_eval_acc:
                    result = eval_generate_epoch(args, pool, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu = result.get('bleu', 0)
                    dev_acc = result.get('acc', 0)
                    dev_edit = result.get('edit', 0)
                    dev_codebleu = result.get('codebleu', 0)
                    if dev_edit > best_edit:
                        not_edit_inc_cnt = 0
                        logger.info("  [%d] Best edit sim: %.2f, acc: %.2f, bleu: %.2f, codebleu %.2f",
                                    cur_epoch, dev_edit, dev_acc, dev_bleu, dev_codebleu)
                        logger.info("  " + "*" * 20)
                        best_edit = dev_edit
                        fa.write("[%d] Best edit sim changed into %.2f (acc: %.2f, bleu: %.2f, codebleu: %.2f)\n" % (
                            cur_epoch, dev_edit, dev_acc, dev_bleu, dev_codebleu))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best acc model into %s", output_model_file)
                    else:
                        not_edit_inc_cnt += 1
                        logger.info("Edit sim does not increase for %d epochs", not_edit_inc_cnt)
                        fa.write(
                            "[%d] Best edit sim (%.2f) does not drop changed for %d epochs, cur edit sim: %.2f, acc: %.2f (acc: %.2f, bleu: %.2f, em: %.2f)\n" % (
                                cur_epoch, best_edit, not_edit_inc_cnt, dev_edit, dev_acc, dev_bleu, dev_codebleu))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_edit_inc_cnt, not_loss_dec_cnt]]):
                            early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, not_edit_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_edit_inc_cnt, not_loss_dec_cnt)
                            logger.info(early_stop_str)
                            fa.write(early_stop_str)
                            break

        del model
        logger.info("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        for criteria in ["best-acc", 'best-ppl', 'last']:
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            if not os.path.exists(file):
                continue
            logger.info("Reload model from {}".format(file))
            config, model, tokenizer = build_or_load_gen_model(args)
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
            model.to(args.device)
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, pool, tokenizer, 'test')
            result = eval_generate_epoch(args, pool, eval_data, eval_examples, model, tokenizer, 'test', criteria)
             
            test_bleu = result.get('bleu', 0)
            test_acc = result.get('acc', 0)
            test_codebleu = result.get('codebleu', 0)
            result_str = "[%s] bleu-4: %.2f, test_acc: %.4f, codebleu: %.4f\n" % (criteria, test_bleu, test_acc, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
