# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import re
from tokenize import tokenize, untokenize, COMMENT, STRING, NEWLINE, ENCODING, ENDMARKER, NL, INDENT, NUMBER
from io import BytesIO
import json
import pickle
import ast
import astor

lits = json.load(open("/home/ubuntu/code/CodeXGLUE/Code-Code/CodeCompletion-token/dataset/py150/literals.json"))

def process_string(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    # string qualifiers like 'r' for regex, 'f' for formatted string, 'b' for bytes, 'u' for unicode, etc (or combination of them)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    # token string without qualifiers
    token_string = re.sub(qualifier_regex, "", token)
    # string literal without quotes
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    # if start_quote in str_quote_options[:2]:
    #     return ""
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits['str']
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )

def new_process(code, ignore=True):
    try:
        token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
        out_tokens = []
        prev_eol = False
        for toknum, tokval, _, _, _ in token_gen:
            if toknum == STRING:
                add_token = process_string(tokval)
                out_tokens.append((toknum, add_token))
                prev_eol = False
            elif toknum in [NEWLINE, NL]:
                if not prev_eol:
                    out_tokens.append((toknum, tokval))
                    prev_eol = True
            elif toknum in [COMMENT]:
                continue
            else:
                out_tokens.append((toknum, tokval))
                prev_eol = False
        out_tokens = untokenize(out_tokens)
        out_tokens = astor.to_source(ast.parse(out_tokens))
        out_tokens = "\n".join(out_tokens.split("\n\n"))
        return out_tokens
    except Exception as e:
        if ignore==True:
            out_tokens = ""
        else:
            return code
    return out_tokens

def process(code, ignore=True):
    try:
        token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
        out_tokens = []
        prev_eol = False
        for toknum, tokval, _, _, _ in token_gen:
            tokval = " ".join(tokval.split())
            if toknum == STRING:
                add_token = process_string(tokval)
                out_tokens.append(add_token)
                prev_eol = False
            elif toknum == NUMBER:
                out_tokens.append(str(toknum))
            elif toknum in [NEWLINE, NL]:
                if not prev_eol:
                    out_tokens.append("<EOL>")
                    prev_eol = True
            elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                continue
            else:
                out_tokens.append(tokval)
                prev_eol = False
        if out_tokens[0] == "<EOL>":
            out_tokens = out_tokens[1:]
        if out_tokens[-1] == "<EOL>":
            out_tokens = out_tokens[:-1]
    except Exception as e:
        if ignore==True:
            out_tokens = []
        else:
            return out_tokens
    return out_tokens

def py_tokenize(args, file_type):
    lis = pickle.load(open(os.path.join(args.base_dir, f"{file_type}.pkl"), "rb"))
    wf = open(os.path.join(args.output_dir, f"{file_type}.txt"), 'w')
    for i, code in enumerate(lis):
        out_tokens = process(code)
        out_tokens = ["<s>"] + out_tokens + ["</s>"]
        out = " ".join(out_tokens)
        wf.write(out+"\n")
        if i % 10000 == 0:
            print(f"{file_type}: {i} are done")
            if i==0:
                print(out)
    wf.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="py150_files", type=str, 
                        help="The downloaded data path")
    parser.add_argument("--output_dir", default="token_completion", type=str, 
                        help="The output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    py_tokenize(args, file_type="train")
    py_tokenize(args, file_type="dev")
    py_tokenize(args, file_type="test")

if __name__ == "__main__":
    main()
