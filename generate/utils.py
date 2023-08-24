# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from .parse_python import parse_file
import numpy as np
import os
import errno

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def mark_targets(lines, calls):
    idx = [np.zeros(len(x)+1, dtype=int)-1 for x in lines]
    targets = []
    for i, item in enumerate(calls):
        node = item["node"]
        x, y = item["position"]["line"], item["position"]["character"] + len(item["name"])
        #print(node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        if x==node.end_lineno-1:
            target = lines[x][y:node.end_col_offset]
            idx[x][y:node.end_col_offset] = i

        else:
            target = lines[x][y:]
            idx[x][y:] = i
            for j in range(x+1, node.end_lineno-1):
                target+="\n"+lines[j]
                idx[j][:] = i
            target+="\n"+lines[node.end_lineno-1][:node.end_col_offset]
            idx[node.end_lineno-1][:node.end_col_offset] = i

        #print(target[1:-1])
        targets.append(target)

    idx = np.concatenate(idx)
    return idx, targets

def mask_multiple_functions(text, calls):
    lines = text.split("\n")
    idx, targets = mark_targets(lines, calls)

    new_text = ""        
    i = 0
    j = 0
    while j<len(idx):
        j = i
        while j<len(idx) and idx[j]==-1: j+=1
        new_text+=text[i:j]
        if j==len(idx): break
        new_text+=f"(<extra_id_{idx[j]}>)"
        i = j
        while i<len(idx) and idx[i]==idx[j]: i+=1        
    return new_text, targets

def mask_single_function(text, calls, left_only=True):
    lines = text.split("\n")
    idx, targets = mark_targets(lines, calls)

    new_texts = []
    i = 0
    j = 0
    while j<len(idx):
        j = i
        while j<len(idx) and idx[j]==-1: j+=1
        if j==len(idx): break
        cur = text[:j]
        cur += "(<extra_id_0>)"
        i = j
        while i<len(idx) and idx[i]==idx[j]: i+=1
        if not left_only:
            cur += text[i:]
        new_texts.append(cur)        
    return new_texts, targets

if __name__ == "__main__":
    import json
    cur = []
    with open("/home/ubuntu/CodeSearchNet/python/final/jsonl/train/python_train_0.jsonl", "r") as f:
        for line in f.readlines():
            cur.append(json.loads(line.strip()))
    print(len(cur))

    text = cur[1000]["code"]
    trees, calls, node = parse_file(text=text)
    print(text)
    print()
    new_text, _ = mask_multiple_functions(text, calls)
    print(new_text)

    new_texts, _ = mask_single_function(text, calls)
    print(new_texts[0])
    print()

    new_texts, _ = mask_single_function(text, calls, False)
    print(new_texts[0])