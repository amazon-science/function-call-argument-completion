# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pickle
import subprocess
import os
import time
import sys

if __name__=="__main__":
    name = sys.argv[1]
    if len(sys.argv) > 2:
        l = int(sys.argv[2])
        r = int(sys.argv[3])
    else:
        l = None
        r = None
    plis = [None] * 20
    src_lis = pickle.load(open(f"{name}", "rb"))
    if l is None:
        l = 0
    if r is None:
        r = len(src_lis)
    t = time.time()
    output_path = "/home/ubuntu/mydata/extract"
    for i, (pj, path) in enumerate(src_lis[l:r]):
        if os.path.isfile(f"{output_path}/{pj}.json"):
            print(f"{i}: Exist {pj}")
            continue
        else:
            print(f"{i}: Create {pj}")
            flag = None
            for i, cp in enumerate(plis):
                if cp is None:
                    flag = i
                    break
            while flag is None:
                for i, cp in enumerate(plis):
                    if cp.poll() is not None:
                        flag = i
                        break
                time.sleep(10)
            p = subprocess.Popen(f"python /home/ubuntu/function-call-argument-completion/generate/request.py {path} {output_path}/{pj}.json  /home/ubuntu/mydata/virtual/env_{pj}/lib/python3.9/site-packages/ >log/{pj}.log 2>&1", stdin=subprocess.PIPE, shell=True)
            plis[flag] = p
            
    s = 20
    while s>0:
        s = len(plis)
        for i, cp in enumerate(plis):
            if cp is not None:
                if cp.poll() is not None:
                    s-=1
            else:
                s-=1
        print(f"remain {s} unfinished")
        time.sleep(60)
    print(time.time(), r-l) 
                    
                    
            