# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import pickle
import subprocess
import os
import time
import sys

if __name__=="__main__":
    plis = [None] * 20
    lis = pickle.load(open(f"/home/ubuntu/mydata/pkl_data/non_exist.pkl", "rb"))
    t = time.time()
    for i, pj in enumerate(lis):
        if os.path.isfile(f"/home/ubuntu/mydata/signatureHelp/{pj}_signature.json"):
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
                time.sleep(5)
            p = subprocess.Popen(f"python signature.py /home/ubuntu/mydata/signatureHelp/{pj}.json /home/ubuntu/mydata/signatureHelp/{pj}_signature.json >/home/ubuntu/mydata/signatureHelp/log/{pj}.log 2>&1", stdin=subprocess.PIPE, shell=True)
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
        if s==0: 
            break
        time.sleep(30)
    print(time.time()-t) 
                    
                    
            