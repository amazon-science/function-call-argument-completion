# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from server import Server
from parse_python import parse_file
from copy import deepcopy
import json
import uuid
import os

def find_signature(server, calls):
    for i, item in enumerate(calls):
        cursor = deepcopy(item["position"])
        cursor["character"] += len(item["name"]) + 1
        filename = item["path"]
        uid = str(uuid.uuid4())
        msg = {
            "jsonrpc": "2.0",
            "id" : uid,
            "method": "textDocument/signatureHelp",
            "params": {
                "textDocument": {"uri": f"file://{filename}"},
                "position": cursor,
                "context": {"triggerKind":1, "isRetrigger":False}
            }  
        }
        server.send_msg(msg)
        x = server.receive()
        if isinstance(x, tuple):
            print("Something wrong with when finding the signatureHelp", item, flush=True)
            print(x)
            print()
            item["signature"] = []
        else:
            x = json.loads(x)
            assert x["id"] == uid
            try:
                item["signature"] = x["result"]["signatures"]
            except:
                print(item)
                item["signature"] = []
        
    return calls

if __name__ == "__main__":
    import time
    assert len(sys.argv) > 2
    path = sys.argv[1]
    output = sys.argv[2]
    with open(path, "r") as f:
        dic = json.load(f)
    extra = dic["extra"]
    folder = dic["folder"]
    lis = dic["lis"]

    server = Server("jedi", extra=extra)
    server.initialize(folder)

    res = find_signature(server, lis)

    with open(output, "w") as f:
        out = json.dumps(res, indent=2, separators=(",", ":"))
        f.write(out)

    server.close()
