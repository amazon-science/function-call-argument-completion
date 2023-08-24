# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from server import Server
from parse_python import parse_file
from copy import deepcopy
import json
import uuid
import os

def find_definition(server, filename):
    filename = os.path.abspath(filename)
    try:
        _, calls, node = parse_file(filename)
    except Exception as e:
        print(e, flush=True)
        return [], None
    for i, item in enumerate(calls):
        uid = str(uuid.uuid4())
        msg = {
            "jsonrpc": "2.0",
            "id" : uid,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": f"file://{filename}"},
                "position": item["position"]
            }   
        }
        server.send_msg(msg)
        x = server.receive()
        if isinstance(x, tuple):
            print("Something wrong with when finding the definition", item, flush=True)
            print(x)
            print()
            item["definition"] = None
        else:
            x = json.loads(x)
            assert x["id"] == uid
            if x.get("error") is not None:
                item["definition"] = None
            else:
                item["definition"] = x["result"]

        # We'd probably find the signatureHelp after finding all the definition locations
        # Save it in a file alone `function_help.json` for the function documentation.
        """
        cursor = deepcopy(item["position"])
        cursor["character"] += len(item["name"]) + 1
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
        if x is None:
            print("Something wrong with when finding the signatureHelp", item, flush=True)
            raise ValueError("Something wrong with when finding the signatureHelp")
        x = json.loads(x)
        assert x["id"] == uid
        if x.get("error") is not None:
            item["signatureHelp"] = None
        else:
            item["signatureHelp"] = x["result"]
        """

    return calls, node

def iterate_folders(server, folder, verbose=False):
    folder = os.path.abspath(folder)
    server.initialize(folder)
    res = {}
    def dfs(folder):
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and path.endswith(".py"):
                print("Working on:", path, flush=True)
                calls, _ = find_definition(server, path)
                res[path] = calls
                if verbose:
                    for item in calls:
                        if item["definition"] is None or len(item["definition"])==0:
                            print("Fail to find:", item["name"], item["position"])
                            pass    
                        else:
                            print(item["name"], item["position"], item["definition"])
                    print()

            elif os.path.isdir(path):
                dfs(path)

    dfs(folder)
    return res

if __name__ == "__main__":
    # python request.py /home/ubuntu/mydata/virtual/env_boto3/lib/python3.9/site-packages/boto3 /home/ubuntu/mydata/first/boto3.json  /home/ubuntu/mydata/virtual/env_boto3/lib/python3.9/site-packages/
    import time
    assert len(sys.argv) > 2
    path = sys.argv[1]
    output = sys.argv[2]
    extra = [] if len(sys.argv) <= 3 else [sys.argv[3]]
    server = Server("jedi", extra=extra)

    t = time.time()
    if os.path.isdir(path):
        res = iterate_folders(server, path, False)
    elif os.path.isfile(path):
        path = os.path.abspath(path)
        folder, _ = os.path.split(path)
        print("It is a file, use folder:", folder)
        server.initialize(folder)
        calls, _ = find_definition(server, path)
        res = {path: calls}
    else:
        raise ValueError("Not correct path!")
    print(time.time()-t)

    with open(output, "w") as f:
        out = json.dumps(res, indent=2, separators=(",", ":"))
        f.write(out)

    server.close()
    