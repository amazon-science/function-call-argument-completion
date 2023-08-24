# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import json
from lsp_init import jedi_init, pylsp_init
from copy import deepcopy
import os 
import uuid
import socket

JSON_RPC_REQ_FORMAT = "Content-Length: {json_string_len}\r\n\r\n{json_string}"
LEN_HEADER = "Content-Length: "
TYPE_HEADER = "Content-Type: "

class Server:
    def __init__(self, name, cmd_place=None, extra=[]):
        #pyls_cmd = ["pyright-langserver", "--stdio"]
        self.name = name
        if name == "pylsp":
            self.pyls_cmd = ["pylsp"] if cmd_place is None else [cmd_place]
            self.init_msg = pylsp_init
        elif name == "jedi":
            self.pyls_cmd = ["jedi-language-server"] if cmd_place is None else [cmd_place]
            print(self.pyls_cmd)
            self.init_msg = jedi_init
        self.extra = extra
        self.folder = None
        self.start()

    def start(self):
        self.p = subprocess.Popen(self.pyls_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("Start process", self.p.pid)
        self.hasinit = None

    def initialize(self, folder=None, verbose=False):
        assert self.p is not None
        if folder is None: # re-start and re-initialize
            assert self.folder is not None
            folder = self.folder
        else:
            self.folder = folder

        path = os.path.abspath(folder)
        uri = f"file://{path}/"
        workfolder = [{'name': uri, 'uri': uri}]
        if self.name == "pylsp":
            if self.hasinit is not None:
                print("Change folder to", folder)
                msg = {
                    "jsonrpc": "2.0",
                    "method": "workspace/didChangeWorkspaceFolders",
                    "params": {
                        "event" :{
                            "added":  workfolder,
                            "removed": self.hasinit
                        }
                    }
                }

            else:
                msg = deepcopy(self.init_msg)
                msg["params"]["workspaceFolders"] = workfolder

        elif self.name == "jedi":
            if self.hasinit is not None:
                print("Need to restart the process")
                self.close()
                self.start()
            msg = deepcopy(self.init_msg)
            msg["params"]["rootUri"] = uri
            msg["params"]["workspaceFolders"] = workfolder
            msg["params"]["initializationOptions"]["workspace"]["extraPaths"] = self.extra

        if verbose:
            print("Send:", msg)
        msg["id"] = str(uuid.uuid4())
        self.send_msg(msg)
        recv = self.receive()
        if verbose:
            print("Receive:", recv)
        self.hasinit = workfolder
        print()

    def close(self):
        self.p.kill()
        print("Stop", self.p.wait())
        self.p = None
        self.hasinit = None

    def send_msg(self, msg):
        str_msg = json.dumps(msg)
        full = JSON_RPC_REQ_FORMAT.format(json_string_len=len(str_msg), json_string=str_msg)
        self.p.stdin.write(full.encode())
        self.p.stdin.flush()

    def _error(self, line):
        a, _ = self.p.communicate()
        self.close()
        print("Restart langauge server")
        self.start()
        self.initialize()
        return ("Error", line)

    def receive(self):
        message_size = None
        while True:
            #read header
            line = self.p.stdout.readline()
            if not line:
                # server quit
                return self._error(None)
            line = line.decode("utf-8")
            line = line[:-2]
            if line == "":
                # done with the headers
                break
            elif line.startswith(LEN_HEADER):
                line = line[len(LEN_HEADER):]
                if not line.isdigit():
                    return self._error(line)
                message_size = int(line)
            elif line.startswith(TYPE_HEADER):
                # nothing todo with type for now.
                pass
            else:
                return self._error(line)

        jsonrpc_res = self.p.stdout.read(message_size).decode("utf-8")
        return jsonrpc_res
