# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
capabilities = {
    'textDocument': {
        'codeAction': {'dynamicRegistration': True},
        'codeLens': {'dynamicRegistration': True},
        'colorProvider': {'dynamicRegistration': True},
        'completion': {
            'completionItem': {
                'commitCharactersSupport': True,
                'documentationFormat': ['markdown', 'plaintext'],
                'snippetSupport': True
            },
            'completionItemKind': {'valueSet': list(range(1,26))},
            'contextSupport': True,
            'dynamicRegistration': True
        },                        
        'definition': {'dynamicRegistration': True, "linkSupport": True},
        "typeDefinition": {"dynamicRegistration": True, "linkSupport": True},
        "implementation": {"dynamicRegistration": True, "linkSupport": True},                                                   
        'documentHighlight': {'dynamicRegistration': True},
        'documentLink': {'dynamicRegistration': True},
        'documentSymbol': {
            'dynamicRegistration': True,
            'symbolKind': {'valueSet': list(range(1,27))}
        },
        'formatting': {'dynamicRegistration': True},
        'hover': {'contentFormat': ['markdown', 'plaintext'], 'dynamicRegistration': True},
        'onTypeFormatting': {'dynamicRegistration': True},
        'publishDiagnostics': {'relatedInformation': True},
        'rangeFormatting': {'dynamicRegistration': True},
        'references': {'dynamicRegistration': True},
        'rename': {'dynamicRegistration': True},
        'signatureHelp': {
            'dynamicRegistration': True,
            'signatureInformation': {
            'documentationFormat': ['markdown', 'plaintext']}
        },
        'synchronization': {'didSave': True, 'dynamicRegistration': True, 'willSave': True,  'willSaveWaitUntil': True},
    },
    'workspace': {
        'applyEdit': True,
        'configuration': True,
        'didChangeConfiguration': {'dynamicRegistration': True},
        'didChangeWatchedFiles': {'dynamicRegistration': True},
        'executeCommand': {'dynamicRegistration': True},
        'symbol': {'dynamicRegistration': True,
        'symbolKind': {'valueSet': list(range(1,27))}},
        'workspaceEdit': {'documentChanges': True},
        'workspaceFolders': True
    }
}

initializeoptions = {
    "codeAction": {
      "nameExtractVariable": "jls_extract_var",
      "nameExtractFunction": "jls_extract_def"
    },
    "completion": {
      "disableSnippets": False,
      "resolveEagerly": False,
      "ignorePatterns": []
    },
    "diagnostics": {
      "enable": True,
      "didOpen": True,
      "didChange": True,
      "didSave": True
    },
    "hover": {
      "enable": True,
      "disable": {
        "class": { "all": False, "names": [], "fullNames": [] },
        "function": { "all": False, "names": [], "fullNames": [] },
        "instance": { "all": False, "names": [], "fullNames": [] },
        "keyword": { "all": False, "names": [], "fullNames": [] },
        "module": { "all": False, "names": [], "fullNames": [] },
        "param": { "all": False, "names": [], "fullNames": [] },
        "path": { "all": False, "names": [], "fullNames": [] },
        "property": { "all": False, "names": [], "fullNames": [] },
        "statement": { "all": False, "names": [], "fullNames": [] }
      }
    },
    "jediSettings": {
      "autoImportModules": [],
      "caseInsensitiveCompletion": True,
      "debug": False
    },
    "markupKindPreferred": "markdown",
    "workspace": {
      "extraPaths": [],
      "symbols": {
        "ignoreFolders": [".nox", ".tox", ".venv", "__pycache__", "venv"],
        "maxSymbols": 20
      }
    }
}


jedi_init = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "capabilities": {
            "textDocument": {
                "hover": {
                    "dynamicRegistration":True,
                    "contentFormat": [
                        "plaintext",
                        "markdown"
                    ]
                },
                "synchronization": {
                    "dynamicRegistration":True,
                    "willSave":False,
                    "didSave":False,
                    "willSaveWaitUntil":False
                },
                "completion": {
                    "dynamicRegistration":True,
                    "completionItem": {
                        "snippetSupport":False,
                        "commitCharactersSupport":True,
                        "documentationFormat": [
                            "plaintext",
                            "markdown"
                        ],
                        "deprecatedSupport":False,
                        "preselectSupport":False
                    },
                    "contextSupport":False
                },
                "signatureHelp": {
                    "dynamicRegistration":True,
                    "signatureInformation": {
                        "documentationFormat": [
                            "plaintext",
                            "markdown"
                        ]
                    }
                },
                "declaration": {
                    "dynamicRegistration":True,
                    "linkSupport":True
                },
                "definition": {
                    "dynamicRegistration":True,
                    "linkSupport":True
                },
                "typeDefinition": {
                    "dynamicRegistration":True,
                    "linkSupport":True
                },
                "implementation": {
                    "dynamicRegistration":True,
                    "linkSupport":True
                }
            },
            "workspace": {
                "didChangeConfiguration": {
                    "dynamicRegistration":True
                }
            }
        },
        "initializationOptions":initializeoptions,
        "processId":None,
        "trace": "verbose"
    }
}

pylsp_init = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params":{
        "capabilities":capabilities
    }
}