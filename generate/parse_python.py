# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/python

import sys
import json as json
import ast
from constant import ignore_functions
def PrintUsage():
    sys.stderr.write("""
Usage:
    parse_python.py <file>

""")
    exit(1)

def read_file_to_string(filename):
    f = open(filename, 'rt')
    s = f.read()
    f.close()
    return s

def check_ignore(name, node):
    if name is None:
        return True
    if name in ignore_functions:
        return True
    if name.endswith("Error"):
        return True
    if node.args is None or len(node.args)==0:
        return True # TEMP we ignore function call without arguments
    return False

def parse_file(filename='<unknown>', text=None):
    global c, d
    if filename != '<unknown>':
        text = read_file_to_string(filename)
    tree = ast.parse(text, filename)
    lines = text.split("\n")
    #print("Lines:", len(lines))
    
    json_tree = []
    calls = []
    def gen_identifier(identifier, node_type = 'identifier'):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        json_node['value'] = identifier
        return pos
    
    def traverse_list(l, node_type = 'list', context=None):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = node_type
        children = []
        for item in l:
            children.append(traverse(item, context))
        if (len(children) != 0):
            json_node['children'] = children
        return pos

    def traverse(node, context):
        pos = len(json_tree)
        json_node = {}
        json_tree.append(json_node)
        json_node['type'] = type(node).__name__
        item = None
        if context is None and json_node["type"] == "FunctionDef":
            context = {
                "line":node.lineno-1,
                "character":node.col_offset,
                "end_line":node.end_lineno-1,
                "end_character":node.end_col_offset,
            }

        if json_node["type"] == "Call":
            name = None
            item = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
                subnode = node.func
                item = {
                    "name": name,
                    "position": {
                        "line": subnode.lineno - 1,
                        "character": subnode.col_offset
                    },
                    #"node": node
                }
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
                subnode = node.func   
                item = {
                    "name": name,
                    "position": {
                        "line": subnode.end_lineno - 1,
                        "character": subnode.end_col_offset - len(name)
                    },
                    #"node": node
                }
            elif isinstance(node.func, ast.Subscript):
                # need to infer the result
                # like CUSTOM_DESC_SUM_FN_DICT.get(obj_type_name)[0](obj, available_space, line_length)
                pass
            elif isinstance(node.func, ast.Call):
                # the function comes from the results of another function call
                # need to infer the result
                # like getattr(self._jc, name)()
                pass
            elif isinstance(node.func, (ast.BinOp, ast.BoolOp)):
                # the unknown function
                # (c_int * len(handles))(*handles)
                pass
            elif isinstance(node.func, ast.IfExp):
                # (input if PY3 else raw_input)('> ')
                pass
            elif isinstance(node.func, ast.Lambda):
                pass
            elif isinstance(node.func, ast.Constant):
                # "  %s = %s" (attr_name, attr_value)
                # ignore it 
                pass 
            else:
                print("Type unknown", node.lineno, node.col_offset, type(node.func))

            if not check_ignore(name, node):
                item["args_position"] = {
                    "line": item["position"]["line"],
                    "character": item["position"]["character"] + len(item["name"]),
                    "end_line": node.end_lineno - 1,
                    "end_character": node.end_col_offset
                }
                item["context"] = context
                calls.append(item)

            #return # TEMP only consider the outer function call for nested function call 
            
        children = []
        if isinstance(node, ast.Name):
            json_node['value'] = node.id
        elif isinstance(node, ast.Num):
            json_node['value'] = str(node.n)
        elif isinstance(node, ast.Str):
            json_node['value'] = node.s
        elif isinstance(node, ast.alias):
            json_node['value'] = str(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))
        elif isinstance(node, ast.FunctionDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.ClassDef):
            json_node['value'] = str(node.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node['value'] = str(node.module)
        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))
        elif isinstance(node, ast.keyword):
            json_node['value'] = str(node.arg)
        

        # Process children.
        if isinstance(node, ast.For):
            children.append(traverse(node.target, context))
            children.append(traverse(node.iter, context))
            children.append(traverse_list(node.body, 'body', context))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse', context))
        elif isinstance(node, ast.If) or isinstance(node, ast.While):
            children.append(traverse(node.test, context))
            children.append(traverse_list(node.body, 'body', context))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse', context))
        elif isinstance(node, ast.withitem):
            children.append(traverse(node.context_expr, context))
            if node.optional_vars:
                children.append(traverse(node.optional_vars, context))
        elif isinstance(node, ast.With):
            children.append(traverse_list(node.items, 'withitem', context))   
            children.append(traverse_list(node.body, 'body', context))
        elif isinstance(node, ast.Try):
            children.append(traverse_list(node.body, 'body', context))
            children.append(traverse_list(node.handlers, 'handlers',context))
            if node.orelse:
                children.append(traverse_list(node.orelse, 'orelse', context))
            if node.finalbody:
                children.append(traverse_list(node.finalbody, 'finalbody', context))
        elif isinstance(node, ast.arguments):
            children.append(traverse_list(node.args, 'args', context))
            children.append(traverse_list(node.defaults, 'defaults', context))
            if node.vararg:
                children.append(gen_identifier(node.vararg, 'vararg'))
            if node.kwarg:
                children.append(gen_identifier(node.kwarg, 'kwarg'))
        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], 'type', context))
            children.append(traverse_list(node.body, 'body', context))
        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, 'bases', context))
            children.append(traverse_list(node.body, 'body', context))
            children.append(traverse_list(node.decorator_list, 'decorator_list', context))
        elif isinstance(node, ast.FunctionDef):
            children.append(traverse(node.args, context))
            children.append(traverse_list(node.body, 'body', context))
            children.append(traverse_list(node.decorator_list, 'decorator_list', context))
        else:
            # Default handling: iterate over children.
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr_context) or isinstance(child, ast.operator) or isinstance(child, ast.boolop) or isinstance(child, ast.unaryop) or isinstance(child, ast.cmpop):
                    # Directly include expr_context, and operators into the type instead of creating a child.
                    json_node['type'] = json_node['type'] + type(child).__name__
                else:
                    children.append(traverse(child, context))
                
        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, 'attr'))
                
        if (len(children) != 0):
            json_node['children'] = children
        return pos
    
    traverse(tree, None)
    calls = sorted(calls, key = lambda x: (x["position"]["line"], x["position"]["character"]))
    return json_tree, calls, tree

if __name__ == "__main__":
    if len(sys.argv) != 2:
        PrintUsage()
    
    json_tree, calls, root = parse_file(sys.argv[1])
    for x in json_tree:
        if x["type"] == "body":
            print(x)
    #print(json.dumps(json_tree, separators=(',', ':'), ensure_ascii=False))
    #print()
    for item in calls:
        print(item)

