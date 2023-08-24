# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
ignore_functions = ["RuntimeError", "AttributeError", "ValueError", 'ImportError', 'NotImplementedError', 
                    'print', 'format', 'info', 'Exception', "warn", "warning", "debug", "exit",
                    "ArgumentParser", "add_argument", "sleep"] # ignore functions related to print information