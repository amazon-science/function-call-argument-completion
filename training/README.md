## Overview
In this folder, we provide the codes for the experiments on `CALLARGS`.

## File structure

### For training general code completion models:
```
# mainly adapted from https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-token/code/
run_lm.py  # entry point
dataset.py # standardize the input
beam.py # modified version of beam search, end criterion: generating a matched ")"
preprocess.py # process the files (from `pkl` to `txt`) to train a general code completion model
```
### For task-specific finetuning
```
# mainly adapted from https://github.com/salesforce/CodeT5/
run_gen.py  # entry point
utils.py # prepocessing
models.py # load models
configs.py # contains the arguments for run_gen.py
```
### Others
`evaluator/`  contains the implementation for BLEU and codeBLEU metrics (depreciated)

`evaluate.ipynb` contains detailed evaluation metrics

`copy.ipynb` contains the method for usage copying.

## Runing scripts
examples are shown in `example_run_lm.sh` and `example_run_gen.sh`

## Requirements
```
torch
transformers
tensorboard
tree_sitter==0.2.2
astor
fuzzywuzzy
```
