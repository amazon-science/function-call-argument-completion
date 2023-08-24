## Overview
In this folder, we publish the codes for building `CALLARGS` from `PYENVS`.

## Concepts
By default, we use `/home/ubuntu/mydata/` as the work directory:

1. You should keep `PYENVS` under the path `/home/ubuntu/mydata/virtual/` so that the path of each virtual environment is like `/home/ubuntu/mydata/virtual/env_xxx` 
2. You should copy `src_env.pkl` and `license.pkl` to there.

`src_env.pkl` contains a list of `tuple(project_name, project_path)`.

`project_name` is what is listed in `top-pypi-packages-30-days.min.json`, it is used as the folder name of the virtual environment `env_{}` and other file name

The name of `package` denote the last folder in the project path (what Python uses to import)

For example: 

`project_name`: `sorl-thumbnail`

`project_path`: `/home/ubuntu/mydata/virtual/env_sorl-thumbnail/lib/python3.9/site-packages/sorl` (The package name is `sorl`)

## Request language server
### File structure

```
lsp_init.py # a template for initializing the language server
server.py # a class for running language server in a separate process
parse_python.py # parse the python file and find the candidate function calls
requests.py # an intergration for parsing and then sending requests to language server
main.py # an entry point for running all projects at scale
constant.py: pre-step for filerting function calls (remove such function call);
```

### Exmaple for running `main.py`:
```
python main.py src_env.pkl
```

Running `main.py` is intended to get the function calls and their definitions for the projects in the pkl file.


## Process for CALLARGS
Once we get the results from the above steps, we can run `distributable.ipynb`

It would generate the first version of CALLARGS: (e.g `res1 = /home/ubuntu/mydata/pkl_data/distributable/level4`)

Then use `signature_temp.ipynb` to record the function location in `res1` into json files and then run `main_signature.py` to get the signatureHelp for each function. After that, load the generated json files to the results of `res1` using (`signature_temp.ipynb` the sixth Cell). It will generate the result as `res2 = /home/ubuntu/mydata/pkl_data/distributable/level4_inter`

(The reason to find signatureHelp is to filter some functions whose defintion is the result of another function call, wher we cannot find their true implementation)

Then use `get_implementation.ipynb`, it will get the implementation body according to the definition location and modify the files in `res2`.

Finally run `usages.ipynb`, it will get the usages for each instance, it will lead to the final path `/home/ubuntu/mydata/pkl_data/distributable/level4_inter_refer`, which is the final version of CALLARGS.




