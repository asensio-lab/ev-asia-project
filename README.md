# A global analysis of electric vehicle charging infrastructure: a cross-lingual deep learning approach across 72 languages in East and Southeast Asia 
This code replicates protocols in the manuscript: “A global analysis of electric vehicle charging infrastructure: a cross-lingual deep learning approach across 72 languages in East and Southeast Asia”. Proprietary data is restricted to authorized researchers only.

The Python scripts will run the neural network-based language models for multi-topic classification. The R scripts will produce the statistical models outputs, tables and figures shown in the paper.

## Requirements

### Python setup
Python code is written in Python 3.7. You can verify the version of Python by running the following command.

```
$ python —-version
# Python 3.7.0
```

Uses the following python packages. Can install with 'pip' as follows:

```
$ pip install torch==1.11.0
$ pip install os==3.1.0
$ pip install numpy==1.14.3
$ pip install pandas==0.23.0
$ pip install scikit-learn==0.19.1
$ pip install tqdm==1.9.0
$ pip install transformers==2.1.0
```



### R setup
R code written using version 4.1.2.

You will need to install the following packages, and set the working directory to this root folder:

```
> install.packages("frm")
> install.packages("tidyverse")
> install.packages("lubridate")
> install.packages("fastDummies")
> install.packages("stringr")

> setwd(<path_to_this_folder>)
```


### Data files
In order to reproduce the results and tables you wil need the following proprietary data files (authorized researchers only):

- train_final.csv
- valid_final.csv
- test_final.csv
- asia_test_final.csv

## Steps to Replicate: 

1. Ensure the data are in the working directory.

2. Run train.py to train, validate and map the labels. This may take 20-30 minutes to run on a cpu

3. Run regression_ev_asia.R to generate regression results. 






 

