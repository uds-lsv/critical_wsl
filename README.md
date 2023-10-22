# Weaker than you think
This repository contains the code for paper [Weaker Than You Think: A Critical Look at Weakly Supervised Learning](https://arxiv.org/abs/2305.17442) (ACL 2023)

## TL;DR

1. We demonstrate that the success of existing weakly supervised learning approaches heavily relies on the availability of clean validation samples. 
2. We show these can be leveraged much more efficiently by simply training on them. 


## Run the code

### 1. Prepare Environment
1. Install Pytorch

```
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

2. Install higher, but do not use pip. Instead, download the source code and install it from the source, See [here](https://github.com/facebookresearch/higher). Otherwise the AdamW optimizer may not work properly.

3. Install dependencies from `requirements.txt`



### 2. Prepare Data
1. We use the same data format as in [WRENCH](https://github.com/JieyuZ2/wrench).
2. An example data (subset of AGNews) is provided in `data_example`.


### 3. Run the code
Please refer to the example codes in `reproducibility` directory.


## Update
2023.10.22: Code is online! 🎉 the vanilla trainer (the FT trainer in the paper), COSINE trainer, and the L2R trainer are now integrated. **Please do not hesitate to contact me if you have any questions or require support on running the code** 🤗.

2023.07.10: Working on code clean up 🧹. Code will be online soon, please stay tuned! 🙌
