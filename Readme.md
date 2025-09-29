## PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models


Lancheng Zou<sup>1</sup>, Shuo Yin<sup>4</sup>, Zehua Pei<sup>1</sup>, Tsung-Yi Ho<sup>1</sup>, Farzan Farnia<sup>1</sup>, Bei Yu<sup>1</sup>

<sup>1</sup> The Chinese University of Hong Kong 
 

Corresponding to {lczou23, byu}@cse.cuhk.edu.hk



#### Setup

Step 1: Create a new conda environment:

```
conda create -n permllm python=3.10
conda activate permllm
```



Step 2: Install relevant packages

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/nightly/cu118

```



Step 3: install lm-evaluation-harness (if `--eval_zero_shot`)

Follow the installation provided by Wanda repo: https://github.com/locuslab/wanda


#### Usage Example

$\text{PermLLM}_{wanda}$ under 2:4 sparsity.

```
python main.py \
	--model YOUR_MODEL_NAME \
	--prune_method permllm \
	--pruning wanda \
	--sparsity_ratio 0.5 \
	--sparsity_type 2:4 \
	--lr 0.001 \
	--iters 50 \
	--bsz 16 \
	--save \
```

Here the pruning can be replaced with ria, which is the zero-shot pruning metric for PermLLM.



#### Acknowledgment

---

This repository is built upon the [SparseGPT](https://github.com/IST-DASLab/sparsegpt), [Wanda](https://github.com/locuslab/wanda), and [RIA](https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning) repository.



#### Citation

----

If you use our code, please consider to cite:

```
@inproceedings{
zou2025permllm,
title={Perm{LLM}: Learnable Channel Permutation for N:M Sparse Large Language Models},
author={Lancheng Zou and Shuo Yin and Zehua Pei and Tsung-Yi Ho and Farzan Farnia and Bei Yu},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=V13dSX1wAs}
}
```
