import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
from importlib.metadata import version
from lib.eval import eval_ppl, eval_zero_shot
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import logging
from datetime import datetime


logger = logging.getLogger("Prune")

print('cuda', torch.version.cuda)
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights", seqlen=2048, args = None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto",
    )

    model.seqlen = seqlen
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--calib_dataset', type=str, default="c4", help='Calibration dataset')
    parser.add_argument('--eval_dataset', type=str, default="wikitext2", help='Evaluation dataset')
    parser.add_argument('--iters', type=int, default=50, help='Number of training iterations.')
    parser.add_argument('--block_size', type=int, default=64, help='Block size')
    parser.add_argument('--start_temp', type=float, default=1, help='Start temperature.')
    parser.add_argument('--end_temp', type=float, default=0.1, help='end temperature.')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--bsz', type=int, default=16, help='batch size for learnable perm.')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="2:4", help="Sparsity type, choose from unstructured, 4:8, 1:4, 2:4, 3:4. \
                        Please choose from the corresponding sparsity ratio")
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "ria", "permllm"])
    parser.add_argument("--pruning", type=str, choices=["ria", "wanda"], default="wanda", help="one-shot pruning method for PermLLM")
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--semi_sparse_acc', action="store_true", help="using pytorch semi sparse acceleration. Only when sparsity type is 2:4")
    parser.add_argument("--eval_zero_shot", action="store_true", help="zero-shot performance")
    parser.add_argument("--a", type=float, default=0.5, help="exponenet of activation")
    parser.add_argument("--reconstruction", action="store_true", help="remaining weight reconstruction based on sparsegpt")
    parser.add_argument("--reallocation", action="store_true", help="Heuristic Channel Reallocation")
    parser.add_argument("--lsa", action="store_true", help="Linear Sum Assignment")
    parser.add_argument("--importance_score", type=str, default="sum", help="assign importance score for columns")
    parser.add_argument("--gptq", action="store_true", help="use gptq or not")
    parser.add_argument("--per_outneuron", action="store_true", help="pruning per outneuron. Wanda's tactic.")
    parser.add_argument("--test_bs", type=int, default=1, help="test batch size")
    parser.add_argument("--use_cusparselt", action="store_true")
    parser.add_argument("--layer_wise", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_datetime = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    model_base_name = os.path.basename(args.model)
    model_name_without_ext = os.path.splitext(model_base_name)[0]
    log_filename = f"{model_name_without_ext}-{current_datetime}.log"
    log_path =  os.path.join(log_dir, log_filename)
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)
    if args.use_cusparselt:
        SparseSemiStructuredTensor._FORCE_CUTLASS = False
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    logger.info(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, args.seqlen, args)
    model.eval()
    if "opt" in args.model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    logger.info(f"use device {device}")
    logger.info(model)
    if args.sparsity_ratio != 0:
        logger.info("pruning starts")
        from lib.prune import prune_magnitude, prune_sparsegpt, prune_ria, check_sparsity, check_learnable_sparsity, prune_permllm
        if args.prune_method == "wanda":
            prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "ria":
            prune_ria(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "permllm":
            prune_permllm(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        ################################################################
        logger.info("*"*30)
        if args.prune_method == "permllm":
            sparsity_ratio = check_learnable_sparsity(args, model)
        else:
            sparsity_ratio = check_sparsity(args, model)
        logger.info(f"sparsity sanity check {sparsity_ratio:.4f}")
        logger.info("*"*30)
        ################################################################
    ppl_test = eval_ppl(model, tokenizer, args.eval_dataset, args.test_bs, device)
    logger.info(f"wikitext perplexity {ppl_test}")
    
    if args.save_model:
        model.save_pretrained(args.save_model+f"_{args.lr}_{args.block_size}_{args.iters}")
        tokenizer.save_pretrained(args.save_model+f"_{args.lr}_{args.block_size}_{args.iters}")
    

    if args.save:
        dirname = "results/{}".format(args.model)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        if args.layer_wise:
            filename = f"log_{args.prune_method}_layer.txt"
        else:
            filename = f"log_{args.prune_method}.txt"
        save_filepath = os.path.join(dirname, filename)
        with open(save_filepath, "a") as f:
            print("method\tactual_sparsity\tsparsity_pattern\treallocation\timportance_score\tlsa\tppl_test", file=f, flush=True)
            if args.sparsity_ratio == 0:
                sparsity_ratio = 0.0
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{args.sparsity_type}\t{args.reallocation}\t{args.importance_score}\t{args.lsa}\t{ppl_test:.4f}", file=f, flush=True)
    
    
    if args.eval_zero_shot:
        accelerate=True

        task_list = ["rte","hellaswag", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)
        logger.info(f"{results}")

    logger.info(f"{args.lr} {args.iters} {args.block_size}")

    
        

    

if __name__ == '__main__':
    main()
