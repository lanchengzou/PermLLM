import torch
import torch.nn as nn
import torch.nn.init as init
from scipy.optimize import linear_sum_assignment
from threading import Thread
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import warnings
from tqdm import tqdm
from typing import Any, Callable, Optional
from functools import partial
import logging
logger = logging.getLogger('Prune')


def sinkhorn(logits, tau=1.0, iter=5):

    P = torch.softmax(logits / tau, dim=-1)
    num_blocks, block_size, _ = P.shape

    for _ in range(iter):

        row_sum = P.sum(dim=-1, keepdim=True)
        P = P / (row_sum) 


        col_sum = P.sum(dim=-2, keepdim=True)
        P = P / (col_sum)
    return P

class LearnablePerm(nn.Module):
    def __init__(self, args, W, device, dtype, temperature=[3.0, 0.1]):
        super().__init__()
        self.args = args
        self.fix_param = False
        self.iteration = 0
        self.temperature = temperature
        self.current_temperature = self.temperature[0]
        if W.shape[1] % self.args.block_size == 0:
            self.num_blocks = W.shape[1] // self.args.block_size
            self.padding = False
            self.padding_size = 0
        logger.info("initalizing Permutation Matrix")
        self.weight = Parameter(torch.empty(
                    self.num_blocks, self.args.block_size, self.args.block_size,
                    device=device, dtype=dtype))
            # torch.nn.init.xavier_uniform_(self.weight)
        import math
        torch.nn.init.uniform_(
                self.weight,
                a=-1.0 / math.sqrt(self.args.block_size),
                b=1.0 / math.sqrt(self.args.block_size)
            )
        self.scale = Parameter(torch.tensor(math.sqrt(self.args.block_size), device=device, dtype=dtype))

        self.curr_index = None
        self.best_index = None

    def update_index(self):
        self.best_index = self.curr_index

    def permutation_matrix_to_index(self, P):

        index = P.argmax(dim=1)
        return index
    
    def index_to_permutation_matrix(self, index):
        n = index.size(0)
        perm_matrix = torch.zeros(n, n, device=index.device, dtype=index.dtype)
        perm_matrix[torch.arange(n), index] = 1 
        return perm_matrix
    
    def get_current_index(self):
        start_temp, end_temp = self.temperature
        self.current_temperature = start_temp + (end_temp - start_temp) * (self.iteration / self.args.iters)
        weight = self.weight * self.scale
        weight.data.clamp_(-1, 1)
        perm = self.block_hungarian_hard_permutation(sinkhorn(weight, tau=self.current_temperature), self.padding, self.padding_size)
        self.curr_index = self.permutation_matrix_to_index(perm)


    def fix_permutation(self):
        self.fix_param = True
        with torch.no_grad():
            new_weight = self.index_to_permutation_matrix(self.best_index).to(self.weight.dtype)
            del self.weight
            del self.scale
            torch.cuda.empty_cache()
            self.weight = new_weight

    def forward(self):
        if self.training:
            if self.weight is None:
                return None
            else:
                if self.fix_param:
                    return self.weight
                else:

                    start_temp, end_temp = self.temperature
                    self.current_temperature = start_temp + (end_temp - start_temp) * (self.iteration / self.args.iters)
                    weight = self.weight * self.scale
                    weight.data.clamp_(-1, 1)
                    # return sinkhorn(self.weight, tau=self.current_temperature)
                    perm = self.block_hungarian_hard_permutation(sinkhorn(weight, tau=self.current_temperature), self.padding, self.padding_size)
                    self.curr_index = self.permutation_matrix_to_index(perm)

                    return perm
        else:
            if self.fix_param:
                return self.weight
            else:
                return  self.index_to_permutation_matrix(self.curr_index).to(self.weight.dtype)


    def block_perm(self, cost_matrix, device):

        row_ind, col_ind = linear_sum_assignment(cost_matrix.to(torch.float32).detach().cpu().numpy())

        hard_permutation = torch.zeros_like(cost_matrix).to(device)
        hard_permutation[row_ind, col_ind] = 1.0

        return hard_permutation


    def block_hungarian_hard_permutation(self, P_soft, padding=False, padding_size=0):
        num_blocks, d_in, _ = P_soft.shape

        result_list = [None] * num_blocks
        for block_idx in range(num_blocks):
            if padding:
                if block_idx == num_blocks - 1:
                    end = P_soft.shape[-1] - padding_size
                    result_list[block_idx] = self.block_perm(-P_soft[block_idx][:end, :end], P_soft.device)
                else:
                    result_list[block_idx] = self.block_perm(-P_soft[block_idx], P_soft.device)
            else:
                result_list[block_idx] = self.block_perm(-P_soft[block_idx], P_soft.device)

        if padding:
            global_size = num_blocks * d_in - padding_size
        else:
            global_size = num_blocks * d_in
        hard_permutation_blocks = torch.zeros((global_size, global_size), dtype=P_soft.dtype, device=P_soft.device)

        for block_idx in range(num_blocks):
            if padding:
                if block_idx == num_blocks - 1:
                    start_idx = block_idx * d_in
                    end_idx = global_size
                else:
                    start_idx = block_idx * d_in
                    end_idx = start_idx + d_in
            else:
                start_idx = block_idx * d_in
                end_idx = start_idx + d_in
            if padding:
                if block_idx == num_blocks - 1:
                    end = d_in - padding_size
                    hard_permutation_blocks[start_idx:end_idx, start_idx:end_idx] = (result_list[block_idx] - P_soft[block_idx][:end, :end]).detach() + P_soft[block_idx][:end, :end]
                else:
                    hard_permutation_blocks[start_idx:end_idx, start_idx:end_idx] = (result_list[block_idx] - P_soft[block_idx]).detach() + P_soft[block_idx]
            else:
                hard_permutation_blocks[start_idx:end_idx, start_idx:end_idx] = (result_list[block_idx] - P_soft[block_idx]).detach() + P_soft[block_idx]

        return hard_permutation_blocks 
       
