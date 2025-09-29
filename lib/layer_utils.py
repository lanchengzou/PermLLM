import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .learnable_sparsity import LearnablePerm
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
import logging
import math

logger = logging.getLogger("Prune")



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def index_to_permutation_matrix(index):
    n = index.size(0)
    perm_matrix = torch.zeros(n, n, device=index.device, dtype=index.dtype)
    perm_matrix[torch.arange(n), index] = 1 
    return perm_matrix

class SparseLinear(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, scaler_row):
        super().__init__()
        self.args = args
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.scaler_row = scaler_row
        self.weight = module.weight
        self.warm_up_iter = int(self.args.iters*0.1)
        self.alpha = 1
        self.fix_param = False
        if getattr(module, 'bias', None) is not None:
            self.bias = module.bias
        else:
            self.register_parameter('bias', None)
        del module

    def mask(self, perm=None):
        assert perm is not None, "perm must not be None"
        if self.args.pruning == 'ria':
            W_metric = (torch.abs(self.weight)/torch.sum(torch.abs(self.weight), dim=0) + torch.abs(self.weight)/torch.sum(torch.abs(self.weight), dim=1).reshape(-1, 1)) * (torch.sqrt(self.scaler_row.reshape((1,-1))))**self.args.a
        else:
            W_metric = torch.abs(self.weight) * torch.sqrt(self.scaler_row.reshape((1,-1)))

        W_metric = W_metric.to(perm.dtype) @ perm
        num_rows, num_cols = W_metric.shape
        W_mask = torch.zeros_like(W_metric)
        num_blocks = num_cols // self.prune_m
        for ii in range(W_metric.shape[1]):
            if ii % self.prune_m == 0:
                tmp = W_metric[:,ii:(ii+self.prune_m)].float()
                W_mask.scatter_(1,ii+torch.topk(tmp, self.prune_n, dim=1, largest=True)[1], 1)
        W_mask = W_mask @ perm.T
        W_metric_reshaped = W_metric.view(num_rows, num_blocks, self.prune_m)
        # del W_metric
        
        soft_mask = torch.softmax(W_metric_reshaped, dim=-1)
        # del W_metric_reshaped
        # torch.cuda.empty_cache()
        soft_mask = soft_mask.view(num_rows, num_cols) @ perm.T
        
        
        return (W_mask - soft_mask).detach() + soft_mask

    def fix(self, perm):
        self.fix_param = True
        with torch.no_grad():
            if perm is None:
                self.weight.copy_(self.weight * self.mask())
            else:
                W_perm = self.weight * self.mask(perm)
                self.weight.copy_(W_perm)
            
    def forward(self, x, perm=None):
        if self.fix_param:
            return F.linear(x, self.weight, self.bias)
        else:
            if perm is None:
                    return F.linear(x, self.weight * self.mask(), self.bias)
            else:
                W_perm = self.weight * self.mask(perm)
                return F.linear(x, W_perm, self.bias)

class SparseLlamaAttention(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, weight_metrics):
        super().__init__()
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.attention_dropout = module.attention_dropout
        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.num_key_value_heads = module.num_key_value_heads
        self.num_key_value_groups = module.num_key_value_groups
        self.max_position_embeddings = module.max_position_embeddings
        self.rope_theta = module.rope_theta
        self.is_causal = module.is_causal
        self.q_proj = SparseLinear(module.q_proj, args, prune_n, prune_m, weight_metrics['self_attn.q_proj'])
        self.k_proj = SparseLinear(module.k_proj, args, prune_n, prune_m, weight_metrics['self_attn.k_proj'])
        self.v_proj = SparseLinear(module.v_proj, args, prune_n, prune_m, weight_metrics['self_attn.v_proj'])
        self.o_proj = SparseLinear(module.o_proj, args, prune_n, prune_m, weight_metrics['self_attn.o_proj'])

        self.perm_qkv = LearnablePerm(args, module.q_proj.weight.data, module.q_proj.weight.device, module.q_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.perm_o = LearnablePerm(args, module.o_proj.weight.data, module.o_proj.weight.device, module.o_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])

        self.rotary_emb = module.rotary_emb

        self.perm_forward = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.perm_forward:
            perm_qkv = self.perm_qkv()
            query_states = self.q_proj(hidden_states, perm_qkv)
            key_states = self.k_proj(hidden_states, perm_qkv)
            value_states = self.v_proj(hidden_states, perm_qkv)
        else:
            query_states = self.q_proj(hidden_states, None)
            key_states = self.k_proj(hidden_states, None)
            value_states = self.v_proj(hidden_states, None)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.perm_forward:
            perm_o = self.perm_o()
            attn_output = self.o_proj(attn_output, perm_o)
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class SparseLlamaMLP(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, weight_metrics):
        super().__init__()
        self.config = module.config
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = SparseLinear(module.gate_proj, args, prune_n, prune_m, weight_metrics['mlp.gate_proj'])
        self.up_proj = SparseLinear(module.up_proj, args, prune_n, prune_m, weight_metrics['mlp.up_proj'])
        self.down_proj = SparseLinear(module.down_proj, args, prune_n, prune_m, weight_metrics['mlp.down_proj'])
        self.act_fn = module.act_fn
        self.args = args

        self.perm_up = LearnablePerm(args, module.up_proj.weight.data, module.up_proj.weight.device, module.up_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.perm_down = LearnablePerm(args, module.down_proj.weight.data, module.down_proj.weight.device, module.down_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])

        self.perm_forward = True

    def forward(self, x):
        # if self.perm_forward:
        #     down_proj = self.down_proj((self.act_fn(self.gate_proj(x, self.perm_gate())) * self.up_proj(x, self.perm_up())), self.perm_down())
        # else:
        #     down_proj = self.down_proj((self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        if self.perm_forward:
            perm_up = self.perm_up()
            perm_down = self.perm_down()
            down_proj = self.down_proj((self.act_fn(self.gate_proj(x, perm_up)) * self.up_proj(x, perm_up)), perm_down)
        else:
            down_proj = self.down_proj((self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj

class SparseDecoderLayers(nn.Module):
    def __init__(self, layers, args):
        super().__init__()
        self.layers = layers
        self.args = args

    def forward(self, x, attention_mask=None, position_ids=None):
        for i in range(len(self.layers)):
            if "llama" in self.args.model or "Qwen" in self.args.model:
                x = self.layers[i](x, attention_mask=attention_mask, position_ids=position_ids)[0]
            elif "opt" in self.args.model:
                x = self.layers[i](x, attention_mask=attention_mask)[0]
        return x
    
def apply_rotary_pos_emb_qwen(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SparseQwenMLP(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, weight_metrics):
        super().__init__()
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = SparseLinear(module.gate_proj, args, prune_n, prune_m, weight_metrics['mlp.gate_proj'])
        self.up_proj = SparseLinear(module.up_proj, args, prune_n, prune_m, weight_metrics['mlp.up_proj'])
        self.down_proj = SparseLinear(module.down_proj, args, prune_n, prune_m, weight_metrics['mlp.down_proj'])
        self.act_fn = module.act_fn
        
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.gate_proj = SparseLinear(module.gate_proj, args, prune_n, prune_m, weight_metrics['mlp.gate_proj'])
        self.up_proj = SparseLinear(module.up_proj, args, prune_n, prune_m, weight_metrics['mlp.up_proj'])
        self.down_proj = SparseLinear(module.down_proj, args, prune_n, prune_m, weight_metrics['mlp.down_proj'])
        self.act_fn = module.act_fn
        
        self.args = args

        self.perm_up = LearnablePerm(args, module.up_proj.weight.data, module.up_proj.weight.device, module.up_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.perm_down = LearnablePerm(args, module.down_proj.weight.data, module.down_proj.weight.device, module.down_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])

        self.perm_forward = True

    def forward(self, x):
        # if self.perm_forward:
        #     down_proj = self.down_proj((self.act_fn(self.gate_proj(x, self.perm_gate())) * self.up_proj(x, self.perm_up())), self.perm_down())
        # else:
        #     down_proj = self.down_proj((self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        if self.perm_forward:
            perm_up = self.perm_up()
            perm_down = self.perm_down()
            down_proj = self.down_proj((self.act_fn(self.gate_proj(x, perm_up)) * self.up_proj(x, perm_up)), perm_down)
        else:
            down_proj = self.down_proj((self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
        return down_proj
    
class SparseQwenAttention(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, weight_metrics):
        super().__init__()        
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.hidden_size = module.hidden_size
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.num_key_value_heads = module.num_key_value_heads
        self.num_key_value_groups = module.num_key_value_groups
        self.max_position_embeddings = module.max_position_embeddings
        self.rope_theta = module.rope_theta
        self.is_causal = module.is_causal
        self.attention_dropout = module.attention_dropout
        
        self.q_proj = SparseLinear(module.q_proj, args, prune_n, prune_m, weight_metrics['self_attn.q_proj'])
        self.k_proj = SparseLinear(module.k_proj, args, prune_n, prune_m, weight_metrics['self_attn.k_proj'])
        self.v_proj = SparseLinear(module.v_proj, args, prune_n, prune_m, weight_metrics['self_attn.v_proj'])
        self.o_proj = SparseLinear(module.o_proj, args, prune_n, prune_m, weight_metrics['self_attn.o_proj'])

        self.perm_qkv = LearnablePerm(args, module.q_proj.weight.data, module.q_proj.weight.device, module.q_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.perm_o = LearnablePerm(args, module.o_proj.weight.data, module.o_proj.weight.device, module.o_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])

        self.rotary_emb = module.rotary_emb

        self.perm_forward = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.perm_forward:
            perm_qkv = self.perm_qkv()
            query_states = self.q_proj(hidden_states, perm_qkv)
            key_states = self.k_proj(hidden_states, perm_qkv)
            value_states = self.v_proj(hidden_states, perm_qkv)
        else:
            query_states = self.q_proj(hidden_states, None)
            key_states = self.k_proj(hidden_states, None)
            value_states = self.v_proj(hidden_states, None)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb_qwen(query_states, key_states, cos, sin, position_ids)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.perm_forward:
            perm_o = self.perm_o()
            attn_output = self.o_proj(attn_output, perm_o)
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    
class SparseOPTAttention(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, weight_metrics):
        super().__init__()
        self.config = module.config
        self.embed_dim = module.embed_dim
        self.num_heads = module.num_heads
        self.dropout = module.dropout
        self.enable_bias = module.enable_bias

        self.head_dim = module.head_dim
        self.is_causal = module.is_causal

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = module.scaling
        self.is_decoder = module.is_decoder
        self.args = args
        self.k_proj = SparseLinear(module.k_proj, args, prune_n, prune_m, weight_metrics['self_attn.k_proj'])
        self.v_proj = SparseLinear(module.v_proj, args, prune_n, prune_m, weight_metrics['self_attn.v_proj'])
        self.q_proj = SparseLinear(module.q_proj, args, prune_n, prune_m, weight_metrics['self_attn.q_proj'])
        self.out_proj = SparseLinear(module.out_proj, args, prune_n, prune_m, weight_metrics['self_attn.out_proj'])

        self.perm_qkv = LearnablePerm(args, module.q_proj.weight.data, module.q_proj.weight.device, module.q_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.perm_o = LearnablePerm(args, module.out_proj.weight.data, module.out_proj.weight.device, module.out_proj.weight.dtype, temperature=[args.start_temp, args.end_temp])

        self.perm_forward = True

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        if self.perm_forward:
            perm_qkv = self.perm_qkv()
        else:
            perm_qkv = None

        query_states = self.q_proj(hidden_states, perm_qkv) * self.scaling

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states, perm_qkv), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states, perm_qkv), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states, perm_qkv), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states, perm_qkv), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states, perm_qkv), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states, perm_qkv), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask.to(attn_weights.device)
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        if self.perm_forward:
            perm_o = self.perm_o()
        else:
            perm_o = None

        attn_output = self.out_proj(attn_output, perm_o)

        return attn_output, attn_weights_reshaped, past_key_value
        
class SparseOPTDecoder(nn.Module):
    def __init__(self, module, args, prune_n, prune_m, weight_metrics):
        super().__init__()
        self.embed_dim = module.embed_dim
        self.args = args
        self.self_attn = SparseOPTAttention(module.self_attn, args, prune_n, prune_m, weight_metrics)

        self.do_layer_norm_before = module.do_layer_norm_before
        self.dropout = module.dropout
        self.activation_fn = module.activation_fn

        self.self_attn_layer_norm = module.self_attn_layer_norm
        self.fc1 = SparseLinear(module.fc1, args, prune_n, prune_m, weight_metrics['fc1'])
        self.fc2 = SparseLinear(module.fc2, args, prune_n, prune_m, weight_metrics['fc2'])

        self.perm_fc1 = LearnablePerm(args, module.fc1.weight.data, module.fc1.weight.device, module.fc1.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.perm_fc2 = LearnablePerm(args, module.fc2.weight.data, module.fc2.weight.device, module.fc2.weight.dtype, temperature=[args.start_temp, args.end_temp])
        self.final_layer_norm = module.final_layer_norm

        self.perm_forward = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        # print(f"before attn {hidden_states.device}")
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # print(f"after attn {hidden_states.device}")
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.perm_forward:
            perm_fc1 = self.perm_fc1()
        else:
            perm_fc1 = None

        hidden_states = self.fc1(hidden_states, perm_fc1)
        hidden_states = self.activation_fn(hidden_states)

        if self.perm_forward:
            perm_fc2 = self.perm_fc2()
        else:
            perm_fc2 = None

        hidden_states = self.fc2(hidden_states, perm_fc2)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
