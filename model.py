"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    ret = {tok: i for i, tok in enumerate(specials)}
    for s in sentences:
        s1 = s.split()
        for w in s1:
            if w in ret.keys():
                continue
            
            ret[w] = len(ret)
    
    return ret

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    ret = {v: k for k, v in token_to_id.items()}
    return ret

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    ret = []

    unk_token_id = token_to_id[unk_token]
    for w in sentence.split():
        t = token_to_id.get(w, unk_token_id)

        ret.append(t)


    return ret

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    ret = [id_to_token[i] for i in ids]
    return ret

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    if len(ids) >= max_len:
        return ids[:max_len]
    
    pads = [pad_id for _ in range(max_len - len(ids))]
    return ids + pads

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""

    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""

    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    even_len = d_model // 2
    
    ret = [10 ** (-4 * i / even_len) for i in range(even_len)]
    
    ret = torch.tensor(ret, dtype = torch.float)
    
    return ret

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""

    return torch.arange(max_len, dtype=torch.float).reshape((-1, 1))

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""

    l, d = pe.shape

    for l_idx in range(l):
        for i in range(0, d, 2):
            pe[l_idx, i] = torch.sin(position[l_idx] * div_term[i // 2])
    
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    l, d = pe.shape
    for i in range(l):
        for j in range(1, d, 2):
            pe[i][j] = torch.cos(position[i] * div_term[(j - 1) // 2])
    
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""

    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)

    pe = torch.zeros((max_len, d_model))

    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    _, L, _ = embedded_batch.shape
    pos_encoding = positional_encoding[:L, :]

    return embedded_batch + pos_encoding

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""

    B, L = token_ids.shape
    ret = token_ids != pad_id
    return ret.reshape((B, 1, 1, L))

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""

    seq = torch.arange(seq_len)
    ret = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        ret[i] = seq <= i
    
    return ret.reshape(1, 1, seq_len, seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""

    k = torch.transpose(key, -1, -2)
    return torch.matmul(query, k)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""

    return scores.masked_fill(~mask, -torch.inf)

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):

    exp_scores = torch.exp(masked_scores)
    exp_sums = torch.sum(exp_scores, -1, keepdim=True)
    exp_sums[exp_sums == 0] = 1
    return exp_scores / exp_sums

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""

    d_k = query.shape[-1]

    scores = compute_raw_attention_scores(query, key)

    scores = scale_attention_scores(scores, d_k)
    masked_scores = mask_attention_scores_with_neg_inf(scores, mask) if mask is not None else scores
    attention_weights = softmax_attention_weights(masked_scores)
    
    ctx = apply_attention_weights_to_values(attention_weights, value)

    return ctx, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):

    B, L, d_model = tensor.shape

    return tensor.reshape(B, L, num_heads, d_model // num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return torch.transpose(split_tensor, 1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    B, _, L, _ = multi_head_tensor.shape
    return torch.transpose(multi_head_tensor, 2, 1).reshape(B, L, -1)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):

    ret = torch.matmul(x, weight.T)
    if bias is not None:
        ret = ret + bias
    return ret

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)

    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    q = split_last_dim_into_heads(q, num_heads)
    q = transpose_heads_before_sequence(q)

    k = split_last_dim_into_heads(k, num_heads)
    k = transpose_heads_before_sequence(k)

    v = split_last_dim_into_heads(v, num_heads)
    v = transpose_heads_before_sequence(v)

    return q, k, v

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    context = merge_heads_back_to_model_dim(context)
    return apply_linear_projection(context, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    q, k, v = apply_linear_projection(query, w_q, None), apply_linear_projection(key, w_k, None), apply_linear_projection(value, w_v, None)
    q, k, v = split_qkv_into_heads(q, k, v, num_heads)
    ctx, _ = multi_head_scaled_dot_product_attention(q, k, v, mask)
    o = merge_heads_and_project_output(ctx, w_o, None)
    return o

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    out = x @ w1 + b1
    out[out < 0] = 0
    return out

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    x = apply_ffn_first_linear_and_relu(x, w1, b1)
    x = apply_ffn_second_linear(x, w2, b2)
    return x

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    
    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    mean, var = compute_layer_norm_mean_and_variance(x)

    x_hat = (x - mean) / torch.sqrt(var + eps)
    y = gamma * x_hat + beta

    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    return normalize_and_scale_with_gamma_beta(residual_input + sublayer_output, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    return (x * keep_mask) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    out = assemble_multi_head_attention_forward(x, x, x, w_q, w_k, w_v, w_o, num_heads, src_mask)
    return apply_residual_add_and_norm(x, out, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    y = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, y, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    y = encoder_layer_self_attention_sublayer(
        x,
        layer_params["w_q"], layer_params["w_k"], layer_params["w_v"], layer_params["w_o"],
        layer_params["attn_gamma"], layer_params["attn_beta"], num_heads, src_mask
    )

    y = encoder_layer_feed_forward_sublayer(
        y,
        layer_params["w1"], layer_params["b1"], layer_params["w2"], layer_params["b2"],
        layer_params["ffn_gamma"], layer_params["ffn_beta"]
    )

    return y

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    y_hat = assemble_multi_head_attention_forward(y, y, y, w_q, w_k, w_v, w_o, num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, y_hat, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

