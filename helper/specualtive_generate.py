"""Optimized OPD EAGLE3 rollout; OFF dispatches to frozen historical FastGRPO."""
import torch
import math
import time
from copy import deepcopy
from transformers import DynamicCache
from helper.tree_verification import pack_tree, trace_verified_path, select_confidence_nodes
from helper.opd_history import ContiguousRolloutHistory as RolloutHistory
from helper.opd_reflex import OPDReflex
from helper.method_config import resolve_method
from helper.historical_fastgrpo import speculative_generate as historical_generate
from helper.opd_reflex import OPD_COUNTER_NAMES
from helper.opd_scheduling import schedule,compact_suffix_inplace
from helper.sampling import build_sampling_probs, sample_from_probs, sample_target_from_logits
total_target_time = 0
total_draft_time = 0
total_check_time = 0

def _cache_num_layers(cache):
    if cache is None:
        return 0
    if hasattr(cache, 'key_cache'):
        return len(cache.key_cache)
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    return len(cache)

def _cache_get_layer(cache, layer_idx):
    if hasattr(cache, 'key_cache'):
        return (cache.key_cache[layer_idx], cache.value_cache[layer_idx])
    if hasattr(cache, 'layers'):
        layer = cache.layers[layer_idx]
        for (key_name, value_name) in (('keys', 'values'), ('key_cache', 'value_cache'), ('key_states', 'value_states'), ('_keys', '_values')):
            if hasattr(layer, key_name) and hasattr(layer, value_name):
                return (getattr(layer, key_name), getattr(layer, value_name))
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            return (layer[0], layer[1])
    return cache[layer_idx]

def _cache_set_layer(cache, layer_idx, key, value):
    if hasattr(cache, 'key_cache'):
        cache.key_cache[layer_idx] = key
        cache.value_cache[layer_idx] = value
        return
    if hasattr(cache, 'layers'):
        layer = cache.layers[layer_idx]
        for (key_name, value_name) in (('keys', 'values'), ('key_cache', 'value_cache'), ('key_states', 'value_states'), ('_keys', '_values')):
            if hasattr(layer, key_name) and hasattr(layer, value_name):
                try:
                    setattr(layer, key_name, key)
                    setattr(layer, value_name, value)
                    return
                except (AttributeError, RuntimeError):
                    pass
        if isinstance(layer, list) and len(layer) >= 2:
            layer[0] = key
            layer[1] = value
            return
    try:
        cache[layer_idx] = (key, value)
        return
    except TypeError as exc:
        raise AttributeError('Unsupported transformers cache layout: cannot set layer key/value tensors') from exc

def _cache_seq_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, 'get_seq_length'):
        return cache.get_seq_length()
    if _cache_num_layers(cache) == 0:
        return 0
    (key, _) = _cache_get_layer(cache, 0)
    return int(key.shape[-2])

def sampling(logits, top_k=None, top_p=None, temperature=0.6, eos_token_id=2):
    """
    Perform combined top-k and top-p (nucleus) sampling on logits.
    
    Args:
        logits (torch.Tensor): Logits from the model output (shape: [batch_size, seq_len, vocab_size]).
        top_k (int or None): Number of highest probability tokens to consider for top-k sampling.
        top_p (float or None): Cumulative probability threshold for top-p sampling.
        temperature (float): Temperature to adjust the sharpness of the distribution.
        eos_token_id (int): The ID of the end-of-sequence token (used as fallback when logits are invalid).
    
    Returns:
        torch.Tensor: Sampled token indices (shape: [batch_size, seq_len]).
    """
    probs = build_sampling_probs(logits, temperature, top_p, top_k, eos_token_id)
    return sample_from_probs(probs)

def get_adaptive_hyperparameters(bsz, verification_capacity, max_draft_token_length, max_draft_k, max_verification_num, min_draft_token_length, draft_token_length_c):
    if bsz <= 0:
        return (1, 1, 1)
    verification_num = max(2, min(math.floor(verification_capacity / bsz), max_verification_num))
    draft_token_length = min(math.floor(math.log2(verification_num / draft_token_length_c)), max_draft_token_length)
    draft_token_length = max(draft_token_length, min_draft_token_length)
    draft_k = max(1, min(verification_num - 1, max_draft_k))
    tree_node_capacity = draft_k + draft_k * draft_k * max(draft_token_length - 1, 0)
    draft_total_token = min(verification_num - 1, tree_node_capacity)
    return (draft_token_length, draft_k, draft_total_token)

@torch.inference_mode()
def speculative_generate(model, input_ids, attention_mask, tokenizer, do_sample=False, repeated_generate_nums=None, temperature=0.8, top_p=0.9, top_k=None, verification_capacity=160, max_draft_token_length=5, max_draft_k=8, max_verification_num=160, min_draft_token_length=3, draft_token_length_c=0.75, statistical_time=False, return_all_draft_input=False, max_length=2048, method='fastgrpo', opd_rank=8, opd_topk=16, opd_fast_lr=0.01, opd_visited_weight=1.0, opd_frontier_weight=1.0, opd_update_stream=False, opd_profile=False, opd_diagnostics=False, opd_backend='auto', opd_train_projector=False, kv_gather_strategy='stacked'):
    (method, _) = resolve_method(method)
    if method=='fastgrpo':
        result=historical_generate(model,input_ids,attention_mask,tokenizer,
            do_sample=do_sample,repeated_generate_nums=repeated_generate_nums,temperature=temperature,
            top_p=top_p,top_k=top_k,verification_capacity=verification_capacity,
            max_draft_token_length=max_draft_token_length,max_draft_k=max_draft_k,max_verification_num=max_verification_num,
            min_draft_token_length=min_draft_token_length,draft_token_length_c=draft_token_length_c,
            statistical_time=statistical_time,return_all_draft_input=return_all_draft_input,max_length=max_length,
            kv_gather_strategy=kv_gather_strategy)
        result.update({name:0. for name in OPD_COUNTER_NAMES})
        result.update(opd_backend='off',opd_profile_time_ms=0.,opd_profile_sections_ms=None)
        return result
    if not getattr(model, 'is_eagle3_specforge', False):
        raise ValueError('Both fair modes require the real SpecForge EAGLE-3 backend')
    if kv_gather_strategy not in {'stacked', 'per_layer'}:
        raise ValueError('invalid KV gather strategy')
    enabled = method == 'opd_reflex'
    opd = None
    compact_to_target = None

    def draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values_tree, draft_token_length, past_position_ids_tensor, padding_positions, draft_k=4, draft_total_token=32):
        global total_check_time
        dtype = model.dtype
        device = model.device
        bsz = draft_hidden_states.shape[0]
        node_nums = draft_k + draft_k * draft_k * (draft_token_length - 1)
        trees = []
        full_parents = [torch.full((bsz, draft_k), -1, device=device, dtype=torch.long)]
        full_contexts = torch.full((bsz, node_nums), -1, device=device, dtype=torch.long)
        beam_node_ids = torch.arange(draft_k, device=device).expand(bsz, -1)
        parents_list = []
        total_input_ids = []
        total_position_ids = []
        confidences = []
        draft_position_ids = past_position_ids_tensor.unsqueeze(-1).repeat(1, draft_k)
        total_position_ids.append(draft_position_ids)
        if enabled and hasattr(model, 'compute_compact_logits_with_inputs'):
            (compact_logits, head_inputs) = model.compute_compact_logits_with_inputs(draft_hidden_states)
        else:
            (compact_logits, head_inputs) = (model.compute_compact_logits(draft_hidden_states), draft_hidden_states)
        (next_token_values, _, draft_next_token) = opd.propose(compact_logits, draft_hidden_states, draft_k, compact_to_target, root=True, context_offset=0, head_inputs=head_inputs)
        # propose() returns a view of reused proposal_q scratch. Both the first
        # beam and confidences[0] must survive later expansion proposals. Merely
        # view()/detach() aliases that scratch and silently rewrites tree scores.
        draft_confidences = opd.tree_root_confidences[:bsz, :draft_k]
        draft_confidences.copy_(next_token_values[:, 0, :])
        past_kv_len = draft_past_key_values_tree[0][0].shape[-2]
        init_kv_len = draft_past_key_values_tree[0][0].shape[-2]
        draft_next_token = draft_next_token.view(bsz, -1)
        (bsz, _, hidden_size) = next_feature_states.shape
        next_feature_states = next_feature_states.expand(bsz, draft_k, hidden_size)
        total_input_ids.append(draft_next_token)
        confidences.append(draft_confidences)
        attention_seen_indices = torch.arange(past_kv_len, past_kv_len + draft_k, device=device).unsqueeze(-1).unsqueeze(0).repeat(bsz, 1, 1)
        for idx_token in range(1, draft_token_length):
            draft_position_ids = draft_position_ids + 1
            total_position_ids.append(draft_position_ids.repeat(1, draft_k))
            min_dtype = torch.finfo(dtype).min
            q_length = draft_k
            draft_attention_mask = torch.zeros((q_length, past_kv_len + q_length), dtype=dtype, device=device)
            draft_attention_mask[..., init_kv_len:] = min_dtype
            draft_attention_mask = draft_attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1)
            zeros = torch.zeros((bsz, 1, q_length, idx_token), dtype=dtype, device=device)
            draft_attention_mask.scatter_(dim=-1, index=attention_seen_indices.unsqueeze(1), src=zeros)
            if isinstance(padding_positions, torch.Tensor):
                padding_positions_tensor = padding_positions
            else:
                padding_positions_indices = []
                for (batch_id, pad_positions) in enumerate(padding_positions):
                    for pos in pad_positions:
                        padding_positions_indices.append([batch_id, pos])
                if padding_positions_indices:
                    padding_positions_indices = torch.tensor(padding_positions_indices, device=model.device)
                padding_positions_tensor = padding_positions_indices
            if isinstance(padding_positions_tensor, torch.Tensor):
                if padding_positions_tensor.dtype==torch.bool:
                    draft_attention_mask.masked_fill_(padding_positions_tensor[:,None,None,:draft_attention_mask.shape[-1]],min_dtype)
                else:draft_attention_mask[padding_positions_tensor[:,0],0,:,padding_positions_tensor[:,1]]=min_dtype
            if statistical_time:
                torch.cuda.synchronize()
                check_time_start = time.time()
            draft_outputs = model(hidden_states=next_feature_states, input_ids=draft_next_token, attention_mask=draft_attention_mask, use_cache=True, past_key_values=draft_past_key_values_tree, position_ids=draft_position_ids)
            if statistical_time:
                torch.cuda.synchronize()
                total_check_time += time.time() - check_time_start
            draft_past_key_values_tree = draft_outputs['past_key_values']
            draft_hidden_states = draft_outputs['hidden_states']
            next_feature_states = draft_outputs['next_feature_states']
            context_ids = 1 + (idx_token - 1) * draft_k + torch.arange(draft_k, device=device)
            full_contexts.scatter_(1, beam_node_ids, context_ids.expand(bsz, -1))
            full_parents.append(beam_node_ids.repeat_interleave(draft_k, dim=1))
            if enabled and hasattr(model, 'compute_compact_logits_with_inputs'):
                (compact_logits, head_inputs) = model.compute_compact_logits_with_inputs(draft_hidden_states)
            else:
                (compact_logits, head_inputs) = (model.compute_compact_logits(draft_hidden_states), draft_hidden_states)
            (next_token_values, _, draft_next_token) = opd.propose(compact_logits, draft_hidden_states, draft_k, compact_to_target, context_offset=1 + (idx_token - 1) * draft_k, head_inputs=head_inputs)
            draft_confidences = draft_confidences.unsqueeze(-1) * next_token_values
            (draft_top_k_token_values, draft_top_k_token_indices) = torch.topk(draft_confidences.view(bsz, -1), k=draft_k, dim=-1)
            beam_node_ids = draft_k + draft_k * draft_k * (idx_token - 1) + draft_top_k_token_indices
            past_kv_len = draft_past_key_values_tree[0][0].shape[-2]
            total_input_ids.append(draft_next_token.view(bsz, -1))
            confidences.append(draft_confidences.view(bsz, -1))
            draft_next_token = draft_next_token.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            draft_confidences = draft_confidences.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            draft_top_k_token_indices_div_k = draft_top_k_token_indices // draft_k
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, next_feature_states.shape[-1])
            next_feature_states = next_feature_states.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, idx_token)
            attention_seen_indices = attention_seen_indices.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            cur_attention_seen_indices = torch.arange(past_kv_len, past_kv_len + draft_k, device=device).unsqueeze(-1).unsqueeze(0).repeat(bsz, 1, 1)
            attention_seen_indices = torch.concat([attention_seen_indices, cur_attention_seen_indices], dim=-1)
            parents_list.append(draft_top_k_token_indices)
        total_input_ids = torch.concat(total_input_ids, dim=-1)
        total_position_ids = torch.concat(total_position_ids, dim=1)
        confidences = torch.concat(confidences, dim=-1)
        draft_total_token = min(int(draft_total_token), int(confidences.shape[-1]))
        chosen_index = select_confidence_nodes(
            confidences, draft_total_token, kernels=opd._kernels,
            workspace=opd.confidence_key_workspace)
        tensor_tree = pack_tree(torch.cat(full_parents, dim=1), full_contexts, chosen_index, total_input_ids, draft_token_length)
        return {'trees': [], 'trees_chosen_index': None, 'tensor_tree': tensor_tree, 'next_token_trees': total_input_ids.gather(1, chosen_index), 'target_position_ids': total_position_ids.gather(1, chosen_index)}

    def model_forward(model, input_ids, attention_mask, past_key_values, position_ids=None):
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'model'):
                model = model.base_model.model
        past_seen_tokens = _cache_seq_length(past_key_values)
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        eagle_layers = getattr(model, '_fastgrpo_eagle3_capture_layers', None)
        forward_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids, 'past_key_values': past_key_values, 'use_cache': True, 'cache_position': cache_position, 'output_attentions': False, 'output_hidden_states': eagle_layers is not None, 'return_dict': True}
        try:
            outputs = model.model(**forward_kwargs)
        except TypeError:
            forward_kwargs.pop('cache_position', None)
            outputs = model.model(**forward_kwargs)
        result = {'last_hidden_state': outputs.last_hidden_state, 'target_hidden_state': outputs.last_hidden_state, 'past_key_values': outputs.past_key_values}
        if eagle_layers is not None:
            result['last_hidden_state'] = torch.cat([outputs.hidden_states[int(layer_id) + 1] for layer_id in eagle_layers], dim=-1)
        return result

    def get_attention_mask(past_seq_len, q_length, dtype, bsz=1, device='cuda', padding_positions=None):
        min_dtype = torch.finfo(dtype).min
        kv_length = past_seq_len + q_length
        attention_mask = torch.triu(torch.full((q_length, kv_length), fill_value=min_dtype, dtype=dtype, device=device), diagonal=kv_length - q_length + 1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1)
        if isinstance(padding_positions, torch.Tensor):
            padding_positions_tensor = padding_positions
            if padding_positions_tensor.dtype==torch.bool:
                attention_mask.masked_fill_(padding_positions_tensor[:,None,None,:kv_length],min_dtype)
            else:attention_mask[padding_positions_tensor[:,0],0,:,padding_positions_tensor[:,1]]=min_dtype
        elif padding_positions:
            batch_indices = []
            pos_indices = []
            for (batch_id, pad_positions) in enumerate(padding_positions):
                for pos in pad_positions:
                    batch_indices.append(batch_id)
                    pos_indices.append(pos)
            if batch_indices:
                attention_mask[batch_indices, 0, :, pos_indices] = min_dtype
        return attention_mask
    if statistical_time:
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    target_past_key_values = DynamicCache()
    avg_acc_length = [0, 0]
    total_accepted_draft_tokens = 0
    total_proposed_draft_tokens = 0
    eos_token_id = tokenizer.eos_token_id
    bsz = input_ids.shape[0]
    end_sig = [0] * bsz
    device = model.target_model.device
    update_stream = torch.cuda.Stream(device) if enabled and opd_update_stream and (torch.device(device).type == 'cuda') else None
    source_ready = update_done = None
    if update_stream is not None:
        source_ready = torch.cuda.Event()
        update_done = torch.cuda.Event()

    def wait_for_opd_update():
        ticket = opd.begin('opd_wait_ms')
        torch.cuda.current_stream(device).wait_event(update_done)
        opd.end(ticket)
    verification_batches = active_response_rounds = verified_tree_nodes = 0
    prefill_time_start = time.time()
    target_time_start = time.time()
    global total_target_time, total_draft_time, total_check_time
    (total_target_time, total_draft_time, total_check_time) = (0, 0, 0)
    all_draft_input_states = None
    all_target_hidden_states = None
    all_draft_input_ids = None
    attention_mask = attention_mask.cpu()
    initial_padding=attention_mask.eq(0).to(device)
    position_ids = [torch.sum(item) for item in attention_mask]
    past_position_ids = [item.item() - 1 for item in position_ids]
    position_ids = [torch.concat([torch.zeros(input_ids.shape[-1] - item, dtype=torch.long), torch.arange(0, item, dtype=torch.long)], dim=-1) for item in position_ids]
    position_ids = torch.stack(position_ids, dim=0)
    padding_positions = []
    for example in attention_mask:
        cur_padding_positions = set()
        for (idx, cur_attention_mask) in enumerate(example):
            if not cur_attention_mask:
                cur_padding_positions.add(idx)
        padding_positions.append(cur_padding_positions)
    input_ids = input_ids.to(device)
    attention_mask = get_attention_mask(0, attention_mask.shape[-1], model.target_model.dtype, bsz, device=device, padding_positions=padding_positions)
    position_ids = position_ids.to(device)
    with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
        target_outputs = model_forward(model.target_model, input_ids=input_ids, attention_mask=attention_mask, past_key_values=target_past_key_values, position_ids=position_ids)
        target_past_key_values = target_outputs['past_key_values']
        feature_states = target_outputs['last_hidden_state']
        target_hidden_states = target_outputs['target_hidden_state']
        target_logits = model.target_model.lm_head(target_hidden_states[:, -1:, :])
    if statistical_time:
        torch.cuda.synchronize()
        total_target_time += time.time() - target_time_start
    (target_next_token, _) = sample_target_from_logits(target_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, eos_token_id=eos_token_id)
    del _, target_logits
    draft_input_ids = torch.concat([input_ids[:, 1:], target_next_token], dim=-1)
    draft_attention_mask = attention_mask.to(model.dtype)
    initial_history = {'generated_ids': target_next_token}
    if return_all_draft_input:
        initial_history.update(features=feature_states, target_hidden=target_hidden_states, input_ids=draft_input_ids)
    history = RolloutHistory(initial_history, repeats=max(1, repeated_generate_nums or 1), max_length=max_length + max_draft_token_length + 1)
    del initial_history
    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start = time.time()
    with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
        draft_outputs = model(hidden_states=feature_states.to(model.dtype), input_ids=draft_input_ids, attention_mask=draft_attention_mask, position_ids=position_ids, use_cache=True)
    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time += time.time() - draft_time_start
    draft_past_key_values = draft_outputs['past_key_values']
    draft_hidden_states = draft_outputs['hidden_states'][:, -1:, :]
    next_feature_states = draft_outputs['next_feature_states'][:, -1:, :]
    if repeated_generate_nums is not None and repeated_generate_nums > 1:
        target_next_token = target_next_token.repeat_interleave(repeated_generate_nums, dim=0)
        target_past_key_values.batch_repeat_interleave(repeated_generate_nums)
        new_past_key_values = []
        for cur_past_key_values in draft_past_key_values:
            cur_past_key_values = [cur_past_key_values[0].repeat_interleave(repeated_generate_nums, dim=0), cur_past_key_values[1].repeat_interleave(repeated_generate_nums, dim=0)]
            new_past_key_values.append(cur_past_key_values)
        draft_past_key_values = new_past_key_values
        draft_hidden_states = draft_hidden_states.repeat_interleave(repeated_generate_nums, dim=0)
        next_feature_states = next_feature_states.repeat_interleave(repeated_generate_nums, dim=0)
        bsz *= repeated_generate_nums
        end_sig = [0] * bsz
        new_past_position_ids = []
        for cur_past_position_ids in past_position_ids:
            for _ in range(repeated_generate_nums):
                new_past_position_ids.append(cur_past_position_ids)
        past_position_ids = new_past_position_ids
        new_padding_positions = []
        for cur_padding_positions in padding_positions:
            for _ in range(repeated_generate_nums):
                new_padding_positions.append(deepcopy(cur_padding_positions))
        padding_positions = new_padding_positions
    (draft_token_length, draft_k, draft_total_token) = get_adaptive_hyperparameters(bsz, verification_capacity, max_draft_token_length, max_draft_k, max_verification_num, min_draft_token_length, draft_token_length_c)
    padding_positions_indices = []
    for (batch_id, pad_positions) in enumerate(padding_positions):
        for pos in pad_positions:
            padding_positions_indices.append([batch_id, pos])
    if padding_positions_indices:
        padding_positions_indices = torch.tensor(padding_positions_indices, device=model.device)
    padding_positions_tensor = padding_positions_indices
    past_position_ids_tensor = torch.tensor(past_position_ids, dtype=torch.int16).to(device).long()
    draft_input_states_dict = {}
    target_hidden_states_dict = {}
    draft_input_ids_dict = {}
    generated_sequences_dict = {}
    padding_positions_dict = {}
    residual_index = [_ for _ in range(bsz)]
    response_accepted_length_sum = [0 for _ in range(bsz)]
    response_verification_rounds = [0 for _ in range(bsz)]
    if getattr(model, 'is_eagle3_specforge', False):
        compact_to_target = model.compact_to_target_ids(device=draft_hidden_states.device)
    cache = getattr(model, '_opd_runtime_cache', None)
    if cache is None:
        cache = {}
        model._opd_runtime_cache = cache
    key = (enabled, opd_rank, opd_topk, opd_fast_lr, opd_visited_weight, opd_frontier_weight, opd_profile, opd_diagnostics, opd_backend,opd_train_projector)
    opd = cache.get(key)
    if opd is None:
        cache.clear()
        opd=OPDReflex(opd_rank,opd_topk,opd_fast_lr,opd_visited_weight,opd_frontier_weight,
                     opd_profile,opd_diagnostics,enabled,opd_backend,opd_train_projector)
        cache[key]=opd
    opd.start(model, bsz, compact_to_target, draft_hidden_states.shape[-1], max_contexts=1 + max_draft_k * (max_draft_token_length - 1), max_nodes=max(verification_capacity+bsz,2*bsz), max_path=max_draft_token_length + 1, max_proposal_contexts=max_draft_k)
    mask_rows_capacity = max(verification_capacity + bsz, 2 * bsz)
    mask_columns_capacity = input_ids.shape[-1] + max_length * (max_draft_token_length + 1) + max_verification_num
    workspace = getattr(model, '_opd_tree_mask_workspace', None)
    elements = mask_rows_capacity * mask_columns_capacity
    if workspace is None or workspace.numel() < elements or workspace.device != torch.device(device):
        workspace = torch.empty(elements, device=device, dtype=model.target_model.dtype)
        model._opd_tree_mask_workspace = workspace
    tree_mask_workspace = workspace
    model._opd_initial_batch=bsz;model._opd_max_path_capacity=max_draft_token_length+1
    pad_capacity=mask_columns_capacity
    pad_mask=getattr(model,'_opd_padding_workspace',None)
    if pad_mask is None or pad_mask.shape!=(bsz,pad_capacity) or pad_mask.device!=torch.device(device):
        pad_mask=torch.empty((bsz,pad_capacity),device=device,dtype=torch.bool);model._opd_padding_workspace=pad_mask
    pad_mask.zero_()
    pad_mask[:,:input_ids.shape[-1]].copy_(initial_padding.repeat_interleave(max(1,repeated_generate_nums or 1),0))
    owners=torch.arange(bsz,device=device,dtype=torch.long)
    padding_positions_tensor=pad_mask
    pad_counts=[len(pad) for pad in padding_positions]
    max_recorded_pad_column=input_ids.shape[-1]

    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start = time.time()
    with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
        outputs = draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values, draft_token_length, past_position_ids_tensor, padding_positions_tensor, draft_k=draft_k, draft_total_token=draft_total_token)
        draft_trees = outputs['trees']
        trees_chosen_index = outputs['trees_chosen_index']
        next_token_trees = outputs['next_token_trees']
        target_position_ids = outputs['target_position_ids']
        tensor_tree = outputs.get('tensor_tree')
    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time += time.time() - draft_time_start
    total_prefill_time = time.time() - prefill_time_start
    for token_num in range(1, max_length):
        past_kv_len = _cache_seq_length(target_past_key_values)
        kv_length = past_kv_len + draft_total_token + 1
        q_length = draft_total_token + 1
        verification_batches += 1
        active_response_rounds += bsz
        verified_tree_nodes += bsz * q_length
        target_trees = draft_trees
        next_token_trees = torch.concat([target_next_token, next_token_trees], dim=-1)
        target_position_ids = target_position_ids + 2
        target_position_ids = torch.concat([(past_position_ids_tensor + 1).unsqueeze(-1), target_position_ids], dim=-1)
        min_dtype = torch.finfo(model.target_model.dtype).min
        target_attention_mask = tensor_tree.attention_mask(past_kv_len, model.target_model.dtype, kernels=opd._kernels, workspace=tree_mask_workspace)
        indices = []
        if indices:
            indices = torch.tensor(indices, dtype=torch.int16).to(device).long()
            target_attention_mask[indices[:, 0], 0, indices[:, 1], indices[:, 2]] = 0
        if isinstance(padding_positions_tensor, torch.Tensor):
            target_attention_mask.masked_fill_(padding_positions_tensor[:,None,None,:kv_length],min_dtype)
        transfer_thread = None
        if statistical_time:
            torch.cuda.synchronize()
            target_time_start = time.time()
        with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
            target_outputs = model_forward(model.target_model, input_ids=next_token_trees, attention_mask=target_attention_mask, past_key_values=target_past_key_values, position_ids=target_position_ids)
            target_past_key_values = target_outputs['past_key_values']
            feature_states_tree = target_outputs['last_hidden_state']
            target_hidden_states_tree = target_outputs['target_hidden_state']
            target_outputs_logits = model.target_model.lm_head(target_hidden_states_tree)
            (target_next_token_tree, target_sampling_probs) = sample_target_from_logits(target_outputs_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, eos_token_id=eos_token_id)
        if statistical_time:
            torch.cuda.synchronize()
            total_target_time += time.time() - target_time_start
        path = trace_verified_path(tensor_tree, target_next_token_tree, eos_token_id, kernels=opd._kernels, workspace=opd.path_workspace)
        if enabled:
            teacher = target_sampling_probs if do_sample else target_next_token_tree
            if update_stream is not None:
                source_ready.record(torch.cuda.current_stream(device))
                update_stream.wait_event(source_ready)
                for shared in (teacher, tensor_tree.parents, tensor_tree.feedback_contexts):
                    shared.record_stream(update_stream)
                with torch.cuda.stream(update_stream):
                    opd.feedback(tensor_tree, path, teacher, greedy=not do_sample)
                    update_done.record(update_stream)
            else:
                opd.feedback(tensor_tree, path, teacher, greedy=not do_sample)
        scheduling_packet,next_token,chosen_index,newly_padded,last_valid_index,max_acc_length=schedule(
            path,past_kv_len,eos_token_id,opd.padded_path_workspace,opd.scheduling_packet,opd._kernels,opd=opd)
        acc_length=[row[0] for row in scheduling_packet]
        end_sig=[row[1] for row in scheduling_packet]
        for idx_tree,length in enumerate(acc_length):
            pad_counts[idx_tree]+=sum(index>=0 for index in scheduling_packet[idx_tree][3:])
            total_proposed_draft_tokens+=draft_total_token
            total_accepted_draft_tokens+=max(length-1,0)
            avg_acc_length[0]+=length;avg_acc_length[1]+=1
        output_columns=past_kv_len+torch.arange(max_acc_length,device=device)
        pad_mask[owners[:,None],output_columns[None,:]]=newly_padded
        padding_positions_tensor[:,output_columns]=newly_padded
        max_recorded_pad_column=max(max_recorded_pad_column,past_kv_len+max_acc_length)
        del target_sampling_probs, target_outputs_logits
        max_acc_length = max(acc_length)
        for (active_index, accepted_length) in enumerate(acc_length):
            if accepted_length > 0:
                original_index = residual_index[active_index]
                response_accepted_length_sum[original_index] += int(accepted_length)
                response_verification_rounds[original_index] += 1
        feature_states_index = chosen_index - past_kv_len
        target_next_token = next_token.gather(index=last_valid_index, dim=-1)
        (B, T, D) = feature_states_tree.shape
        feature_states_index = feature_states_index.unsqueeze(-1).expand(B, -1, D)
        feature_states = feature_states_tree.gather(dim=1, index=feature_states_index)
        target_hidden_index = feature_states_index[..., :1].expand(B, feature_states_index.shape[1], target_hidden_states_tree.shape[-1])
        target_hidden_states = target_hidden_states_tree.gather(dim=1, index=target_hidden_index)
        history_chunk = {'generated_ids': next_token}
        if return_all_draft_input:
            history_chunk.update(features=feature_states, target_hidden=target_hidden_states, input_ids=next_token)
        history.append(residual_index, history_chunk, owners=owners)
        del history_chunk
        finished_indices = [index for (index, finished) in enumerate(end_sig) if finished]
        if 0 not in end_sig:
            if update_stream is not None:
                wait_for_opd_update()
            break
        real_sequences_length = max(history.lengths['generated_ids']+input_ids.shape[-1]-count for count in pad_counts)
        if real_sequences_length >= max_length:
            if update_stream is not None:
                wait_for_opd_update()
            break
        if finished_indices:
            keep_rows = [row for (row, finished) in enumerate(end_sig) if not finished]
            keep = torch.tensor(keep_rows, device=device, dtype=torch.long)
            for row in finished_indices:
                original = residual_index[row]
                history.mark_finished(original)
            end_sig = [end_sig[row] for row in keep_rows]
            pad_counts=[pad_counts[row] for row in keep_rows]
            owners=owners.index_select(0,keep)
            padding_positions_tensor=padding_positions_tensor.index_select(0,keep)
            past_position_ids_tensor=past_position_ids_tensor.index_select(0,keep)
            residual_index = [residual_index[row] for row in keep_rows]
            chosen_index = chosen_index.index_select(0, keep)
            for layer in range(_cache_num_layers(target_past_key_values)):
                (key, value) = _cache_get_layer(target_past_key_values, layer)
                _cache_set_layer(target_past_key_values, layer, key.index_select(0, keep), value.index_select(0, keep))
            draft_past_key_values = [[key.index_select(0, keep), value.index_select(0, keep)] for (key, value) in draft_past_key_values]
            next_token = next_token.index_select(0, keep)
            newly_padded=newly_padded.index_select(0,keep)
            target_next_token = target_next_token.index_select(0, keep)
            feature_states = feature_states.index_select(0, keep)
            target_hidden_states = target_hidden_states.index_select(0, keep)
            last_valid_index = last_valid_index.index_select(0, keep)
            bsz = len(keep_rows)
            (draft_token_length, draft_k, draft_total_token) = get_adaptive_hyperparameters(bsz, verification_capacity, max_draft_token_length, max_draft_k, max_verification_num, min_draft_token_length, draft_token_length_c)
        positions = torch.arange(max_acc_length, device=device)[None, :] + past_kv_len
        extension = min(row[2] for row in scheduling_packet if row[1]==0)
        prefix_length = min(_cache_seq_length(target_past_key_values), past_kv_len + extension)
        full_chosen_length = past_kv_len + max_acc_length
        # OPD ONLY. Historical baseline retains its original stacked gather.
        for layer in range(_cache_num_layers(target_past_key_values)):
            key,value=_cache_get_layer(target_past_key_values,layer)
            key,value=compact_suffix_inplace(key,value,chosen_index,past_kv_len,extension,model)
            _cache_set_layer(target_past_key_values,layer,key,value)
        target_past_key_values.crop(full_chosen_length)
        draft_attention_mask = get_attention_mask(draft_past_key_values[0][0].shape[-2], max_acc_length, model.dtype, bsz, padding_positions=padding_positions_tensor)
        assert _cache_seq_length(target_past_key_values)==draft_past_key_values[0][0].shape[-2]+max_acc_length
        draft_position_ids=past_position_ids_tensor[:,None]+(~newly_padded).long().cumsum(1)
        past_position_ids_tensor=draft_position_ids[:,-1]
        if statistical_time:
            torch.cuda.synchronize()
            draft_time_start = time.time()
        with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
            if statistical_time:
                torch.cuda.synchronize()
                check_time_start = time.time()
            draft_outputs = model(hidden_states=feature_states.to(model.dtype), input_ids=next_token, attention_mask=draft_attention_mask, use_cache=True, position_ids=draft_position_ids, past_key_values=draft_past_key_values)
            if statistical_time:
                torch.cuda.synchronize()
                total_check_time += time.time() - check_time_start
            draft_past_key_values = draft_outputs['past_key_values']
            (B, S, D) = draft_outputs['hidden_states'].shape
            last_valid_index = last_valid_index.unsqueeze(-1).expand(-1, -1, D)
            draft_hidden_states = draft_outputs['hidden_states'].gather(index=last_valid_index, dim=1)
            next_feature_states = draft_outputs['next_feature_states'].gather(index=last_valid_index, dim=1)
            if update_stream is not None:
                wait_for_opd_update()
            outputs = draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values, draft_token_length, past_position_ids_tensor, padding_positions_tensor, draft_k=draft_k, draft_total_token=draft_total_token)
            draft_trees = outputs['trees']
            trees_chosen_index = outputs['trees_chosen_index']
            next_token_trees = outputs['next_token_trees']
            target_position_ids = outputs['target_position_ids']
            tensor_tree = outputs.get('tensor_tree')
        if statistical_time:
            torch.cuda.synchronize()
            total_draft_time += time.time() - draft_time_start
    post_time_start = time.time()
    padding_cpu=pad_mask[:,:max_recorded_pad_column].cpu()
    padding_positions_dict={str(row):set(torch.nonzero(mask,as_tuple=False).flatten().tolist())
        for row,mask in enumerate(padding_cpu)}
    for ori_idx,finished_history in enumerate(history.finalize()):
        generated_sequences_dict[str(ori_idx)] = finished_history['generated_ids']
        if return_all_draft_input:
            draft_input_states_dict[str(ori_idx)] = finished_history['features']
            target_hidden_states_dict[str(ori_idx)] = finished_history['target_hidden']
            draft_input_ids_dict[str(ori_idx)] = finished_history['input_ids']
        del finished_history
    del history
    bsz = len(generated_sequences_dict)
    if return_all_draft_input:
        all_draft_input_states_without_padding = []
        all_target_hidden_states_without_padding = []
        all_draft_input_ids_without_padding = []
        for idx_batch in range(bsz):
            chosen_index = []
            cur_draft_input_states = draft_input_states_dict.pop(str(idx_batch))
            cur_target_hidden_states = target_hidden_states_dict.pop(str(idx_batch))
            cur_draft_input_ids = draft_input_ids_dict.pop(str(idx_batch))
            for index in range(cur_draft_input_ids.shape[-1]):
                if index not in padding_positions_dict[str(idx_batch)]:
                    chosen_index.append(index)
            if len(chosen_index) == cur_draft_input_ids.shape[-1]:
                all_draft_input_states_without_padding.append(cur_draft_input_states)
                all_target_hidden_states_without_padding.append(cur_target_hidden_states)
                all_draft_input_ids_without_padding.append(cur_draft_input_ids)
            else:
                all_draft_input_states_without_padding.append(cur_draft_input_states[chosen_index, :])
                all_target_hidden_states_without_padding.append(cur_target_hidden_states[chosen_index, :])
                all_draft_input_ids_without_padding.append(cur_draft_input_ids[chosen_index])
        all_draft_input_states = all_draft_input_states_without_padding
        all_target_hidden_states = all_target_hidden_states_without_padding
        all_draft_input_ids = all_draft_input_ids_without_padding
    new_padding_positions = [[] for _ in range(bsz)]
    for idx_batch in range(bsz):
        cur_padding_positions = []
        for index in sorted(padding_positions_dict[str(idx_batch)]):
            if index + 1 >= input_ids.shape[-1]:
                cur_padding_positions.append(index - input_ids.shape[-1] + 1)
        new_padding_positions[idx_batch] = cur_padding_positions
    filtered_generated_token_ids = []
    max_sequence_length = 0
    for idx_batch in range(bsz):
        generated_sequence = generated_sequences_dict.pop(str(idx_batch)).tolist()
        cur_position_ids = new_padding_positions[idx_batch]
        sequence_without_padding = []
        cur_padding_index = 0
        for (idx_token, token) in enumerate(generated_sequence):
            if cur_padding_index < len(cur_position_ids):
                if idx_token == cur_position_ids[cur_padding_index]:
                    cur_padding_index += 1
                    continue
                else:
                    sequence_without_padding.append(token)
            else:
                sequence_without_padding.append(token)
            if token == eos_token_id:
                break
        filtered_generated_token_ids.append(sequence_without_padding)
        max_sequence_length = max(max_sequence_length, len(sequence_without_padding))
    draft_acceptance_rate = total_accepted_draft_tokens / max(total_proposed_draft_tokens, 1)
    opd_statistics = opd.finish()
    opd.clear()
    result = {'generated_token_ids': filtered_generated_token_ids, 'max_sequence_length': max_sequence_length, 'total_acc_length': avg_acc_length[0], 'total_acc': max_sequence_length / token_num, 'total_decoded_token_num': avg_acc_length[1], 'total_accepted_draft_tokens': total_accepted_draft_tokens, 'total_proposed_draft_tokens': total_proposed_draft_tokens, 'total_accepted_medusa_tokens': total_accepted_draft_tokens, 'total_proposed_medusa_tokens': total_proposed_draft_tokens, 'draft_acceptance_rate': draft_acceptance_rate, 'medusa_acceptance_rate': draft_acceptance_rate, 'total_time_cost': time.perf_counter() - start_time, 'target_time_cost': total_target_time, 'draft_time_cost': total_draft_time, 'check_time_cost': total_check_time, 'prefill_time_cost': total_prefill_time, 'post_time_cost': time.time() - post_time_start, 'all_draft_input_states': all_draft_input_states, 'all_target_hidden_states': all_target_hidden_states, 'all_draft_input_ids': all_draft_input_ids, 'response_accepted_length_sum': response_accepted_length_sum, 'response_verification_rounds': response_verification_rounds, 'response_generated_tokens': [len(item) for item in filtered_generated_token_ids], 'batch_verification_rounds': verification_batches, 'verification_batches': verification_batches, 'active_response_rounds': active_response_rounds, 'verified_tree_nodes': verified_tree_nodes}
    result.update(opd_statistics)
    return result
