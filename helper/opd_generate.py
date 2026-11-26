from helper.generation_telemetry import GenerationTelemetry, cap_path, response_limit, storage_bytes, validate_response_context
"""GPU tensor implementation of the FastGRPO tree with ReflexOPD proposals."""
import torch
import math
import time
from copy import deepcopy
from transformers import DynamicCache
from helper.tree_verification import pack_tree, trace_verified_path, PackedTree, VerifiedPath
from helper.opd_history import ContiguousRolloutHistory as RolloutHistory
from helper.opd_reflex import OPDReflex
from helper.method_config import resolve_method
from helper.fastgrpo_generate import get_adaptive_hyperparameters
from helper.opd_reflex import OPD_COUNTER_NAMES
from helper.opd_scheduling import schedule,compact_suffix_inplace
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_ablation import feedback_due, UPDATE_INTERVALS
from helper.opd_static_cache import OPDStaticCache,persistent_cache,swap_remove_plan
import os
from helper.opd_attention import AttentionWorkspace
from helper.static_correction import StaticCorrection, validate_static_correction
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

@torch.inference_mode()
def speculative_generate(model, input_ids, attention_mask, tokenizer, do_sample=False, repeated_generate_nums=None, temperature=0.8, top_p=0.9, top_k=None, verification_capacity=160, max_draft_token_length=5, max_draft_k=8, max_verification_num=160, min_draft_token_length=3, draft_token_length_c=0.75, statistical_time=False, return_all_draft_input=False, max_length=2048, method='opd_reflex', opd_rank=8, opd_topk=16, opd_fast_lr=0.01, opd_visited_weight=1.0, opd_frontier_weight=1.0, opd_update_stream=True, opd_profile=False, opd_diagnostics=False, opd_backend='auto', opd_train_projector=False, kv_gather_strategy='stacked', opd_update_interval_rounds=1, opd_b_sharing_scope='rollout_batch', opd_b_reset_mode='rollout', max_response_tokens=None, generation_profile="off", round_log_frequency=0,
                         opd_static_correction='none', opd_static_alpha=1., opd_static_count_mode='miss',
                         opd_cache_order=1, opd_cache_slots=1, opd_cache_bonus=1.):
    max_response_tokens = response_limit(max_response_tokens)
    validate_response_context(model, input_ids.shape[-1], max_response_tokens, max_draft_token_length)
    telemetry = GenerationTelemetry(generation_profile, model.target_model.device,
                                    repeated_generate_nums, round_log_frequency, spark=True)
    if telemetry.detailed:
        opd_profile = True
    rounds_limit = max_response_tokens or max_length
    context_capacity = input_ids.shape[-1] + max_response_tokens if max_response_tokens else max_length
    (method, _) = resolve_method(method)
    if method != 'opd_reflex':
        raise ValueError('use the upstream dispatcher for fastgrpo')
    if kv_gather_strategy not in {'stacked', 'per_layer'}:
        raise ValueError('invalid KV gather strategy')
    if opd_update_interval_rounds not in UPDATE_INTERVALS:
        raise ValueError('invalid OPD update interval')
    enabled = method == 'opd_reflex'
    adaptation_enabled = enabled and opd_update_interval_rounds > 0
    opd = None
    vocabulary_ids = None
    static = None
    static_groups = None
    static_kwargs = dict(opd_static_correction=opd_static_correction, opd_static_alpha=opd_static_alpha,
                         opd_static_count_mode=opd_static_count_mode, opd_cache_order=opd_cache_order,
                         opd_cache_slots=opd_cache_slots, opd_cache_bonus=opd_cache_bonus)
    validate_static_correction(method, opd_update_interval_rounds, static_kwargs)

    def draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values_tree, draft_token_length, past_position_ids_tensor, padding_positions, draft_k=4, draft_total_token=32):
        global total_check_time
        dtype = model.dtype
        device = model.device
        bsz = draft_hidden_states.shape[0]
        node_nums = draft_k + draft_k * draft_k * (draft_token_length - 1)
        full_parents=opd.tree_buffers['parents'][:bsz,:node_nums]
        full_parents[:,:draft_k].fill_(-1)
        full_contexts=opd.tree_buffers['contexts'][:bsz,:node_nums];full_contexts.fill_(-1)
        beam_node_ids=opd.tree_arange[:draft_k].expand(bsz,-1)
        total_input_ids=opd.tree_buffers['tokens'][:bsz,:node_nums]
        total_position_ids=opd.tree_buffers['positions'][:bsz,:node_nums]
        # Historical confidence TopK consumed a contiguous concatenated matrix.
        # Use a flat-prefix view, preserving its stride/tie path
        # without allocating/copying confidence history each round.
        confidences=opd.tree_buffers['confidence'].view(-1)[:bsz*node_nums].view(bsz,node_nums)
        draft_position_ids = opd.tree_positions[:bsz,:draft_k]
        draft_position_ids.copy_(past_position_ids_tensor[:,None])
        total_position_ids[:,:draft_k].copy_(draft_position_ids)
        head_inputs = draft_hidden_states.to(model.target_model.dtype)
        draft_logits = model.lm_head(head_inputs)
        if static is not None:
            static.apply(draft_logits, static_groups)
        beam_parents = target_next_token if static is not None and static.history is not None and opd_cache_order == 2 else None
        (next_token_values, _, draft_next_token) = opd.propose(draft_logits, draft_hidden_states, draft_k, vocabulary_ids, root=True, context_offset=0, head_inputs=head_inputs)
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
        total_input_ids[:,:draft_k].copy_(draft_next_token)
        confidences[:,:draft_k].copy_(draft_confidences)
        seen_pool,seen_scratch=opd.tree_seen
        attention_seen_indices=seen_pool[:bsz,:draft_k,:1]
        attention_seen_indices.copy_((opd.tree_arange[:draft_k]+past_kv_len)[None,:,None])
        for idx_token in range(1, draft_token_length):
            node_start=draft_k+(idx_token-1)*draft_k*draft_k
            node_end=node_start+draft_k*draft_k
            draft_position_ids.add_(1)
            total_position_ids[:,node_start:node_end].view(bsz,draft_k,draft_k).copy_(draft_position_ids[:,None,:])
            min_dtype = torch.finfo(dtype).min
            q_length = draft_k
            draft_attention_mask = opd.attention_workspace.buffer('draft_expansion',(bsz,1,q_length,past_kv_len+q_length),dtype)
            draft_attention_mask.zero_()
            draft_attention_mask[..., init_kv_len:] = min_dtype
            draft_attention_mask.scatter_(dim=-1, index=attention_seen_indices.unsqueeze(1), value=0.)
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
            telemetry.observe_cache('draft', draft_past_key_values_tree)
            draft_hidden_states = draft_outputs['hidden_states']
            next_feature_states = draft_outputs['next_feature_states']
            context_ids = opd.tree_arange[1+(idx_token-1)*draft_k:1+idx_token*draft_k]
            full_contexts.scatter_(1, beam_node_ids, context_ids.expand(bsz, -1))
            full_parents[:,node_start:node_end].view(bsz,draft_k,draft_k).copy_(beam_node_ids[:,:,None])
            head_inputs = draft_hidden_states.to(model.target_model.dtype)
            draft_logits = model.lm_head(head_inputs)
            if static is not None:
                static.apply(draft_logits, static_groups, last=draft_next_token, previous=beam_parents)
            beam_tokens = draft_next_token if beam_parents is not None else None
            (next_token_values, _, draft_next_token) = opd.propose(draft_logits, draft_hidden_states, draft_k, vocabulary_ids, context_offset=1 + (idx_token - 1) * draft_k, head_inputs=head_inputs)
            branches=opd.tree_branch_confidence.view(-1)[:bsz*draft_k*draft_k].view(bsz,draft_k,draft_k)
            torch.mul(draft_confidences.unsqueeze(-1),next_token_values,out=branches)
            draft_confidences=branches
            draft_top_k_token_values=opd.tree_top_values[:bsz,:draft_k]
            draft_top_k_token_indices=opd.tree_top_indices[:bsz,:draft_k]
            torch.topk(draft_confidences.reshape(bsz,-1),k=draft_k,dim=-1,out=(draft_top_k_token_values,draft_top_k_token_indices))
            beam_node_ids = draft_k + draft_k * draft_k * (idx_token - 1) + draft_top_k_token_indices
            past_kv_len = draft_past_key_values_tree[0][0].shape[-2]
            total_input_ids[:,node_start:node_end].copy_(draft_next_token.view(bsz,-1))
            confidences[:,node_start:node_end].copy_(draft_confidences.reshape(bsz,-1))
            draft_next_token = draft_next_token.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            draft_confidences = draft_top_k_token_values
            draft_top_k_token_indices_div_k = draft_top_k_token_indices // draft_k
            if beam_tokens is not None:
                beam_parents = beam_tokens.gather(1, draft_top_k_token_indices_div_k)
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, next_feature_states.shape[-1])
            next_feature_states = next_feature_states.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, idx_token)
            torch.gather(attention_seen_indices,-2,draft_top_k_token_indices_expanded,out=seen_scratch[:bsz,:draft_k,:idx_token])
            seen_scratch[:bsz,:draft_k,idx_token].copy_(opd.tree_arange[:draft_k]+past_kv_len)
            seen_pool,seen_scratch=seen_scratch,seen_pool
            attention_seen_indices=seen_pool[:bsz,:draft_k,:idx_token+1]
        chosen_index=torch.topk(confidences,k=draft_total_token,dim=-1).indices.sort(dim=-1).values
        tensor_tree = pack_tree(full_parents, full_contexts, chosen_index, total_input_ids, draft_token_length,workspace=opd.pack_workspace)
        if hasattr(draft_past_key_values_tree,'crop'):draft_past_key_values_tree.crop(init_kv_len)
        return {'trees': [], 'trees_chosen_index': None, 'tensor_tree': tensor_tree, 'next_token_trees': total_input_ids.gather(1, chosen_index), 'target_position_ids': total_position_ids.gather(1, chosen_index)}

    def model_forward(model, input_ids, attention_mask, past_key_values, position_ids=None):
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'model'):
                model = model.base_model.model
        past_seen_tokens = _cache_seq_length(past_key_values)
        cache_position = attention_workspace.positions('cache_position',past_seen_tokens,input_ids.shape[1])
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        hidden_states = model.model.embed_tokens(input_ids)
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in model.model.layers[:model.model.config.num_hidden_layers]:
            outputs = decoder_layer(hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, past_key_value=past_key_values,
                output_attentions=False, use_cache=True, cache_position=cache_position,
                position_embeddings=position_embeddings)
            hidden_states = outputs[0]
        hidden_states = model.model.norm(hidden_states)
        return dict(last_hidden_state=hidden_states, target_hidden_state=hidden_states,
                    past_key_values=past_key_values)

    def get_attention_mask(past_seq_len, q_length, dtype, bsz=1, device='cuda', padding_positions=None):
        if isinstance(padding_positions,torch.Tensor) and padding_positions.dtype==torch.bool:
            return attention_workspace.causal('draft_committed',past_seq_len,q_length,bsz,dtype,padding_positions)
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
    telemetry.started = start_time
    static_kv=True
    max_retained_tokens=int(os.environ.get('OPD_KV_MAX_RETAINED_TOKENS','0'))
    if max_retained_tokens<0:raise ValueError('OPD_KV_MAX_RETAINED_TOKENS must be nonnegative')
    kv_capacity=256
    rollout_batch_capacity=input_ids.shape[0]*max(1,repeated_generate_nums or 1)
    target_past_key_values = persistent_cache(model,'_opd_target_kv_pool',input_ids.shape[0],max(1,repeated_generate_nums or 1),model.target_model.device,model.target_model.dtype) if static_kv else DynamicCache()
    avg_acc_length = [0, 0]
    total_accepted_draft_tokens = 0
    total_proposed_draft_tokens = 0
    eos_token_id = tokenizer.eos_token_id
    bsz = input_ids.shape[0]
    end_sig = [0] * bsz
    device = model.target_model.device
    attention_workspace=getattr(model,'_opd_attention_workspace',None)
    if attention_workspace is None or attention_workspace.device!=torch.device(device):
        attention_workspace=AttentionWorkspace(device);model._opd_attention_workspace=attention_workspace
    update_stream = torch.cuda.Stream(device) if adaptation_enabled and opd_update_stream and (torch.device(device).type == 'cuda') else None
    source_ready = update_done = None
    if update_stream is not None:
        source_ready = torch.cuda.Event()
        update_done = torch.cuda.Event()

    def wait_for_opd_update():
        ticket = opd.begin('opd_wait_ms')
        torch.cuda.current_stream(device).wait_event(update_done)
        opd.end(ticket)
    verification_batches = active_response_rounds = verified_tree_nodes = 0
    feedback_rounds = 0
    round_feedback_due = False
    prefill_ticket = telemetry.begin("prefill")
    prefill_time_start = time.time()
    target_time_start = time.time()
    global total_target_time, total_draft_time, total_check_time
    (total_target_time, total_draft_time, total_check_time) = (0, 0, 0)
    all_draft_input_states = None
    all_draft_input_ids = None
    attention_mask = attention_mask.cpu()
    initial_padding=attention_mask.eq(0).to(device)
    position_ids = [torch.sum(item) for item in attention_mask]
    past_position_ids = [item.item() - 1 for item in position_ids]
    prompt_lengths = [position + 1 for position in past_position_ids]
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
    sample_ticket = telemetry.begin("target_sampling")
    (target_next_token, _, prefill_metadata) = sample_target_with_metadata(target_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, eos_token_id=eos_token_id)
    telemetry.end(sample_ticket)
    del _, prefill_metadata, target_logits
    prefill_eos = (target_next_token[:, 0].eq(eos_token_id).tolist() if max_response_tokens else None)
    draft_input_ids = torch.concat([input_ids[:, 1:], target_next_token], dim=-1)
    draft_attention_mask = attention_mask.to(model.dtype)
    initial_history = {'generated_ids': target_next_token}
    if return_all_draft_input:
        initial_history.update(features=feature_states, input_ids=draft_input_ids)
    history = RolloutHistory(initial_history, repeats=max(1, repeated_generate_nums or 1), max_length=context_capacity + max_draft_token_length + 1)
    del initial_history
    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start = time.time()
    with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
        draft_outputs = model(hidden_states=feature_states.to(model.dtype), input_ids=draft_input_ids, attention_mask=draft_attention_mask, position_ids=position_ids, use_cache=True,
                              past_key_values=persistent_cache(model,'_opd_draft_kv_pool',input_ids.shape[0],max(1,repeated_generate_nums or 1),device,model.dtype) if static_kv else None)
    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time += time.time() - draft_time_start
    draft_past_key_values = draft_outputs['past_key_values']
    draft_hidden_states = draft_outputs['hidden_states'][:, -1:, :].clone()
    next_feature_states = draft_outputs['next_feature_states'][:, -1:, :].clone()
    del attention_mask,draft_attention_mask,draft_outputs,target_outputs,feature_states,target_hidden_states
    if repeated_generate_nums is not None and repeated_generate_nums > 1:
        target_next_token = target_next_token.repeat_interleave(repeated_generate_nums, dim=0)
        target_past_key_values.batch_repeat_interleave(repeated_generate_nums)
        if static_kv:draft_past_key_values.batch_repeat_interleave(repeated_generate_nums)
        else:
            new_past_key_values = []
            for cur_past_key_values in draft_past_key_values:
                new_past_key_values.append([x.repeat_interleave(repeated_generate_nums,dim=0) for x in cur_past_key_values])
            draft_past_key_values=new_past_key_values
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
    draft_input_ids_dict = {}
    generated_sequences_dict = {}
    padding_positions_dict = {}
    residual_index = list(range(bsz))
    if max_response_tokens:
        end_sig = [int(eos or max_response_tokens == 1) for eos in prefill_eos
                   for _ in range(repeated_generate_nums or 1)]
    telemetry.observe_cache("target", target_past_key_values)
    telemetry.observe_cache("draft", draft_past_key_values)
    telemetry.end(prefill_ticket)
    response_accepted_length_sum = [0 for _ in range(bsz)]
    response_verification_rounds = [0 for _ in range(bsz)]
    vocabulary_ids = model.full_vocabulary_ids
    if enabled:
        cache = getattr(model, '_opd_runtime_cache', None)
        if cache is None:cache={};model._opd_runtime_cache=cache
        key=(enabled,opd_rank,opd_topk,opd_fast_lr,opd_visited_weight,opd_frontier_weight,opd_profile,opd_diagnostics,opd_backend,opd_train_projector,opd_update_interval_rounds,opd_b_sharing_scope,opd_b_reset_mode)
        opd=cache.get(key)
        if opd is None:
            cache.clear()
            opd=OPDReflex(opd_rank,opd_topk,opd_fast_lr,opd_visited_weight,opd_frontier_weight,
                         opd_profile,opd_diagnostics,adaptation_enabled,opd_backend,opd_train_projector and adaptation_enabled,
                         b_sharing_scope=opd_b_sharing_scope,b_reset_mode=opd_b_reset_mode)
            cache[key]=opd
    opd.start(model, bsz, vocabulary_ids, draft_hidden_states.shape[-1], max_contexts=1 + max_draft_k * (max_draft_token_length - 1), max_nodes=max(verification_capacity+bsz,2*bsz), max_path=max_draft_token_length + 1, max_proposal_contexts=max_draft_k,
              prompt_groups=input_ids.shape[0],responses_per_prompt=repeated_generate_nums or 1)
    opd.async_updates=update_stream is not None
    mask_columns_capacity = input_ids.shape[-1] + rounds_limit * (max_draft_token_length + 1) + max_verification_num
    model._opd_initial_batch=bsz;model._opd_max_path_capacity=max_draft_token_length+1
    pad_capacity=mask_columns_capacity
    pad_mask=getattr(model,'_opd_padding_workspace',None)
    if pad_mask is None or pad_mask.shape!=(bsz,pad_capacity) or pad_mask.device!=torch.device(device):
        pad_mask=torch.empty((bsz,pad_capacity),device=device,dtype=torch.bool);model._opd_padding_workspace=pad_mask
    pad_mask.zero_()
    pad_mask[:,:input_ids.shape[-1]].copy_(initial_padding.repeat_interleave(max(1,repeated_generate_nums or 1),0))
    owners=torch.arange(bsz,device=device,dtype=torch.long)
    if opd_static_correction != 'none':
        static_groups = opd.row_groups if opd.grouped else torch.zeros_like(owners)
        static_history = None
        if opd_static_correction == 'context_cache':
            columns = torch.arange(input_ids.shape[1], device=device)[None, :]
            last_prompt_column = torch.where(initial_padding, -1, columns).amax(1, keepdim=True)
            last_prompt = input_ids.gather(1, last_prompt_column.clamp_min(0)).masked_fill(last_prompt_column < 0, -1)
            last_prompt = last_prompt.repeat_interleave(max(1, repeated_generate_nums or 1), dim=0)
            static_history = torch.cat((last_prompt, target_next_token), dim=1)
        static = StaticCorrection(vocabulary_ids.numel(), opd.num_groups, device, bsz,
                                  history=static_history, **static_kwargs)
    padding_positions_tensor=pad_mask
    pad_counts=[len(pad) for pad in padding_positions]
    max_recorded_pad_column=input_ids.shape[-1]

    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start = time.time()
    with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
        proposal_ticket = telemetry.begin("draft_proposal")
        outputs = draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values, draft_token_length, past_position_ids_tensor, padding_positions_tensor, draft_k=draft_k, draft_total_token=draft_total_token)
        telemetry.end(proposal_ticket)
        draft_trees = outputs['trees']
        trees_chosen_index = outputs['trees_chosen_index']
        next_token_trees = outputs['next_token_trees']
        target_position_ids = outputs['target_position_ids']
        tensor_tree = outputs.get('tensor_tree')
    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time += time.time() - draft_time_start
    total_prefill_time = time.time() - prefill_time_start
    canonical_to_physical=physical_to_canonical=None
    feedback_path=None
    def compact_teacher_metadata(tokens,probs,sorted_metadata):
        nonlocal path,feedback_path
        trace_ticket = telemetry.begin("verified_path_tracing")
        feedback_path=trace_verified_path(feedback_tree,tokens,eos_token_id,kernels=opd._kernels,workspace=opd.path_workspace)
        if max_response_tokens:
            ordered_remaining = remaining if canonical_to_physical is None else remaining.index_select(0, canonical_to_physical)
            cap_path(feedback_path, ordered_remaining)
        telemetry.end(trace_ticket)
        path=feedback_path if physical_to_canonical is None else VerifiedPath(
            *[x.index_select(0,physical_to_canonical) for x in (feedback_path.tokens,feedback_path.packed_indices,feedback_path.feedback_contexts,feedback_path.lengths)])
        if round_feedback_due:
            return opd.prepare_compact_teacher(feedback_tree,feedback_path,probs if do_sample else tokens,sorted_metadata,greedy=not do_sample)
        return None
    counter_indices = [OPD_COUNTER_NAMES.index(name) for name in ("opd_selected_states", "opd_updates")]
    def apply_feedback(tree, path, teacher, metadata):
        collect = telemetry.samples(verification_batches)
        before = torch.stack([opd.counters[i] for i in counter_indices]) if collect else None
        opd.feedback(tree, path, teacher, greedy=not do_sample, sampling_metadata=metadata)
        return torch.stack([opd.counters[i] for i in counter_indices]) - before if collect else None

    token_num = 1
    for token_num in range(1, rounds_limit):
        if max_response_tokens:
            if all(end_sig):
                break
            host_remaining = [0 if end_sig[i] else max_response_tokens - 1 - response_accepted_length_sum[owner]
                              for i, owner in enumerate(residual_index)]
            remaining = torch.tensor(host_remaining, device=device)
        tree_ticket = telemetry.begin("tree_attention_mask")
        past_kv_len = _cache_seq_length(target_past_key_values)
        kv_length = past_kv_len + draft_total_token + 1
        q_length = draft_total_token + 1
        verification_batches += 1
        round_feedback_due = feedback_due(verification_batches, opd_update_interval_rounds)
        active_response_rounds += sum(not done for done in end_sig) if max_response_tokens else bsz
        verified_tree_nodes += bsz * q_length
        target_trees = draft_trees
        tree_inputs=attention_workspace.buffer('target_tokens',(bsz,q_length),torch.long)
        tree_inputs[:,:1].copy_(target_next_token);tree_inputs[:,1:].copy_(next_token_trees)
        next_token_trees=tree_inputs
        positions_out=attention_workspace.buffer('target_positions',(bsz,q_length),torch.long)
        torch.add(target_position_ids,2,out=positions_out[:,1:])
        torch.add(past_position_ids_tensor,1,out=positions_out[:,0])
        target_position_ids=positions_out
        min_dtype = torch.finfo(model.target_model.dtype).min
        tree_mask_workspace=attention_workspace.buffer('tree_mask',(bsz*q_length*kv_length,),model.target_model.dtype)
        target_attention_mask = tensor_tree.attention_mask(past_kv_len, model.target_model.dtype, kernels=opd._kernels, workspace=tree_mask_workspace)
        indices = []
        if indices:
            indices = torch.tensor(indices, dtype=torch.int16).to(device).long()
            target_attention_mask[indices[:, 0], 0, indices[:, 1], indices[:, 2]] = 0
        if isinstance(padding_positions_tensor, torch.Tensor):
            target_attention_mask.masked_fill_(padding_positions_tensor[:,None,None,:kv_length],min_dtype)
        telemetry.end(tree_ticket)
        transfer_thread = None
        if statistical_time:
            torch.cuda.synchronize()
            target_time_start = time.time()
        with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
            verify_ticket = telemetry.begin("target_verification")
            target_outputs = model_forward(model.target_model, input_ids=next_token_trees, attention_mask=target_attention_mask, past_key_values=target_past_key_values, position_ids=target_position_ids)
            target_past_key_values = target_outputs['past_key_values']
            feature_states_tree = target_outputs['last_hidden_state']
            target_hidden_states_tree = target_outputs['target_hidden_state']
            # Native sampler must see original response order, even when KV
            # physical rows swap-remove. Gather H-sized head INPUT, not [N,V]
            # logits/probs. No additional head/transformer forward.
            head_input=target_hidden_states_tree if canonical_to_physical is None else target_hidden_states_tree.index_select(0,canonical_to_physical)
            target_outputs_logits = model.target_model.lm_head(head_input)
            telemetry.end(verify_ticket)
            telemetry.observe_cache("target", target_past_key_values)
            del head_input
            feedback_tree=tensor_tree if canonical_to_physical is None else PackedTree(
                *[x.index_select(0,canonical_to_physical) for x in (tensor_tree.parents,tensor_tree.tokens,tensor_tree.feedback_contexts)],tensor_tree.max_depth)
            opd.feedback_row_map=canonical_to_physical
            path=None
            sample_ticket = telemetry.begin("target_sampling")
            (target_next_token_tree, target_sampling_probs, sampling_metadata) = sample_target_with_metadata(target_outputs_logits, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, eos_token_id=eos_token_id,metadata_builder=compact_teacher_metadata,return_probs=False)
            telemetry.end(sample_ticket)
            del target_outputs_logits
        if statistical_time:
            torch.cuda.synchronize()
            total_target_time += time.time() - target_time_start
        if path is None:
            path = trace_verified_path(tensor_tree, target_next_token_tree, eos_token_id, kernels=opd._kernels, workspace=opd.path_workspace)
        round_feedback = torch.zeros(2, device=device) if telemetry.samples(verification_batches) and not round_feedback_due else None
        if round_feedback_due:
            feedback_rounds += 1
            teacher = None # compact metadata owns every teacher coordinate needed
            if update_stream is not None:
                source_ready.record(torch.cuda.current_stream(device))
                update_stream.wait_event(source_ready)
                for shared in (feedback_tree.parents, feedback_tree.feedback_contexts):
                    shared.record_stream(update_stream)
                if canonical_to_physical is not None:canonical_to_physical.record_stream(update_stream)
                if opd.grouped:opd.row_groups.record_stream(update_stream)
                if sampling_metadata is not None:
                    for shared in sampling_metadata:shared.record_stream(update_stream)
                with torch.cuda.stream(update_stream):
                    round_feedback = apply_feedback(feedback_tree, feedback_path, teacher, sampling_metadata)
                    update_done.record(update_stream)
                    if opd.carry_b:opd.model._opd_persistent_b['pending_event']=update_done
            else:
                round_feedback = apply_feedback(feedback_tree, feedback_path, teacher, sampling_metadata)
            # record_stream above keeps asynchronous readers safe. Drop this
            # alias too, otherwise previous-round full probs survive until the
            # NEXT sampler has already allocated its full-vocabulary arrays.
            del teacher
        scheduling_ticket = telemetry.begin("scheduling_batching")
        scheduling_packet,next_token,chosen_index,newly_padded,last_valid_index,max_acc_length=schedule(
            path,past_kv_len,eos_token_id,opd.padded_path_workspace,opd.scheduling_packet,opd._kernels,opd=opd)
        acc_length=[row[0] for row in scheduling_packet]
        end_sig=[row[1] for row in scheduling_packet]
        if max_response_tokens:
            for row, limit in zip(scheduling_packet, host_remaining):
                if row[0] == limit:
                    row[1] = 1
            end_sig = [row[1] for row in scheduling_packet]
            last_valid_index = last_valid_index.clamp_min(0)
        for idx_tree,length in enumerate(acc_length):
            pad_counts[idx_tree]+=sum(index>=0 for index in scheduling_packet[idx_tree][3:])
            total_proposed_draft_tokens+=draft_total_token
            total_accepted_draft_tokens+=max(length-1,0)
            avg_acc_length[0]+=length;avg_acc_length[1]+=int(length>0)
        output_columns=attention_workspace.positions('output_columns',past_kv_len,max_acc_length)
        pad_mask[owners[:,None],output_columns[None,:]]=newly_padded
        padding_positions_tensor[:,output_columns]=newly_padded
        max_recorded_pad_column=max(max_recorded_pad_column,past_kv_len+max_acc_length)
        del target_sampling_probs, sampling_metadata
        max_acc_length = max(acc_length)
        for (active_index, accepted_length) in enumerate(acc_length):
            if accepted_length > 0:
                original_index = residual_index[active_index]
                response_accepted_length_sum[original_index] += int(accepted_length)
                response_verification_rounds[original_index] += 1
        telemetry.round(verification_batches, residual_index, prompt_lengths,
                        response_accepted_length_sum, acc_length, end_sig, draft_total_token,
                        draft_token_length, draft_k, round_feedback,
                        max_budget=max(1, min(verification_capacity // bsz, max_verification_num)))
        feature_states_index = chosen_index - past_kv_len
        if static is not None:
            static.observe(path, target_next_token, static_groups, owners, tensor_tree)
        target_next_token = next_token.gather(index=last_valid_index, dim=-1)
        (B, T, D) = feature_states_tree.shape
        feature_states_index = feature_states_index.unsqueeze(-1).expand(B, -1, D)
        feature_states = feature_states_tree.gather(dim=1, index=feature_states_index)
        target_hidden_states = feature_states
        # Gathered accepted features own storage; release whole-tree activations
        # before the next target forward, not after its new outputs are allocated.
        del target_outputs, feature_states_tree, target_hidden_states_tree
        history_chunk = {'generated_ids': next_token}
        if return_all_draft_input:
            history_chunk.update(features=feature_states, input_ids=next_token)
        history.append(residual_index, history_chunk, owners=owners)
        del history_chunk
        telemetry.end(scheduling_ticket)
        finished_indices = [index for (index, finished) in enumerate(end_sig) if finished]
        if 0 not in end_sig:
            if update_stream is not None:
                wait_for_opd_update()
            break
        real_sequences_length = max(history.lengths['generated_ids']+input_ids.shape[-1]-count for count in pad_counts)
        if not max_response_tokens and real_sequences_length >= max_length:
            if update_stream is not None:
                wait_for_opd_update()
            break
        if finished_indices:
            compaction_timer=opd.begin('kv_batch_compaction_ms')
            keep_rows,move_sources,move_destinations=swap_remove_plan(end_sig)
            keep = torch.tensor(keep_rows, device=device, dtype=torch.long)
            sources=torch.tensor(move_sources,device=device,dtype=torch.long)
            destinations=torch.tensor(move_destinations,device=device,dtype=torch.long)
            for row in finished_indices:
                original = residual_index[row]
                history.mark_finished(original)
            end_sig = [end_sig[row] for row in keep_rows]
            pad_counts=[pad_counts[row] for row in keep_rows]
            owners=owners.index_select(0,keep)
            padding_positions_tensor=padding_positions_tensor.index_select(0,keep)
            past_position_ids_tensor=past_position_ids_tensor.index_select(0,keep)
            residual_index = [residual_index[row] for row in keep_rows]
            opd.compact_group_rows(keep,residual_index)
            if static is not None:
                static.compact(keep)
                static_groups = opd.row_groups if opd.grouped else static_groups.index_select(0, keep)
            chosen_index = chosen_index.index_select(0, keep)
            if static_kv:
                target_past_key_values.swap_remove(len(keep_rows),sources,destinations)
            else:
                for layer in range(_cache_num_layers(target_past_key_values)):
                    (key, value) = _cache_get_layer(target_past_key_values, layer)
                    _cache_set_layer(target_past_key_values, layer, key.index_select(0, keep), value.index_select(0, keep))
            if static_kv:draft_past_key_values.swap_remove(len(keep_rows),sources,destinations)
            else:draft_past_key_values = [[key.index_select(0, keep), value.index_select(0, keep)] for (key, value) in draft_past_key_values]
            opd.end(compaction_timer)
            next_token = next_token.index_select(0, keep)
            newly_padded=newly_padded.index_select(0,keep)
            target_next_token = target_next_token.index_select(0, keep)
            feature_states = feature_states.index_select(0, keep)
            target_hidden_states = target_hidden_states.index_select(0, keep)
            last_valid_index = last_valid_index.index_select(0, keep)
            bsz = len(keep_rows)
            order=sorted(range(bsz),key=residual_index.__getitem__)
            if order==list(range(bsz)):canonical_to_physical=physical_to_canonical=None
            else:
                canonical_to_physical=torch.tensor(order,device=device,dtype=torch.long)
                inverse=[0]*bsz
                for canonical,physical in enumerate(order):inverse[physical]=canonical
                physical_to_canonical=torch.tensor(inverse,device=device,dtype=torch.long)
            (draft_token_length, draft_k, draft_total_token) = get_adaptive_hyperparameters(bsz, verification_capacity, max_draft_token_length, max_draft_k, max_verification_num, min_draft_token_length, draft_token_length_c)
        extension = min(row[2] for row in scheduling_packet if row[1]==0)
        prefix_length = min(_cache_seq_length(target_past_key_values), past_kv_len + extension)
        full_chosen_length = past_kv_len + max_acc_length
        # OPD ONLY. Historical baseline retains its original stacked gather.
        compaction_timer=opd.begin('kv_suffix_compaction_ms')
        for layer in range(_cache_num_layers(target_past_key_values)):
            key,value=_cache_get_layer(target_past_key_values,layer)
            key,value=compact_suffix_inplace(key,value,chosen_index,past_kv_len,extension,model)
            _cache_set_layer(target_past_key_values,layer,key,value)
            del key,value  # do not pin a previous pool view across geometric grow
        target_past_key_values.crop(full_chosen_length)
        opd.end(compaction_timer)
        draft_attention_mask = get_attention_mask(draft_past_key_values[0][0].shape[-2], max_acc_length, model.dtype, bsz, padding_positions=padding_positions_tensor)
        assert _cache_seq_length(target_past_key_values)==draft_past_key_values[0][0].shape[-2]+max_acc_length
        draft_position_ids=attention_workspace.buffer('draft_positions',(bsz,max_acc_length),torch.long)
        torch.cumsum(~newly_padded,1,out=draft_position_ids)
        draft_position_ids.add_(past_position_ids_tensor[:,None])
        past_position_ids_tensor=attention_workspace.buffer('past_positions',(bsz,),torch.long)
        past_position_ids_tensor.copy_(draft_position_ids[:,-1])
        if statistical_time:
            torch.cuda.synchronize()
            draft_time_start = time.time()
        with torch.amp.autocast(str(model.target_model.device), dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16):
            if statistical_time:
                torch.cuda.synchronize()
                check_time_start = time.time()
            advance_ticket = telemetry.begin("draft_advance")
            draft_outputs = model(hidden_states=feature_states.to(model.dtype), input_ids=next_token, attention_mask=draft_attention_mask, use_cache=True, position_ids=draft_position_ids, past_key_values=draft_past_key_values)
            if statistical_time:
                torch.cuda.synchronize()
                total_check_time += time.time() - check_time_start
            draft_past_key_values = draft_outputs['past_key_values']
            telemetry.end(advance_ticket)
            (B, S, D) = draft_outputs['hidden_states'].shape
            last_valid_index = last_valid_index.unsqueeze(-1).expand(-1, -1, D)
            draft_hidden_states = draft_outputs['hidden_states'].gather(index=last_valid_index, dim=1)
            next_feature_states = draft_outputs['next_feature_states'].gather(index=last_valid_index, dim=1)
            if update_stream is not None:
                wait_for_opd_update()
            telemetry.observe_cache("draft", draft_past_key_values)
            proposal_ticket = telemetry.begin("draft_proposal")
            outputs = draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values, draft_token_length, past_position_ids_tensor, padding_positions_tensor, draft_k=draft_k, draft_total_token=draft_total_token)
            telemetry.end(proposal_ticket)
            draft_trees = outputs['trees']
            trees_chosen_index = outputs['trees_chosen_index']
            next_token_trees = outputs['next_token_trees']
            target_position_ids = outputs['target_position_ids']
            tensor_tree = outputs.get('tensor_tree')
        if statistical_time:
            torch.cuda.synchronize()
            total_draft_time += time.time() - draft_time_start
    other_ticket = telemetry.begin("other_generation_overhead")
    post_time_start = time.time()
    padding_cpu=pad_mask[:,:max_recorded_pad_column].cpu()
    padding_positions_dict={str(row):set(torch.nonzero(mask,as_tuple=False).flatten().tolist())
        for row,mask in enumerate(padding_cpu)}
    for ori_idx,finished_history in enumerate(history.finalize()):
        generated_sequences_dict[str(ori_idx)] = finished_history['generated_ids']
        if return_all_draft_input:
            draft_input_states_dict[str(ori_idx)] = finished_history['features']
            draft_input_ids_dict[str(ori_idx)] = finished_history['input_ids']
        del finished_history
    del history
    bsz = len(generated_sequences_dict)
    if return_all_draft_input:
        all_draft_input_states_without_padding = []
        all_draft_input_ids_without_padding = []
        for idx_batch in range(bsz):
            chosen_index = []
            cur_draft_input_states = draft_input_states_dict.pop(str(idx_batch))
            cur_draft_input_ids = draft_input_ids_dict.pop(str(idx_batch))
            for index in range(cur_draft_input_ids.shape[-1]):
                if index not in padding_positions_dict[str(idx_batch)]:
                    chosen_index.append(index)
            if len(chosen_index) == cur_draft_input_ids.shape[-1]:
                all_draft_input_states_without_padding.append(cur_draft_input_states)
                all_draft_input_ids_without_padding.append(cur_draft_input_ids)
            else:
                all_draft_input_states_without_padding.append(cur_draft_input_states[chosen_index, :])
                all_draft_input_ids_without_padding.append(cur_draft_input_ids[chosen_index])
        all_draft_input_states = all_draft_input_states_without_padding
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
    telemetry.end(other_ticket)
    opd_statistics = opd.finish()
    opd.clear()
    result = {'generated_token_ids': filtered_generated_token_ids, 'max_sequence_length': max_sequence_length, 'total_acc_length': avg_acc_length[0], 'total_acc': max_sequence_length / token_num, 'total_decoded_token_num': avg_acc_length[1], 'total_accepted_draft_tokens': total_accepted_draft_tokens, 'total_proposed_draft_tokens': total_proposed_draft_tokens, 'total_accepted_medusa_tokens': total_accepted_draft_tokens, 'total_proposed_medusa_tokens': total_proposed_draft_tokens, 'draft_acceptance_rate': draft_acceptance_rate, 'medusa_acceptance_rate': draft_acceptance_rate, 'total_time_cost': time.perf_counter() - start_time, 'target_time_cost': total_target_time, 'draft_time_cost': total_draft_time, 'check_time_cost': total_check_time, 'prefill_time_cost': total_prefill_time, 'post_time_cost': time.time() - post_time_start, 'all_draft_input_states': all_draft_input_states, 'all_draft_input_ids': all_draft_input_ids, 'response_accepted_length_sum': response_accepted_length_sum, 'response_verification_rounds': response_verification_rounds, 'response_generated_tokens': [len(item) for item in filtered_generated_token_ids], 'batch_verification_rounds': verification_batches, 'verification_batches': verification_batches, 'active_response_rounds': active_response_rounds, 'verified_tree_nodes': verified_tree_nodes}
    result.update(opd_statistics)
    if static is not None:
        result.update(static.stats())
    result.update(opd_feedback_rounds=feedback_rounds,
                  opd_skipped_rounds=verification_batches-feedback_rounds)
    if not enabled:result.update({name:0. for name in OPD_COUNTER_NAMES})
    result['opd_proposal_profile_effective']=opd.profile_path
    result['opd_host_syncs']=opd.host_sync_count
    result['opd_host_syncs_per_round']=opd.host_sync_count/max(verification_batches,1)
    if static_kv:
        for prefix,cache in (('target',target_past_key_values),('draft',draft_past_key_values)):
            result.update({f'opd_{prefix}_{key}':value for key,value in cache.statistics().items()})
            cache.end_rollout(max_retained_tokens)
    result['opd_attention_workspace_bytes']=attention_workspace.memory_bytes()+opd.attention_workspace.memory_bytes()
    result['total_time_cost']=time.perf_counter()-start_time
    memory = None
    if telemetry.enabled:
        matrices = opd.num_groups if opd.enabled else 0
        workspace_names = (
            'bitmap', 'active_ids', 'active_count', 'active_slots', 'dispatch_count', 'dispatch_snapshot',
            'row_groups', 'group_first', 'proposal_u', 'proposal_q', 'proposal_ids', 'proposal_norm', 'sparse_scores',
            'score_workspace', 'head_cache', 'u_cache', 'ids_cache', 'q_cache', 'norm_cache',
            'selected_weights', 'selected_kind', 'selected_ids', 'selected_count', 'teacher_p',
            'teacher_ids', 'teacher_mass', 'teacher_q', 'teacher_draft_p', 'union_ids', 'union_g',
            'state_stats', 'round_weight', 'group_weight', 'counters', 'projector_head',
            'projector_v', 'projector_delta',
        )
        workspace = [getattr(opd, name, None) for name in workspace_names]
        workspace += opd.proposal_tiles + getattr(opd, 'teacher_tiles', [])
        workspace += [getattr(model, name, None) for name in
                      ('opd_projector_grad_sum', 'opd_projector_grad_weight')]
        memory = dict(correction_matrix_bytes=storage_bytes([opd.B_fast]),
                      correction_matrix_count=matrices, dense_fp32_correction_bytes=matrices*opd.vocab*opd.rank*4,
                      projector_bytes=storage_bytes([getattr(opd, 'projector', None)]),
                      correction_workspace_bytes=storage_bytes(workspace) if opd.enabled else 0)
    return telemetry.finish(result, opd_memory=memory, prompt_lengths=prompt_lengths)
