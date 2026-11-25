"""Objectives copied from sources/FastGRPO/grpo_speculative.py.

Only formerly global configuration is passed explicitly. In particular, the
upstream final draft microbatch is NOT divided by draft_accumulation_steps.
Default accumulation=1; retaining this behavior makes nondefault parity testable.
"""
import torch
from torch.utils.checkpoint import checkpoint

def compute_target_loss(logits,ref_logits,old_logits,labels,mask,reward,epsilon,beta,grpo_iteration):

    logits = logits[...,:-1,:].float()
    mask = mask[...,:-1]

    labels = labels.to(logits.device)
    labels = labels[..., 1:]
    
    logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if grpo_iteration==0:
        ref_logits = ref_logits[...,:-1,:].float()
        ref_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2).detach()
        old_logps = logps.clone().detach()
    else:
        ref_logps=ref_logits
        old_logps=old_logits

    return target_loss_from_logps(logps, ref_logps, old_logps, mask, reward, epsilon, beta)


def target_loss_from_logps(logps, ref_logps, old_logps, mask, reward, epsilon, beta):
    coef1=torch.exp(logps-old_logps)
    coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
    loss1=torch.min(coef1*reward,coef2*reward)

    coef3=ref_logps-logps
    loss2=torch.exp(coef3)-coef3-1

    loss=-(loss1-beta*loss2)
    loss=loss*mask
    loss=loss.sum(-1)/mask.sum(-1)
    
    loss1=loss1*mask
    loss1=loss1.sum(-1)/mask.sum(-1)
    abs_loss1=torch.sum(torch.abs(loss1))
    loss2=loss2*mask
    loss2=loss2.sum(-1)/mask.sum(-1)
    
    return loss.sum(-1),abs_loss1,loss2.sum(-1),old_logps,ref_logps

def _chunked_draft_losses(model, draft_outputs, features, mask, chunk_size):
    """Apply the original draft objectives with bounded vocabulary storage."""
    mask = mask[:, :-1]
    count = mask.sum(-1).clamp_min(1)
    next_features = draft_outputs['next_feature_states'][:, :-1]
    feature_loss = torch.nn.functional.smooth_l1_loss(
        next_features.float(), features[:, 1:].float(), reduction='none').mean(-1)
    loss1 = ((feature_loss * mask).sum(-1) / count).sum() * 2.0

    def cross_entropy(draft_hidden, target_hidden):
        draft_hidden = draft_hidden.to(model.target_model.dtype)
        with torch.no_grad():
            target_p = model.target_model.lm_head(target_hidden.to(draft_hidden.dtype)).float().softmax(-1)
        draft_p = model.lm_head(draft_hidden).float().softmax(-1)
        return (target_p * draft_p.log()).sum(-1)

    length = mask.shape[-1]
    hidden = draft_outputs['hidden_states']
    chunks = []
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        chunks.append(checkpoint(cross_entropy, hidden[:, start:end], features[:, start+1:end+1],
                                 use_reentrant=False, preserve_rng_state=False))
    loss2 = -((torch.cat(chunks, dim=-1) * mask).sum(-1) / count).sum() * 0.1
    return loss1, loss2


def _full_draft_losses(model, draft_outputs, features, mask):
    next_features = draft_outputs['next_feature_states']
    draft_hidden = draft_outputs['hidden_states'].to(model.target_model.dtype)
    draft_logits = model.lm_head(draft_hidden)
    with torch.no_grad():
        target_logits = model.target_model.lm_head(features.to(model.target_model.dtype))
        target_p = target_logits[:, 1:].float().softmax(dim=-1).detach()

    loss_mask = mask[:, :-1]
    denom = loss_mask.sum(-1)
    feature = torch.nn.functional.smooth_l1_loss(
        next_features[:, :-1].float(), features[:, 1:].float(), reduction='none')
    loss1 = ((feature.mean(dim=-1) * loss_mask).sum(-1) / denom).sum(-1) * 2.0
    draft_p = draft_logits[:, :-1].float().softmax(dim=-1)
    loss2 = -(((target_p * torch.log(draft_p)).sum(dim=-1) * loss_mask).sum(-1) / denom).sum(-1) * 0.1
    return loss1, loss2


def training_draft_model(model,outputs,prompt_mask,*,repeated_generate_nums,
                         max_training_token,max_training_padding_gap,draft_accumulation_steps,logps_chunk_size=None):
    

    all_draft_input_states = outputs['all_draft_input_states']
    all_draft_input_ids = outputs['all_draft_input_ids']
    all_prompt_length = [prompt_mask[idx // repeated_generate_nums].sum().item() for idx in range(len(all_draft_input_states))]
    
    prompt_mask=prompt_mask.cpu()
    device=model.target_model.device
    
    sorted_pairs = sorted(
        zip(all_draft_input_ids, all_draft_input_states, all_prompt_length),
        key=lambda x: len(x[0]),
        reverse=False  
    )

    all_draft_input_ids_sorted, all_draft_input_states_sorted, all_prompt_length_sorted = zip(*sorted_pairs)

    all_draft_input_ids = list(all_draft_input_ids_sorted)
    all_draft_input_states = list(all_draft_input_states_sorted)
    all_prompt_length = list(all_prompt_length_sorted)
    
    total_loss1,total_loss2=0,0
    
    draft_input_states_list=[]
    draft_input_ids_list=[]
    prompt_length_list=[]
    
    cur_max_length=0
    hidden_size=all_draft_input_states[0].shape[-1]
    
    for idx , (draft_input_states,draft_input_ids,prompt_length) in enumerate(zip(all_draft_input_states,all_draft_input_ids,all_prompt_length)):
        
        if ((draft_input_ids.shape[-1]*(len(draft_input_states_list)+1)<=max_training_token*2 and
            (draft_input_ids.shape[-1]-cur_max_length)*len(draft_input_states_list)<=max_training_padding_gap) or
            len(draft_input_states_list)==0):
            
                draft_input_states_list.append(draft_input_states)
                draft_input_ids_list.append(draft_input_ids)
                prompt_length_list.append(prompt_length)
                
                cur_max_length=max(cur_max_length, draft_input_ids.shape[-1])
            
        else:
            
            cur_batch=len(draft_input_states_list)

            loss_mask=[[] for _ in range(cur_batch)]
            attention_mask=[[] for _ in range(cur_batch)]
            
            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
                attention_mask[idx_seq]=[1]*cur_len

            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    draft_input_states_list[idx_seq]=torch.concat(
                        [draft_input_states_list[idx_seq],
                        torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                        dim=-2)
                    
                    draft_input_ids_list[idx_seq]=torch.concat(
                        [draft_input_ids_list[idx_seq],
                        torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                        dim=-1)
                    
                    loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
                    attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
            
            draft_input_states=torch.stack(draft_input_states_list,dim=0)
            draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
            loss_mask=torch.tensor(loss_mask,device=device)
            attention_mask=torch.tensor(attention_mask,device=device)

            with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
                draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                                attention_mask=attention_mask,use_cache=False)
                
            if logps_chunk_size:
                loss1, loss2 = _chunked_draft_losses(
                    model, draft_outputs, draft_input_states, loss_mask, logps_chunk_size)
            else:
                loss1, loss2 = _full_draft_losses(model, draft_outputs, draft_input_states, loss_mask)

            loss=loss1+loss2
                
            total_loss1+=loss1.item()
            total_loss2+=loss2.item()
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                
                loss = loss.detach()
                del loss
                torch.cuda.empty_cache()
            else:

                loss=loss/len(all_draft_input_states)
                loss=loss/draft_accumulation_steps
                loss.backward()
                
            draft_input_states_list=[all_draft_input_states[idx]]
            draft_input_ids_list=[all_draft_input_ids[idx]]
            prompt_length_list=[all_prompt_length[idx]]
            cur_max_length=all_draft_input_ids[idx].shape[-1]
            
    cur_batch=len(draft_input_states_list)

    loss_mask=[[] for _ in range(cur_batch)]
    attention_mask=[[] for _ in range(cur_batch)]
    
    cur_max_length=0
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
        attention_mask[idx_seq]=[1]*cur_len
        
        cur_max_length=max(cur_max_length, cur_len)
        
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        padding_len=cur_max_length-cur_len
        
        if padding_len>0:
            draft_input_states_list[idx_seq]=torch.concat(
                [draft_input_states_list[idx_seq],
                torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                dim=-2)
            
            draft_input_ids_list[idx_seq]=torch.concat(
                [draft_input_ids_list[idx_seq],
                torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                dim=-1)
            
            loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
            attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
    
    draft_input_states=torch.stack(draft_input_states_list,dim=0)
    draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
    loss_mask=torch.tensor(loss_mask,device=device)
    attention_mask=torch.tensor(attention_mask,device=device)
    
    with torch.amp.autocast(str(model.target_model.device),
                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                        attention_mask=attention_mask,use_cache=False)
        
    if logps_chunk_size:
        loss1, loss2 = _chunked_draft_losses(
            model, draft_outputs, draft_input_states, loss_mask, logps_chunk_size)
    else:
        loss1, loss2 = _full_draft_losses(model, draft_outputs, draft_input_states, loss_mask)

    loss=loss1+loss2
        
    total_loss1+=loss1.item()
    total_loss2+=loss2.item()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        
        loss = loss.detach()
        del loss
        torch.cuda.empty_cache()
    else:

        loss=loss/len(all_draft_input_states)
        loss.backward()
            
        
    total_loss1/=len(all_draft_input_states)
    total_loss2/=len(all_draft_input_states)
    
    return total_loss1,total_loss2
