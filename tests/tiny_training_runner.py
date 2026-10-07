"""Synthetic entrypoint integration fixture; controlled rewards, no benchmark claims."""
import sys,os,runpy,json
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM,PreTrainedTokenizerFast
from tokenizers import Tokenizer,models,pre_tokenizers,decoders
from copy import deepcopy
from helper.fastgrpo_model import FastGRPOModel
root=Path(sys.argv.pop(1));root.mkdir(parents=True,exist_ok=True)
modeldir=root/'model'
if not (root/'sharegpt.json').exists():
    torch.manual_seed(44)
    config=Qwen2Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024,torch_dtype=torch.bfloat16)
    target=Qwen2ForCausalLM(config).bfloat16();target.save_pretrained(modeldir)
    dc=deepcopy(config);dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=target.dtype
    FastGRPOModel(dc,target).save_model(root/'draft.pth')
    vocab={'t'+str(i):i for i in range(97)}
    tok=Tokenizer(models.WordLevel(vocab,unk_token='t0'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.decoder=decoders.WordPiece(prefix='')
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,unk_token='t0',pad_token='t0',eos_token='t96')
    tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    tokenizer.save_pretrained(modeldir)
    (root/'train.json').write_text(json.dumps([{'question':'t3 t4','answer':'2'}]*4))
    (root/'sharegpt.json').write_text(json.dumps([{'conversations':[{'from':'human','value':'t3 t4'},{'from':'gpt','value':'t5 t6 t7 t8'}]}]*4))
if len(sys.argv)==1:sys.exit(0)
method=sys.argv[1];runname=sys.argv[2];remaining=sys.argv[3:];out=root/runname
for sub in ('logs','target','draft','statistics','resume'):(out/sub).mkdir(parents=True,exist_ok=True)
# Controlled nonconstant rewards ensure the integration exercises GRPO optimizer
# steps even though the randomly initialized tiny model has no math ability.
import helper.rewards as rewards
import itertools
reward_counter=itertools.count()
rewards.accuracy_reward_func=lambda completions,solution,**kw:[float(next(reward_counter)%2) for _ in completions]
rewards.format_reward_func=lambda completions,**kw:[0.]*len(completions)
sys.argv=['grpo_speculative.py','--method',method,'--model_dir',str(modeldir),'--adapter_path',str(root/'draft.pth'),
 '--dataset_path',str(root/'train.json'),'--train_data_fraction','1','--batch_size','2','--accumulation_steps','1','--repeated_generate_nums','2',
 '--num_epochs','1','--max_length','112','--max_prompt_length','96','--max_training_token','512','--max_training_padding_gap','512',
 '--verification_capacity','28','--max_verification_num','7','--max_draft_k','2','--max_draft_token_length','3','--min_draft_token_length','3',
 '--num_workers','0','--persistent_workers','false','--dtype','bf16','--attn_implementation','sdpa',
 '--log_file',str(out/'logs/metrics.jsonl'),'--timing_file',str(out/'logs/timing.csv'),'--summary_file',str(out/'summary.json'),
 '--saved_model_dir',str(out/'target'),'--saved_draft_model_dir',str(out/'draft'),'--saved_statistics_dir',str(out/'statistics'),'--checkpoint_dir',str(out/'resume'),
 '--opd_train_projector','1',*remaining]
events=[]
import helper.specualtive_generate as generation
import helper.fastgrpo_training as training
old_generate=generation.speculative_generate
old_train=training.training_draft_model
old_init=torch.optim.AdamW.__init__
old_step=torch.optim.AdamW.step
optimizer_labels=iter(('target_step','draft_step'))
optimizers=[]
restore_pending=None
if '--resume_checkpoint' in remaining:
    restore_pending=torch.load(remaining[remaining.index('--resume_checkpoint')+1],map_location='cpu',weights_only=False)
def init(self,*a,**k):
    old_init(self,*a,**k);self.test_label=next(optimizer_labels);optimizers.append(self)
def step(self,*a,**k):
    events.append(self.test_label);return old_step(self,*a,**k)
def generate(*a,**k):
    global restore_pending
    if restore_pending is not None:
        from test_training_end_to_end import assert_nested_equal
        from helper.checkpointing import capture_rng_state
        from peft import get_peft_model_state_dict
        import numpy as np
        model=a[0] if a else k['model']
        def cpu(value):
            if torch.is_tensor(value):return value.cpu()
            if isinstance(value,dict):return {key:cpu(v) for key,v in value.items()}
            if isinstance(value,list):return [cpu(v) for v in value]
            return value
        assert_nested_equal(cpu(model.draft_model.state_dict()),restore_pending['draft_model'])
        assert_nested_equal(cpu(get_peft_model_state_dict(model.target_model)),restore_pending['target_lora'])
        for opt,key in zip(optimizers,('optimizer_target','optimizer_draft')):
            assert_nested_equal(cpu(opt.state_dict()),restore_pending[key])
        actual=capture_rng_state();expected=restore_pending['rank_states'][0]['rng']
        for key in ('torch','cuda','python'):assert_nested_equal(actual[key],expected[key])
        for x,y in zip(actual['numpy'],expected['numpy']):assert np.array_equal(x,y)
        (out/'test_restore.json').write_text(json.dumps({'bitwise_restore':True}))
        restore_pending=None
    events.append('rollout');return old_generate(*a,**k)
def train(*a,**k):
    events.append('draft_backward');return old_train(*a,**k)
generation.speculative_generate=generate;training.training_draft_model=train
torch.optim.AdamW.__init__=init;torch.optim.AdamW.step=step
runpy.run_path(str(REPO/'grpo_speculative.py'),run_name='__main__')
(out/'test_events.json').write_text(json.dumps(events))
