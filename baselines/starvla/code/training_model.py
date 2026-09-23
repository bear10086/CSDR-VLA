"""Training-only CSDR adapter. The native prediction path is left untouched."""
import numpy as np
import torch
from torch import nn

def flow_loss_per_sample(head, vl_embs, actions, attention_mask=None):
    """Native GR00T flow loss, retaining per-sample errors before reduction.

    Sampling order and arithmetic match FlowmatchingActionHead.forward.
    Clean-action error is a detached one-step estimate, not a rollout metric.
    """
    noise=torch.randn(actions.shape,device=actions.device,dtype=actions.dtype)
    t=head.sample_time(actions.shape[0],device=actions.device,dtype=actions.dtype)[:,None,None]
    noisy=(1-t)*noise+t*actions
    velocity=actions-noise
    discrete=(t[:,0,0]*head.num_timestep_buckets).long()
    features=head.action_encoder(noisy,discrete)
    if head.config.add_pos_embed:
        features=features+head.position_embedding(torch.arange(features.shape[1],device=features.device)).unsqueeze(0)
    future=head.future_tokens.weight.unsqueeze(0).expand(len(actions),-1,-1)
    encoded=head.model(hidden_states=torch.cat((future,features),dim=1),
                       encoder_hidden_states=vl_embs,encoder_attention_mask=attention_mask,
                       timestep=discrete,return_all_hidden_states=False)
    pred=head.action_decoder(encoded)[:,-actions.shape[1]:]
    # Preserve BF16 native elementwise arithmetic; accumulate the resulting values in FP32.
    squared=(pred-velocity)**2
    task=squared.float().mean((1,2))
    error=((1-t.float())*(pred.float()-velocity.float())).abs().mean((1,2)).detach()
    return task,error

class TrainingModel(nn.Module):
    def __init__(self,native):
        super().__init__()
        self.native=native

    def forward(self,examples):
        model=self.native
        inputs=model.qwen_vl_interface.build_qwenvl_inputs(
            images=[x['image'] for x in examples],instructions=[x['lang'] for x in examples])
        # Conditional generation calls this same backbone before its unused LM head.
        hidden=model.qwen_vl_interface.model.model(**inputs,return_dict=True,
                   output_hidden_states=False,use_cache=False).last_hidden_state
        assert inputs['attention_mask'][:,-1].bool().all(), 'Expected left-padded prompts'
        representation=hidden[:,-1:,:]  # fused last context token; no action noise
        actions=torch.as_tensor(np.stack([x['action'] for x in examples]),device=hidden.device,dtype=hidden.dtype)
        repeats=int(model.config.framework.action_model.repeated_diffusion_steps)
        mask=inputs['attention_mask'].bool().repeat(repeats,1)
        loss,error=flow_loss_per_sample(model.action_model,hidden.repeat(repeats,1,1),
                                        actions.repeat(repeats,1,1),mask)
        return loss.reshape(repeats,len(examples)).mean(0),representation,actions,error.reshape(repeats,len(examples)).mean(0)

class FP32MasterAdamW(torch.optim.Optimizer):
    """BF16 model storage, FP32 master weights and Adam moments on each ZeRO shard."""
    def __init__(self,params,lr=1e-5,betas=(0.9,0.95),eps=1e-8,weight_decay=1e-8,**kwargs):
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        masters=[]
        for group in self.param_groups:
            masters.append({**{k:v for k,v in group.items() if k!='params'},
                            'params':[nn.Parameter(p.detach().float().clone()) for p in group['params']]})
        self.inner=torch.optim.AdamW(masters,fused=all(p.is_cuda for g in masters for p in g['params']))

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:
            raise ValueError('Closures are not used by this training loop')
        for source,master in zip(self.param_groups,self.inner.param_groups):
            for k,v in source.items():
                if k!='params':
                    master[k]=v
            for p,m in zip(source['params'],master['params']):
                m.grad=None if p.grad is None else p.grad.detach().float()
        self.inner.step()
        for source,master in zip(self.param_groups,self.inner.param_groups):
            for p,m in zip(source['params'],master['params']):
                if m.grad is not None:
                    p.copy_(m)
                m.grad=None

    def state_dict(self):
        return {'adam':self.inner.state_dict(),
                'master_weights':[[p.detach().cpu() for p in g['params']] for g in self.inner.param_groups]}

    def load_state_dict(self,state):
        self.inner.load_state_dict(state['adam'])
        for group,values in zip(self.inner.param_groups,state['master_weights']):
            for p,value in zip(group['params'],values):
                p.data.copy_(value)
