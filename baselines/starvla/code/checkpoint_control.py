"""Coordinated save/pause requests at complete optimizer-step boundaries."""
import json
import time
import torch
import torch.distributed as dist
from common import ROOT


def synchronize_control(grad_norm,rank,device,enabled=True):
    if not enabled:
        flag=torch.tensor(int(torch.isfinite(grad_norm)),device=device)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        return bool(flag.item()),False,False
    save=pause=True  # MIN takes rank zero's request, together with all finite flags.
    if rank==0:
        path=ROOT/'checkpoint_request.json'
        save=path.exists();pause=False
        if save:
            try:pause=bool(json.loads(path.read_text()).get('pause',False))
            except (ValueError,OSError):save=False
    flags=torch.tensor([int(torch.isfinite(grad_norm)),int(save),int(pause)],device=device)
    dist.all_reduce(flags,op=dist.ReduceOp.MIN)
    return tuple(bool(x) for x in flags.cpu().tolist())


def requested_save(save_function,output,native,optimizer,scheduler,step,epoch,next_micro,args,pause):
    recovery=output/'recovery'
    save_function(recovery,native,optimizer,scheduler,step,epoch,next_micro,args)
    if dist.get_rank()==0:
        latest=json.loads((recovery/'latest.json').read_text())
        tmp=output/'latest.json.tmp';tmp.write_text(json.dumps(latest));tmp.replace(output/'latest.json')
        record={'step':step,'checkpoint_directory':latest['directory'],'paused':pause,
                'time':time.strftime('%Y-%m-%d %H:%M:%S')}
        (output/f'requested_save_{step}.json').write_text(json.dumps(record,indent=2))
        if pause:(output/'training_paused.json').write_text(json.dumps(record,indent=2))
        request=ROOT/'checkpoint_request.json'
        if request.exists():request.replace(ROOT/f'checkpoint_request_handled_{step}.json')
        print('REQUESTED_CHECKPOINT_COMPLETE',json.dumps(record),flush=True)
    dist.barrier()
