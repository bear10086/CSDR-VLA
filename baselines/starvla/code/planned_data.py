import json
from collections import OrderedDict
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from common import ROOT

class PlannedDataset(Dataset):
    def __init__(self,epoch,rank,start_micro=0):
        cache=ROOT/'cache'
        self.actions=np.load(cache/'actions.npy',mmap_mode='r')
        self.offsets=np.load(cache/'offsets.npy')
        self.prompt_ids=np.load(cache/'prompt_ids.npy',mmap_mode='r')
        self.prompts=json.loads((cache/'prompts.json').read_text())
        self.episodes=json.loads((cache/'episodes.json').read_text())
        with np.load(ROOT/f'plans/epoch_{epoch}.npz') as plan:
            self.group_counts=plan['loss_mask'].sum((1,2)).astype(np.int64)
            self.index=plan['global_index'][start_micro:,rank].reshape(-1)
            self.loss_mask=plan['loss_mask'][start_micro:,rank].reshape(-1)
            self.csdr_mask=plan['csdr_mask'][start_micro:,rank].reshape(-1)
        self.maps=OrderedDict()

    def __len__(self):
        return len(self.index)

    def __getitem__(self,index):
        idx=int(self.index[index])
        ep_idx=int(np.searchsorted(self.offsets,idx,side='right')-1)
        ep=int(self.episodes[ep_idx]['episode_index'])
        t=idx-int(self.offsets[ep_idx])
        image_map=self.maps.pop(ep,None)
        if image_map is None:
            image_map=np.load(ROOT/f'cache/chunk-{ep//1000:03d}/episode_{ep:06d}.images.npy',mmap_mode='r')
        self.maps[ep]=image_map
        while len(self.maps)>32:
            self.maps.popitem(last=False)
        # Native default: repeat the final action for out-of-episode offsets.
        chunk=np.minimum(idx+np.arange(16),self.offsets[ep_idx+1]-1)
        prompt=int(self.prompt_ids[idx])
        return {'image':[Image.fromarray(np.array(image_map[t],copy=True))],
                'lang':self.prompts[prompt],'action':np.array(self.actions[chunk],copy=True),
                'prompt_id':prompt,'loss_mask':int(self.loss_mask[index]),
                'csdr_mask':int(self.csdr_mask[index]),'global_index':idx,'action_valid_length':int(min(16,self.offsets[ep_idx+1]-idx))}

def collate(examples):
    return examples
