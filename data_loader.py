import numpy as np 
import torch
from PIL import Image
import pandas as pd


class CovidLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, transforms):
        self.args= args

        if split=='train':
            self.img_paths= np.load(self.args.train_npy, allow_pickle=True)
        
        elif split=='val':
            self.img_paths= np.load(self.args.val_npy, allow_pickle=True)
        
        else:
            self.img_paths = np.load(self.args.test_npy, allow_pickle=True)
         
        self.transforms= transforms

    def __getitem__(self, idx):
        if self.img_paths[idx][1] == 0:
            root_dir= '{}/CT_NonCOVID'.format(self.args.root_dir)
        elif self.img_paths[idx][1] == 1:
            root_dir= '{}/CT_COVID'.format(self.args.root_dir)
        else:
            raise exception('label not implemented! {}'.format(self.img_paths[idx][1]))

        img= Image.open('{}/{}'.format(root_dir, self.img_paths[idx][0]))
        label= self.img_paths[idx][1]
        label= torch.tensor(label, dtype=torch.float32)

        if self.transforms:
            img= self.transforms(img.convert('RGB'))

        return img, label


    def __len__(self):
        return len(self.img_paths)
    
    
    
class MyTransforms():
    def __init__(self, *args):
        # this is where you initialise certain parameters / other functions you want to call.
        pass
    
    def __call__(self, input):
        # this is where you define what transformation will happen
        pass
