import numpy as np 
import torch
from PIL import Image
import pandas as pd


class CovidLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, transforms):
        self.args= args

        if split=='train':
            self.img_paths= np.load(self.args.train_npy)
        elif split=='val':
            self.img_paths= np.load(self.args.val_npy)
        else:
            self.img_paths = np.load(self.args.test_npy)
         
        self.transforms= transforms

    def __getitem__(self, idx):
        img= Image.open('{}/{}'.format(self.args.root_dir, self.img_paths[idx][0]))
        label= self.img_paths[idx][1]
        label= torch.from_numpy(label, dtype=torch.float32)

        if self.transforms:
            img= self.transforms(covid_img.convert('RGB'))

        return img, label


    def __len__(self):
        return len(img_paths)