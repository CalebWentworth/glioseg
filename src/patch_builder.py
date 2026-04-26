from pathlib import Path
import torchio as tio


class PatchBuilder:
    
    def __init__(self, dataset, patch_size=64, overlap=0, stride=None, 
             num_workers=2, batch_size=1,
             sampler=tio.LabelSampler(patch_size = 64,label_name='seg',
                                     label_probabilities={0:0, 1:0.25,2:0.25,3:0.25,4:0.25})):
       
        self.dataset = dataset
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = stride
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.sampler = sampler
    
    def build_patches(self):
       
        sampler = self.sampler

        transform = tio.Compose([
            tio.ZNormalization(),
            tio.RescaleIntensity((0,1))
        ])
        #transform = tio.ZNormalization()
        dataset = tio.SubjectsDataset(self.dataset,transform=transform)

        queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=300,        
        samples_per_volume=12,  
        sampler=sampler,
        num_workers=3,
        shuffle_subjects=True,
        shuffle_patches=True
        )
        
        return queue
    