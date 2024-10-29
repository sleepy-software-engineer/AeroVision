import warnings
warnings.filterwarnings('ignore')
import os
import glob
import logging
import numpy as np
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
from albumentations import Compose, OneOf, HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6
)

def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int64) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls) * label, new_cls)
    return new_cls

class Data(Dataset):
    def __init__(self, image_dir, mask_dir=None, transforms=None):
        self.rgb_filepath_list = []
        self.cls_filepath_list = []
        self.transforms = transforms

        if isinstance(image_dir, list) and isinstance(mask_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        elif isinstance(image_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif')) + glob.glob(os.path.join(image_dir, '*.png'))
        logger.info('%s -- Dataset images: %d' % (os.path.dirname(image_dir), len(rgb_filepath_list)))
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        
        cls_filepath_list = []
        if mask_dir is not None:
            for fname in rgb_filename_list:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
        
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list

    def __getitem__(self, idx):
        image = imread(self.rgb_filepath_list[idx])
        if len(self.cls_filepath_list) > 0:
            mask = imread(self.cls_filepath_list[idx]).astype(np.long) - 1
            if self.transforms is not None:
                transformed = self.transforms(image=image, mask=mask)
                image, mask = transformed["image"], transformed["mask"]
            return image, {"cls": mask, "fname": os.path.basename(self.rgb_filepath_list[idx])}
        else:
            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            return image, {"fname": os.path.basename(self.rgb_filepath_list[idx])}

    def __len__(self):
        return len(self.rgb_filepath_list)

class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Loader(DataLoader):
    def __init__(self, config):
        dataset = Data(config.image_dir, config.mask_dir, config.transforms)
        if config.cv_index != -1:
            train_sampler = RandomSampler(dataset) if config.training else SequentialSampler(dataset)
            sampler = train_sampler if config.training else SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset) if config.training else SequentialSampler(dataset)
        
        super(Loader, self).__init__(dataset,
                                           batch_size=config.batch_size,
                                           sampler=sampler,
                                           num_workers=config.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

def create_default_config():
    return Config(
        image_dir=None,
        mask_dir=None,
        batch_size=4,
        num_workers=4,
        cv_index=-1,  # -1 indicates no cross-validation
        training=True,
        transforms=Compose([
            Resize(height=224, width=224),  # Resize for Swin Transformer
            OneOf([
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
            ], p=0.75),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True),
            ToTensorV2()
        ])
    )
        
if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    # # Paths to the images and masks for the dataset (update these as needed)
    config = create_default_config()
    config.image_dir = "/home/lucian/University/MSc-Courses/ComputerVision/data/Train/Urban/images_png"
    config.mask_dir = "/home/lucian/University/MSc-Courses/ComputerVision/data/Train/Urban/masks_png"
    dataloader = Loader(config)
    for batch in dataloader:
        images, targets = batch
        print(images.shape, targets)
    # image_dir = "/home/lucian/University/MSc-Courses/ComputerVision/data/Train/Rural/images_png"
    # mask_dir = "/home/lucian/University/MSc-Courses/ComputerVision/data/Train/Rural/masks_png"

    # # Instantiate the dataset class
    # dataset = LoveDA(image_dir=image_dir, mask_dir=mask_dir, transforms=None)  # Temporarily set transforms to None
    # # Function to display the image and mask side-by-side for visual inspection
    # def display_sample(idx):
    #     image, data_dict = dataset[idx]
    #     mask = data_dict['cls']  # Extract the mask

    #     # Plot image and mask
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     axs[0].imshow(image)
    #     axs[0].set_title("Image")
    #     axs[1].imshow(mask, cmap='tab20')  # Using a discrete color map for mask classes
    #     axs[1].set_title("Mask")
    #     # save the plot to a file 
    #     plt.savefig(f"sample_{idx}.png")

    # # Test with a specific sample (e.g., the first one)
    # display_sample(20)