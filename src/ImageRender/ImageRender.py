from PIL import Image
import numpy as np
import logging

COLOR_MAP = dict(
    IGNORE=(0, 0, 0),
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)



class ImageRender(object):
    def __init__(self):
        self.logger = logging.getLogger('ImageRender')
        self.logger.addHandler(logging.StreamHandler())
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.logger.handlers[0].setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.info('ImageRender is initialized') 
    
    def render(self, mask_path, vis_path):
        self.logger.info('Rendering mask to color image')
        new_mask = np.array(Image.open(mask_path)).astype(np.uint8)
        color_map = np.array(list(COLOR_MAP.values())).astype(np.uint8)
        color_img = color_map[new_mask]
        color_img = Image.fromarray(np.uint8(color_img))
        color_img.save(vis_path)
        self.logger.info('Rendering is done')
        
if __name__ == '__main__':
    imageRender = ImageRender()
    
    example_urban_directory = './example/urban'
    mask_path = example_urban_directory + '/mask.png'
    vis_path = example_urban_directory + '/segmentation.png'
    imageRender.render(mask_path, vis_path)
    
    example_rural_directory = 'example/rural'
    mask_path = example_rural_directory + '/mask.png'
    vis_path = example_rural_directory + '/segmentation.png'
    imageRender.render(mask_path, vis_path) 
        