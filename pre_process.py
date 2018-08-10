import numpy as np
from skimage.transform import resize

class PreProcess:
    def crop(self, img):
        img_cropped = np.zeros((168, 148))
        img_cropped[:,:] = img[29:197,6:154]
        return img_cropped

    def to_grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2]

    # return [84, 84] frame shape
    def preprocess(self, img):
        return resize(self.downsample(self.crop(self.to_grayscale( img))), (84,84))

    def transform_reward(self, reward):
        return np.sign(reward)