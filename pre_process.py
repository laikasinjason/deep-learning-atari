
import numpy as np

class PreProcess:
    def crop(self, img):
        img_cropped = np.zeros((166, 148))
        img_cropped[:,:] = img[30:196,6:154]
        return img_cropped

    def to_grayscale(self, img):
        return np.mean(img, axis=2).astype(np.uint8)

    def downsample(self, img):
        return img[::2, ::2]

    def preprocess(self, img):
        return to_grayscale(downsample(img))

    def transform_reward(self, reward):
        return np.sign(reward)