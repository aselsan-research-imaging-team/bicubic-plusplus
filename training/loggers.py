import cv2
import os
import numpy as np
from tqdm import tqdm


class ImageLogger:
    def __init__(self, clip=True, ext='png'):
        self.p = './out_images/'
        os.makedirs(self.p) if not os.path.exists(self.p) else None

        self.clip = clip
        self.ext = ext

    def __call__(self, out):
        idx = 0
        with tqdm(out, desc='Saving validation outputs') as all_sr_batches:
            for sr_batch in all_sr_batches:
                sr_batch = (sr_batch.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255)
                if self.clip:
                    sr_batch = np.clip(sr_batch, 0, 255)
                sr_batch = sr_batch.astype(np.uint8)

                for img_sr in sr_batch:
                    path_sr = os.path.join(self.p, f"{idx}_SR.{self.ext}")
                    cv2.imwrite(path_sr, img_sr)
                    idx += 1
