from os import getcwd
import cv2
import numpy as np
import os
import glob

imgs_source_path = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/BBBC010_v2_images/"
foreground_source_path = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/BBBC010_v1_foreground_eachworm/"

target_path_images = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/images_filtered/"
target_path = "/Users/alain/Documents/desarrollo/Fiber-Unet/datasets/wormbodies/foreground_overlapping/"
# images_dir = os.path.join(target_path, 'imgs')
# masks_dir = os.path.join(target_path, 'masks')


if __name__ == '__main__':

    # image = np.zeros((10,)) 
    # mask = np.zeros((10,))


    # file_path = os.path.realpath(__file__)

    # print(getcwd())
    # print(file_path)


    for file in sorted(os.listdir(imgs_source_path)):
        if not file.startswith('.'):

            img = cv2.imread(os.path.join(imgs_source_path, file),0)

            if img.sum() > 99999:

                img = img[110:460, 170:520]

                img_id = file.split('_')[6]

                # print(img_id)

                instances = glob.glob(os.path.join(foreground_source_path, img_id + "*" + "_ground_truth.png"))

                mask, old_mask = None, None

                for instance in instances:
                    mask = cv2.imread(instance, 0)
                    mask = mask[110:460, 170:520]
                    mask[mask>0] = 1
                    if old_mask is not None:
                        mask += old_mask
                        mask[mask>1] = 2
                    old_mask = mask

                cv2.imwrite(os.path.join(target_path_images, img_id + ".png"), img)
                cv2.imwrite(os.path.join(target_path, img_id + "_ground_truth.png"), mask)