import cv2
import gc
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib import tzip


def resize_image(img, org_width, org_height, new_width, new_height):
    img = 255 - img  # invert
    img = (img * (255.0 / img.max())).astype(np.uint8)  # normalize
    img = img.reshape(org_height, org_width)  # reshape
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)  # resize
    return img


def resize_and_save_image(train_dir, img, org_width, org_height, new_width, new_height, image_id):
    image_resized = resize_image(img, org_width, org_height, new_width, new_height)
    cv2.imwrite(train_dir + str(image_id) + '.png', image_resized)


def generate_images(data_dir, train_dir, org_width, org_height, new_width, new_height):
    print("Extracting images from parquet files, this will take a while...")

    for i in range(0, 4):
        print("\tReading file {}...".format(i))
        df = pd.read_parquet(os.path.join(data_dir, 'train_image_data_' + str(i) + '.parquet'))
        image_ids = df['image_id'].values
        df = df.drop(['image_id'], axis=1)

        print("\tCreating images...")
        for image_id, index in tzip(image_ids, range(df.shape[0])):
            resize_and_save_image(train_dir, df.loc[df.index[index]].values, org_width,
                                  org_height, new_width, new_height, image_id)

        print("\tCleaning up...")
        del df
        gc.collect()

    print("Done!")
