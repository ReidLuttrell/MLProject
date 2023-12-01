import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import os

print(os.getcwd())
write_file = open("datasets/ScanDataPreprocessed.txt", "a")

directories = [
    "datasets/AbdomenCT/",  # class id 1
    "datasets/BreastMRI/",  # class id 2
    "datasets/ChestCT/",  # class id 3
    "datasets/CXR/",  # class id 4
    "datasets/Hand/",  # class id 5
    "datasets/HeadCT/",  # class id 6
]

class_id = 1

for directory in directories:
    encoded_dir = os.fsencode(directory)
    for file in os.listdir(encoded_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".jpeg"):
            print(filename)
            img = cv.imread(directory + filename, cv.IMREAD_GRAYSCALE)
            for x in range(0, img.shape[0]):
                for y in range(0, img.shape[1]):
                    write_file.write(
                        str(float(img[x, y]) / 255) + " "
                    )  # this is gonna put a trailing space at the end of the last feature
            write_file.write(
                class_id
            )  # put class_id at the end, this will deal with trailing space
            write_file.write(",")  # comma separated
            # plt.imshow(img)
            # plt.show()
        else:
            continue
    class_id += 1  # new directory, so next class_id

write_file.close()
