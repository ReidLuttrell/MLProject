import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import os

directory = os.fsencode("datasets/AbdomenCT")

write_file = open("datasets/PreprocessedAbdomenCT/data.txt", "a")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpeg"):
        print(filename)
        img = cv.imread("datasets/AbdomenCT/" + filename, cv.IMREAD_GRAYSCALE)
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                write_file.write(
                    str(float(img[x, y]) / 255) + " "
                )  # this is gonna put a trailing space at the end of the last feature
                # print(img[x, y])
        write_file.write(",")  # comma separated
        # plt.imshow(img)
        # plt.show()
    else:
        continue

write_file.close()
