import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

import os

print(os.getcwd())
write_file = open("../datasets/ScanDataPreprocessed.csv", "a")

directories = [
    "../datasets/AbdomenCT/",  # class id 0
    "../datasets/BreastMRI/",  # class id 1
    "../datasets/ChestCT/",  # class id 2
    "../datasets/CXR/",  # class id 3
    "../datasets/Hand/",  # class id 4
    "../datasets/HeadCT/",  # class id 5
]

class_id = 0

for directory in directories:
    count = 0

    print(directory)
    encoded_dir = os.fsencode(directory)
    for file in os.listdir(encoded_dir):
        count += 1
        if count >= 1000:
            break

        filename = os.fsdecode(file)
        if filename.endswith(".jpeg"):
            img = cv.imread(directory + filename, cv.IMREAD_GRAYSCALE)

            write_file.write(str(class_id))  # put class_id at beginning

            for x in range(0, img.shape[0]):
                for y in range(0, img.shape[1]):
                    write_file.write(" " + str(float(img[x, y]) / 255))

            write_file.write(",")  # comma separated
            # plt.imshow(img)
            # plt.show()
        else:
            continue
    class_id += 1  # new directory, so next class_id

write_file.close()
