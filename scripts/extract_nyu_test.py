#!/usr/bin/env python
import os
import sys

import cv2
import h5py
import numpy as np
import scipy.io


# Function for writing one RGB + depth pair into out_dir/test/
def write_pair(i, depth, image, out_dir):
    folder = os.path.join(out_dir, "test")
    os.makedirs(folder, exist_ok=True)

    # depth comes in meters -> store as 16-bit millimeters (divide by 1000 to read back)
    depth_mm = (depth * 1000.0).astype(np.uint16)
    cv2.imwrite("%s/sync_depth_%05d.png" % (folder, i), depth_mm)

    # RGB -> BGR for cv2, and blank the few-pixel border BTS crops off
    image = image[:, :, ::-1]
    bordered = np.zeros((480, 640, 3), dtype=np.uint8)
    bordered[7:474, 7:632, :] = image[7:474, 7:632, :]
    cv2.imwrite("%s/rgb_%05d.jpg" % (folder, i), bordered)


def main():
    if len(sys.argv) < 4:
        print("usage: %s <labeled.mat> <splits.mat> <out_dir>" % sys.argv[0], file=sys.stderr)
        sys.exit(1)

    mat_path, splits_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    mat = h5py.File(mat_path, "r")              # big file: only h5py can open it
    splits = scipy.io.loadmat(splits_path)      # small file: scipy reads it

    # .ravel() is the fix: testNdxs is (654, 1), we want a flat list of ints (1-based)
    test_images = set(int(x) for x in splits["testNdxs"].ravel())
    print("%d test images to extract" % len(test_images))

    images = mat["images"]
    raw_depths = mat["rawDepths"]

    for i, image in enumerate(images):
        idx = i + 1                              # matlab indices are 1-based
        if idx not in test_images:
            continue
        write_pair(i, raw_depths[i, :, :].T, image.T, out_dir)
        print("wrote test image", idx)

    print("Finished")


if __name__ == "__main__":
    main()
