import os
from matplotlib import pyplot as plt
import numpy as np
import random
import cv2

#IMAGE HISTOGRAM
def histogram(gray, plot=False, bins=26):
    """
    Compute the histogram of a grayscale image.

    :param gray: grayscale image
    :param plot: (bool)
    :return: (dict) histogram
    """
    hist, bins = np.histogram(gray, bins=bins, range=(0, 255))

    if plot:
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot grayscale image
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title("Grayscale Image")
        axes[0].axis("off")  # Hide axes

        # Plot histogram
        axes[1].plot(bins[:-1], hist, color='black')
        axes[1].set_title("Brightness Histogram")
        axes[1].set_xlabel("Normalized Brightness (0=Dark, 1=Bright)")
        axes[1].set_ylabel("Pixel Count")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    tot = sum(hist)
    histo_dict = {k: n / tot for n, k in zip(hist, bins[:-1])}
    return histo_dict


def get_histo_distance(hist1, hist2):
    """
    Compute the Chi-Square distance between two histograms represented as dictionaries.

    :param hist1: dict, first histogram {bin: frequency}
    :param hist2: dict, second histogram {bin: frequency}
    :return: float, chi-square distance
    """
    # get all unique bins
    all_bins = set(hist1.keys()).union(set(hist2.keys()))
    chi_sq = 0.0
    for bin in all_bins:
        h1 = hist1.get(bin, 0)  # Default to 0 if bin is missing
        h2 = hist2.get(bin, 0)  # Default to 0 if bin is missing
        if h1 + h2 > 0:  # Avoid division by zero
            chi_sq += ((h1 - h2) ** 2) / (h1 + h2)

    return chi_sq


def dataset_split_and_save(dataset, path1, path2, split_ratio):
    filenames1 = random.sample(list(dataset.keys()), int(len(dataset) * split_ratio))
    os.makedirs(path1,exist_ok=True)
    os.makedirs(path2,exist_ok=True)
    cnt_1, cnt_2 = 0, 0
    for filename, img in dataset.items():
        if filename in filenames1:
            path = os.path.join(path1, filename)
            cnt_1 += 1
        else:
            path = os.path.join(path2, filename)
            cnt_2 += 1
        cv2.imwrite(path, img)

    print(f"Saved {cnt_1} items in {path1}")
    print(f"Saved {cnt_2} items in {path2}")

