import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_palette(image_path, n_colors=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    
    return colors

# Usage
image_path = sys.argv[1] 
palette = extract_palette(image_path, n_colors=5)
print(palette)

