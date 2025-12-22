"""
Strategy to find local features based on a threshold and a mininum of matches
It uses ORB keypoints and descriptors for each image, followed by a k-NN matching and filtering
RANSAC is also applied for geometric consitency
"""
from pathlib import Path
from itertools import combinations
import cv2
import numpy as np

def find_local_features_candidates(images, threshold=0.20, min_matches=20, debug=False):
    """
    threshold (float): minimum similarity score 
    min_matches (int): minimum number of good ORB matches to attempt homography.
    """ 
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    keypoints = {}
    descriptors = {}

    # Extracting features
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            keypoints[p.name] = kp
            descriptors[p.name] = des

    results = []

    # Comparison of unique image pairs
    for a, b in combinations(descriptors.keys(), 2):
        des1, des2 = descriptors[a], descriptors[b]
        kp1, kp2 = keypoints[a], keypoints[b]

        # k-NN matching and filtering
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.65 * n.distance]

        if len(good) < min_matches:
            continue

        # RANSAC to complete geometric verification 
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if mask is None:
            continue

        inliers = int(mask.sum())
        score = inliers / min(len(des1), len(des2))

        if debug:
            print(f"{a} <-> {b} | good: {len(good)} | inliers: {inliers} | score: {score:.3f}")

        if score >= threshold:
            results.append((a, b, float(score)))

    return results
