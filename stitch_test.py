import cv2
import numpy as np
import matplotlib.pyplot as plt
# from stitch_v1 import Matcher, Stitcher
import time
import threading

# Import 4 images to test stitching
left = cv2.imread('./videos/left.png')
top = cv2.imread('./videos/top.png')
right = cv2.imread('./videos/right.png')
bottom = cv2.imread('./videos/bottom.png')

img_list = [left, top, right, bottom]
background_img = np.zeros((1500, 2500, 3))

# Init a SURF feature extractor
surf = cv2.xfeatures2d.SURF_create()

# Extract features of the image
left_kp, left_des = surf.detectAndCompute(left, None)
left_with_kp = cv2.drawKeypoints(left, left_kp[:50], None, (255, 0, 0), 4)
top_kp, top_des = surf.detectAndCompute(top, None)
top_with_kp = cv2.drawKeypoints(top, top_kp[:50], None, (255, 0, 0), 4)

# Plot the images with keypoints drawn
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("Images with keypoints")
ax1.imshow(top_with_kp)
ax2.imshow(left_with_kp)
plt.show()

# Match the keypoints pairs
# PLANN parameters
FLANN_INDEX_KDTREE = 0
MIN_MATCH_COUNT = 10
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(top_des, left_des, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# Ratio test as per Lowe's paper
good = []
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]
        good.append(m)

draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matchesMask,
                    flags=0)

img = cv2.drawMatchesKnn(top, top_kp, left, left_kp, matches, None, **draw_params)
plt.imshow(img)
plt.show()

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([top_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([left_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = top.shape[:2]
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    # pts_top = [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
    # for i in pts_top:
    #     plt.imshow(top)
    #     plt.plot(i[0], i[1], 'r*')
    #     plt.show()
    dst = cv2.perspectiveTransform(pts, M)
    # print([np.int32(dst)])
    # x_offset, y_offset = np.int32(dst)[0][0][0], np.int32(dst)[0][0][1]
    # plt.figure()
    # for i in np.int32(dst):
    #     plt.imshow(left)
    #     plt.plot(i[0][0], i[0][1], 'c+')
    #     plt.show()
    # left_h, left_w = left.shape[:2]
    
    # background_img = np.zeros((1000, 2000, 3))
    # # Stitch the left image into the background
    # background_img[100:100+left_h, 100:100+left_w, :] = left/255.0
    # # Stich the top image on top of the left image
    # background_img[100+y_offset:100+y_offset+h, 100+x_offset:100+x_offset+w, :] = top/255.0
    # plt.figure()
    # plt.imshow(background_img)
    # plt.show()
    left = cv2.polylines(left, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask, # draw only inliers
                   flags=2)

img3 = cv2.drawMatches(top, top_kp, left, left_kp, good, None, **draw_params)

plt.imshow(img3, 'gray')
plt.show()