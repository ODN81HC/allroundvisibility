import os
import time
from threading import Thread
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Matcher():
    """
    This is the class to match the features of two images by the SUFT algorithm
    to extract the features descriptor. The match algorithm is based on the Flann
    algorithm, which is useful in a large set of data pairs.
    References:
    -----------
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html
    """

    def __init__(self, FLANN_INDEX_KDTREE=0, MIN_MATCH_COUNT=10):
        self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
        self.surf = cv2.xfeatures2d.SURF_create()
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def get_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.surf.detectAndCompute(gray, None)
        return (kp, des)
    
    def match(self, img1, img2):
        """
        This is the main function to match the features of two images using FLANN algorithm
        to match the features of SURF.
        """
        img1_kp, img1_des = self.get_features(img1)
        img2_kp, img2_des = self.get_features(img2)

        matches = self.flann.knnMatch(img2_des, img1_des, k=2)
        # Get good matches by the ratio test as per Lowe's paper
        good = []
        for (m, n) in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        if len(good) >= self.MIN_MATCH_COUNT:
            src_pts = np.float32([img2_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([img1_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return M
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            return None


class Frame():
    """
    This class is initialised after the commission part, when it has the recommended coordinates
    for the subvideos already.
    """

class Stitcher():
    """
    This is the class that is used to stitch the images based on the features get at class
    Matcher(), get the coordinates of the contour and simply make an overlay into it to
    another bigger background image.
    References:
    -----------
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
    
    Parameters:
    -----------
    `unmatched_positions`:  [list((x, y))] Positions of frames if they're unmatched.

    Attributes:
    -----------
    `background_img`:       [np.zeros((h, w, c))] Black image which acts as a background to stitch the images.
    `matcher_obj`:          [Matcher()] Matcher object of two images.
    `images`:               [list(images)] List of images.
    """
    def __init__(self, unmatched_positions):
        # self.background_img = None
        self.images = []
        self.matcher_obj = Matcher()
        self.unmatched_positions = unmatched_positions
        self.frames = {}

    def stitch_commision(self, images):
        """
        This method is only used to setup the camera views and record the recommended coordinates.
        """
        self.images = images
        
        for i in range(len(self.images)-1):
            img1 = self.images[i]
            img2 = self.images[i+1]

            M = self.matcher_obj.match(img1, img2)
            if M is None:
                break
            else:
                img2_h, img2_w = img2.shape[:2]

                pts = np.float32([[0, 0], [0, img2_h-1], [img2_w-1, img2_h-1], [img2_w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                x_offset, y_offset = np.int32(dst)[0][0][0], np.int32(dst)[0][0][1]

                if i == 0:
                    if i not in self.frames.keys():
                        self.frames[i] = [(300, 300)]
                        self.frames[i+1] = [(300+x_offset, 300+y_offset)]
                    else:
                        self.frames[i].append((300, 300))
                        self.frames[i+1].append((300+x_offset, 300+y_offset))
                else:
                    ref_posx, ref_posy = self.frames[i][-1]
                    if i+1 not in self.frames.keys():
                        self.frames[i+1] = [(ref_posx+x_offset, ref_posy+y_offset)]
                    else:
                        self.frames[i+1].append((ref_posx+x_offset, ref_posy+y_offset))

    def recommended_coords(self):
        img_pos = {}
        for key, values in self.frames.items():
            list_x, list_y = [], []
            for (x, y) in values:
                list_x.append(x)
                list_y.append(y)
            img_pos[key] = (np.int32(np.median(list_x)), np.int32(np.median(list_y)))
        return img_pos


    def stitch(self, images):
        """
        This is the main algorithm to stitch multiple images, as defined: the left subimage
        will be stitched first and put as the reference, then the other subimages will be
        stitched later to a bigger canvas.

        Parameters:
        -----------
        `images`:           [list(images)] This is the list of images that are put into the stitch
                            method, if it has four camera views, it should be in order
                            [left, top, right, bottom].
        `background_shape`: [(h, w)] Define the background image height and width shape (Removed).

        Output:
        -------
        `imgs_pos`:         [dict(int : (x, y))] the coordinates position of the frames.
        """
        # The list of images will be as: [left, top, right, bottom]
        self.images = images
        # Get images pairs
        imgs_pos = {}
        start_time = time.time()

        # Loop pairs of images
        for i in range(0, len(self.images)-1):
            img1 = self.images[i]
            img2 = self.images[i+1]
            # Perspective transform img2 and put into img1
            M = self.matcher_obj.match(img1, img2)
            if M is None:
                print("No correlation between frame {} and {}".format(i, i+1))
                # If there is no corrolated between subvideos, assign them as seperate
                if i == 0:
                    # Hard-coded left frame
                    imgs_pos[i] = self.unmatched_positions[i]
                    # Hard-coded top frame
                    imgs_pos[i+1] = self.unmatched_positions[i+1]
                else:
                    imgs_pos[i+1] = self.unmatched_positions[i+1]
            else:
                # img1_h, img1_w = img1.shape[:2]
                img2_h, img2_w = img2.shape[:2]

                pts = np.float32([[0, 0], [0, img2_h-1], [img2_w-1, img2_h-1], [img2_w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                x_offset, y_offset = np.int32(dst)[0][0][0], np.int32(dst)[0][0][1]
                # if x_offset < 0 or y_offset < 0:
                #     plt.figure()
                #     plt.imshow(img1)
                #     for u in range(4):
                #         plt.plot(np.int32(dst)[u][0][0], np.int32(dst)[u][0][1], 'r+')
                #     plt.show()

                if i == 0:
                    # If this is the first pair, place two images at the same loop
                    # self.background_img[300:300+img1_w, 300:300+img1_h, :] = img1/255.0
                    imgs_pos[i] = (300, 300)
                    # Save the next image position
                    # self.background_img[300+x_offset:300+x_offset+img2_w, 300+y_offset:300+y_offset+img2_h, :] = img2/255.0
                    imgs_pos[i+1] = (300+x_offset, 300+y_offset)
                else:
                    # If not, stitch only img2
                    ref_posx, ref_posy = imgs_pos[i]
                    # self.background_img[ref_posx+x_offset:ref_posx+x_offset+img2_w, ref_posy+y_offset:ref_posy+y_offset+img2_h, :] = img2/255.0
                    imgs_pos[i+1] = (ref_posx+x_offset, ref_posy+y_offset)
        print("[INFO] Stitching took {:.2f} seconds".format(time.time()-start_time))

        # return self.background_img, imgs_pos
        return imgs_pos


class ThreadWithReturnedVal(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                            args=(), kwargs=None, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def main():
    # Define params
    folderDir = './videos/LF_subvids'
    output_vids = ['left.mp4', 'top.mp4', 'right.mp4', 'bottom.mp4']

    last_saved_coords = {0: None, 1: None, 2: None, 3: None}
    unmatched_pos = [(0, 250), (550, 0), (1680, 250), (550, 770)]
    # Init a stitcher and control it by a main option
    stitcher = Stitcher(unmatched_positions=unmatched_pos)
    # This is the positions in the case that the frame pair doesn't match
    start_time = time.time()
    cap = []
    for vid in output_vids:
        cap.append(cv2.VideoCapture(os.path.join(folderDir, vid)))

    ret = [None] * len(output_vids)
    frame = [None] * len(output_vids)

    while True:
        coords = None
        background_img = np.zeros((1500, 2500, 3))
        for i, c in enumerate(cap):
            if c is not None:
                ret[i], frame[i] = c.read()
        # After this step, the frame with respected to the output_vids are ready
        if time.time() - start_time >= 10:
            # new_thread = ThreadWithReturnedVal(target=stitcher.stitch, args=(frame,))
            # new_thread.start()
            # coords = new_thread.join()
            coords = stitcher.stitch(frame)
            # Reset the timer
            start_time = time.time()

            # Compare to get the good coords for frames
            for i in last_saved_coords.keys():
                # try:
                #     coords[i]
                # except KeyError:
                #     print("[ERROR] No matching between frame {} and {}".format(i+1, i+2))
                # else:
                #     if last_saved_coords[i] is None:
                #         last_saved_coords[i] = coords[i]
                #     else:
                #         # Compare the coordinates to see if it's good enough
                #         prev_x, prev_y = last_saved_coords[i]
                #         current_x, current_y = coords[i]
                #         if abs(prev_y - current_y) <= 5:
                #             current_y = prev_y
                #         if abs(prev_x - current_x) <= 5:
                #             current_x = prev_x
                #         last_saved_coords[i] = (current_x, current_y)
                if last_saved_coords[i] is None:
                    last_saved_coords[i] = coords[i]
                else:
                    # Compare the coordinates to see if it's good enough
                    prev_x, prev_y = last_saved_coords[i]
                    current_x, current_y = coords[i]
                    if abs(prev_y - current_y) <= 5:
                        current_y = prev_y
                    if abs(prev_x - current_x) <= 5:
                        current_x = prev_x
                    last_saved_coords[i] = (current_x, current_y)

        for a, f in enumerate(frame):
            if not ret[a]:
                pass
            else:
                if None in last_saved_coords.values():
                    h, w = f.shape[:2]
                    x, y = unmatched_pos[a]
                    background_img[y:y+h, x:x+w, :] = f/255.0
                else:
                    (h, w) = f.shape[:2]
                    posx, posy = last_saved_coords[a]
                    try:
                        background_img[posy:posy+h, posx:posx+w, :] = f/255.0
                    except ValueError:
                        print("Wrong perspective transform of frame {}".format(a))
                        unmatched_pos_x, unmatched_pos_y = unmatched_pos[a]
                        background_img[unmatched_pos_y:unmatched_pos_y+h, unmatched_pos_x:unmatched_pos_x+w, :] = f/255.0

        cv2.imshow("Background_output", background_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    for c in cap:
        c.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
