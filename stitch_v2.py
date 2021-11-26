import os
import time
import argparse
import cv2
import numpy as np
from stitch_v1 import Matcher
from stitch_configs import *

class Commission_Stitcher():
    """This is only to use in commissioning phase, in which to find a recommend position
    for the subvideos and later modifications
    """
    def __init__(self):
        self.images = []
        self.matcher = Matcher()
        self.frames = {}

    def stitch(self, images):
        self.images = images

        for i in range(len(self.images)):
            if i < len(self.images)-1:
                id1 = i
                id2 = i+1
            else:
                id1 = 0
                id2 = len(self.images)-1
            
            img1 = self.images[id1]
            img2 = self.images[id2]
            # Find the homography matrix
            M = self.matcher.match(img1, img2)

            if M is None:
                break
            else:
                img2_h, img2_w = img2.shape[:2]

                # Map the points of img2 to img1 using the homography matrix
                pts = np.float32([[0, 0], [0, img2_h-1], [img2_w-1, img2_h-1], [img2_w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                # Remove the bad dst points (based on observation)
                x1, y1 = np.int32(dst)[0][0][0], np.int32(dst)[0][0][1]
                x2, y2 = np.int32(dst)[1][0][0], np.int32(dst)[1][0][1]
                if abs(x1 - x2) > 30 or abs(y1 - y2) < 100:
                    break

                if id1 == 0:
                    if id1 not in self.frames.keys():
                        self.frames[id1] = [LEFT_REF_COORDS]
                        self.frames[id2] = [(LEFT_REF_COORDS[0]+x1, LEFT_REF_COORDS[1]+y1)]
                    else:
                        self.frames[id1].append(LEFT_REF_COORDS)
                        self.frames[id2].append((LEFT_REF_COORDS[0]+x1, LEFT_REF_COORDS[1]+y1))
                else:
                    ref_posx, ref_posy = self.frames[id1][-1]
                    if id2 not in self.frames.keys():
                        self.frames[id2] = [(ref_posx+x1, ref_posy+y1)]
                    else:
                        self.frames[id2].append((ref_posx+x1, ref_posy+y1))

    def recommend_coords(self):
        img_pos = []
        stddev = []
        for value in self.frames.values():
            list_x, list_y = [], []
            for (x, y) in value:
                list_x.append(x)
                list_y.append(y)
            stddev_x = np.std(list_x)
            stddev_y = np.std(list_y)
            img_pos.append((np.int32(np.median(list_x)), np.int32(np.median(list_y))))
            stddev.append((stddev_x+10, stddev_y+10))
        return img_pos, stddev


class Frame():
    """ Frame class that contains the attributes of the Frame object, which can update
    the coordinates, frame status whether it's still in the recommend coordinates or not.
    """
    def __init__(self, current_coords, iscorr, max_uncorr=MAX_UNCORR):
        self.max_uncorr = max_uncorr*2
        self.current_coords = current_coords
        if iscorr:
            self.num_uncorr = 0
        else:
            self.num_uncorr = 1
        self.status = 0
    
    def update(self, current_coords, iscorr):
        self.current_coords = current_coords
        if iscorr:
            self.num_uncorr = 0
        else:
            self.num_uncorr += 1
        
        if self.num_uncorr >= self.max_uncorr:
            self.status = 1
        else:
            self.status = 0

    def get_status(self):
        return self.status

    def get_coords(self):
        return self.current_coords

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
    `unmatched_pos`:  [list((x, y))] Positions of frames if they're unmatched.
    `recommend_pos`" [list((x, y))] Positions of frames after commisioning.
    """
    def __init__(self, recommend_pos: list, std_dev: list):
        self.matcher_obj = Matcher()
        self.recommend_pos = recommend_pos
        self.std_dev = std_dev
        self.imgs_pos = {}
        self.images = []

    def frame_update(self, idx: int, iscorr: bool = True):
        if idx not in self.imgs_pos.keys():
            self.imgs_pos[idx] = Frame(self.recommend_pos[idx], iscorr)
        else:
            self.imgs_pos[idx].update(self.recommend_pos[idx], iscorr)

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
        `imgs_pos`:         [dict(img_key: Frame())] the coordinates position of the frames.
        """
        iter_idx = np.arange(len(images)).tolist()
        iter_idx.append(iter_idx[0])
        start_time = time.time()

        for i in range(len(iter_idx)-1):
            id1 = iter_idx[i]
            id2 = iter_idx[i+1]
            M = self.matcher_obj.match(images[id1], images[id2])

            if M is None:
                print("[WARNING] No correlation between frame {} and {}".format(id1, id2))
                self.frame_update(id1, False)
                self.frame_update(id2, False)
            else:
                img2_h, img2_w = images[id2].shape[:2]

                pts = np.float32([[0, 0], [0, img2_h-1], [img2_w-1, img2_h-1], [img2_w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                x_offset, y_offset = np.int32(dst)[0][0][0], np.int32(dst)[0][0][1]

                # Check the condition
                ref_x, ref_y = self.recommend_pos[id1]
                compare_x, compare_y = self.recommend_pos[id2]
                if abs(ref_x + x_offset - compare_x) < self.std_dev[id2][0] and abs(ref_y + y_offset - compare_y) < self.std_dev[id2][1]:
                    self.frame_update(id1)
                    self.frame_update(id2)
                else:
                    self.frame_update(id1, False)
                    self.frame_update(id2, False)

        # self.images = images[:]
        # self.images.append(images[0]) # Therefore, the list self.images will have [left, top, right, bottom, left]
        
        # recommend_coords = self.recommend_pos[:]
        # recommend_coords.append(self.recommend_pos[0]) # Copy list so that it will have [left, top, right, bottom, left]

        # start_time = time.time()
        
        # for i in range(len(self.images)-1):
        #     img1 = self.images[i]
        #     img2 = self.images[i+1]

        #     M = self.matcher_obj.match(img1, img2)

        #     if M is None:
        #         print("No correlation between frame {} and {}".format(i, i+1))
        #         if i not in self.imgs_pos.keys():
        #             self.imgs_pos[i] = Frame(recommend_coords[i], False)
        #         else:
        #             if self.imgs_pos[i].get_status() == 0:
        #                 self.imgs_pos[i].update(recommend_coords[i], False)
        #             else:
        #                 self.imgs_pos[i].update(self.unmatched_pos[i], False)
        #     else:
        #         img2_h, img2_w = img2.shape[:2]

        #         pts = np.float32([[0, 0], [0, img2_h-1], [img2_w-1, img2_h-1], [img2_w-1, 0]]).reshape(-1, 1, 2)
        #         dst = cv2.perspectiveTransform(pts, M)
        #         x_offset, y_offset = np.int32(dst)[0][0][0], np.int32(dst)[0][0][1]

        #         # Check the condition
        #         ref_x, ref_y = recommend_coords[i]
        #         compare_x, compare_y = recommend_coords[i+1]
        #         if abs(ref_x + x_offset - compare_x) < 20 and abs(ref_y + y_offset - compare_y) < 20:
        #             if i not in self.imgs_pos.keys():
        #                 self.imgs_pos[i] = Frame(recommend_coords[i], True)
        #             else:
        #                 self.imgs_pos[i].update(recommend_coords[i], True)
        #         else:
        #             if i not in self.imgs_pos.keys():
        #                 self.imgs_pos[i] = Frame(recommend_coords[i], False)
        #             else:
        #                 if self.imgs_pos[i].get_status() == 0:
        #                     self.imgs_pos[i].update(recommend_coords[i], False)
        #                 else:
        #                     self.imgs_pos[i].update(self.unmatched_pos[i], False)

        print("[INFO] Stitching took {:.2f} seconds".format(time.time() - start_time))

        return self.imgs_pos


def main(folder_dir):
    output_vids = OUTPUT_VIDS
    unmatch_pos = UNMATCHED_COORDS
    background_img = np.zeros(BACKGROUND_IMG)
    recommend_pos = []
    std_dev = []
    coords = {}
    commission_stitcher = Commission_Stitcher()
    stitcher = None

    # Custom implementation
    desire_frame_size = (960, 600)
    output_dir = OUTPUT_DIR
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outframe = cv2.VideoWriter(output_dir, fourcc, 20, desire_frame_size)

    cap = []
    # Option to get the camera views from webcams
    if np.array(output_vids).dtype == np.int32:
        for vid in output_vids:
            cap.append(cv2.VideoCapture(vid))
    else:
        # sub-videos files
        for vid in output_vids:
            cap.append(cv2.VideoCapture(os.path.join(folder_dir, vid)))
    
    ret = [None] * len(output_vids)
    frame = [None] * len(output_vids)

    commission_time = COMMISSION_TIME
    commission_flag = True
    check_interval = CHECK_INTERVAL
    print("[INFO] Start commissioning phase...")
    start_time = time.time()

    while True:
        # Get image list
        for i, c in enumerate(cap):
            if c is not None:
                ret[i], frame[i] = c.read()
        
        # Do the commissioning phase first
        if (time.time() - start_time) < commission_time and commission_flag:
            commission_stitcher.stitch(frame)
        else:
            # Finish the commissioning phase and output the recommendation coordinates
            if not recommend_pos:
                print("[INFO] Commissioning finished...")
                recommend_pos, std_dev = commission_stitcher.recommend_coords()
                stitcher = Stitcher(recommend_pos, std_dev)
                # Reset the timer to turn to the auto alignment phase
                start_time = time.time()
                commission_flag = False
                # Clean the background
                background_img = np.zeros(BACKGROUND_IMG)
            # Do the auto alignment phase
            if time.time() - start_time >= check_interval:
                coords = stitcher.stitch(frame)
                # Reset the timer
                start_time = time.time()

        for i, f in enumerate(frame):
            if not ret[i]:
                pass
            else:
                h, w = f.shape[:2]
                if commission_flag:
                    x, y = unmatch_pos[i]
                    background_img[y:y+h, x:x+w] = f/255.0
                else:
                    if not coords:
                        x, y = recommend_pos[i]
                        background_img[y:y+h, x:x+w] = f/255.0
                    else:
                        f_coords, f_status = coords[i].get_coords(), coords[i].get_status()
                        x, y = f_coords
                        # background_img[y:y+h, x:x+w] = f/255.0
                        if f_status == 1:
                            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                            blank = np.zeros_like(gray)
                            shaded_img = cv2.merge([blank, blank, gray])
                            background_img[y:y+h, x:x+w] = shaded_img/255.0
                            # cv2.rectangle(background_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        else:
                            background_img[y:y+h, x:x+w] = f/255.0
        # Only show vid after the commission phase
        # if not commission_flag:
        #     cv2.imshow("Background_output", background_img)
        resized = cv2.resize(background_img, desire_frame_size)
        cv2.imshow("Background_output", resized)
        
        # outframe.write((resized*255).astype(np.uint8))
        # Keyboard interrupt
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    for c in cap:
        c.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify the input folder used')
    parser.add_argument('--input', default='./videos/LF_subvids', help='The input video folder path')
    args = parser.parse_args()

    input_path = args.input
    main(input_path)
    # Run the script by: 
    # python stitch_v2.py --input ./videos/LF_subvids
