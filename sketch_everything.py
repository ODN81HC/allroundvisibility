import time
import threading
from queue import Queue
import numpy as np
import cv2

# def func(input_str):
#     print("This is the string that we want to print: {}".format(input_str))
#     time.sleep(0.8)
#     print("Back to 2nd thread")
#     return "This is ok"

# class ThreadWithReturnedVal(threading.Thread):
#     def __init__(self, group=None, target=None, name=None,
#                             args=(), kwargs=None, Verbose=None):
#         threading.Thread.__init__(self, group, target, name, args, kwargs)
#         self._return = None
    
#     def run(self):
#         print(type(self._target))
#         if self._target is not None:
#             self._return = self._target(*self._args, **self._kwargs)

#     def join(self, *args):
#         threading.Thread.join(self, *args)
#         return self._return
    
# new_thread = ThreadWithReturnedVal(target=func, args=("Oliver",))
# new_thread.start()
# print(new_thread.join())
# print("Go back to main thread")

last_saved_coords = {1:None, 2:None, 3:None, 4:None}
coords = {1: 'a', 2:'b', 4:'c'}

for i in last_saved_coords.keys():
    try:
        coords[i]
    except KeyError:
        print("No matching between frames")
    else:
        last_saved_coords[i] = coords[i]

print(last_saved_coords)

a = [18, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 19]
print(np.median(a))


iter_idx = np.arange(4).tolist()
iter_idx.append(iter_idx[0])
print(iter_idx)

list_a = [1, 2, 3, 4]
print(np.array(list_a).dtype)


            # # AnTon's implementation
            # frames = []
            # # Extract the list name
            # for name in self.names:
            #     norm_img = self.QImageToCvMat(videoStreamData[name]['source'])
            #     frames.append(norm_img)

            # if count <= 15 and self.commission_flag:
            #     self.commission_stitcher.stitch(frames)
            # else:
            #     if not self.recommend_pos:
            #         self.recommend_pos = self.commission_stitcher.recommend_coords()
            #         self.stitcher = Stitcher(self.unmatched_pos, self.recommend_pos)
            #         self.commission_flag = False
            #         count = 0
            #     # Alignment phase
            #     if count >= 10:
            #         self.coords = self.stitcher.stitch(frames)
            #         # Reset the counter
            #         count = 0

            # # Give out the coordinates
            # for i, imageName in enumerate(self.names):
            #     if self.commission_flag:
            #         x, y = self.unmatched_pos[i]
            #     else:
            #         if not self.coords:
            #             x, y = self.recommend_pos[i]
            #         else:
            #             f_coords = self.coords[i].get_coords()
            #             x, y = f_coords
            #             # if f_status == 1:
            #             #     w, h = videoStreamData[imageName]['w'], videoStreamData[imageName]['h']
            #             #     cv2.rectangle(videoStreamData[imageName]['image'], (0, 0), (w-1, h-1), (0, 0, 255), 2)
            #     self.notifyUpdatePosition.emit(imageName, y, x)

        # def QImageToCvMat(self, incomingImage):
        #     '''  Converts a QImage into an opencv MAT format  '''

        #     incomingImage = incomingImage.convertToFormat(QImage.Format_RGB32)

        #     width = incomingImage.width()
        #     height = incomingImage.height()

        #     ptr = incomingImage.bits()
        #     ptr.setsize(incomingImage.byteCount())
        #     arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
        #     return arr

# img = cv2.imread('./videos/bottom.png')
# cv2.imshow('frame', img)
# cv2.waitKey(0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# result = np.zeros_like(gray)
# shaded_img = cv2.merge([result, result, gray])
# cv2.imshow('result', shaded_img)
# cv2.waitKey(0)

a = np.array([1, 2, 3])
print(a.type)