import os
import configparser
import argparse
import cv2
import numpy as np
# import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

class Rectangle(object):
    """
    This class represents a single subvideo that is made from a part of the original
    video, and is specified by the bounding box coordinates, with the attibutes of it.
    
    Parameters:
    -----------
    `name`: [str] the name of the rectangle, and also the name of the output mp4
                    file.
    `coords`: [list((x, y, w, h))] the coordinates of the rectangle, and is defined
                by (x, y, w, h), in which (x, y) is the top-left position of the
                rectangle, and (w, h) is the width and height of the bounding box.
    `movingBox`: [OrderedDict(time, pair(x, y))] is the OrderedDict() that specifies
                the moving in pixel in x and y in time.
    Attributes:
    -----------
    `framelist`: [list()] the list of frames of the rectangle based on the condition
                above.
    `isDynamic`: [bool] specify if the bounding is moving over time in the original
                video.
    """
    def __init__(self, name, coords, movingBox):
        self.name = name
        self.coords = coords
        self.movingBox = movingBox
        if self.movingBox is None:
            self.isDynamic = False
        else:
            self.isDynamic = True
        self.framelist = []


class CustomParser(configparser.ConfigParser):
    """
    This class is the custom modification of the .ini parser file, which allow to turn the
    normal parse method of the class `configparser` into a list of define class.
    The idea of this is inspired from here:
    https://stackoverflow.com/questions/3220670/read-all-the-contents-in-ini-file-into-dictionary-with-python
    """
    def as_class(self):
        rects = []
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
            name = k
            coords = (int(d[k]['x']), int(d[k]['y']), int(d[k]['w']), int(d[k]['h']))
            try:
                d[k]['time']
            except KeyError:
                movingBox = None
            else:
                timeList = [int(i) for i in d[k]['time'].split(',')]
                offset_x_list = [int(i) for i in d[k]['offset_x'].split(',')]
                offset_y_list = [int(i) for i in d[k]['offset_y'].split(',')]
                movingBox = OrderedDict()
                for i in range(0, len(timeList)):
                    movingBox[timeList[i]] = (offset_x_list[i], offset_y_list[i])
            newRect = Rectangle(name, coords, movingBox)
            rects.append(newRect)
        return rects

class DataSourceTool():
    """
    This class the the main tool for the video cutting and exporting.
    It accepts the rectangles that are being recorded beforehand and generate the video
    cutting method to export the video.
    However, this class works best for a video file, not the video stream (webcam, URL camera)
    because the fps capture using Cv2 work best with the video file.

    Parameters:
    -----------
    `inputVid`: [str] the input video file path.
    `rects`: [list(Rectangle())] the list of the Rectangle class above.
    """
    def __init__(self, inputVid, rects):
        self.inputVid = inputVid
        self.rects = rects
        self.fps = 0

    def load_frames(self):
        """
        This method will create a video frame list for each of the pre-defined subvideos
        As the input video is being played, the output video will then be cut when condition
        and append into the framelist.
        """
        cap = cv2.VideoCapture(self.inputVid)
        if not cap.isOpened():
            raise RuntimeError("Cannot open the video")
        else:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            n_frame = 0
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                else:
                    # You can remove this if you know your boxes cut. Else, if using
                    # the test.ini file for the cutting, it's recommend to resize into
                    # (1280, 720)
                    frame = cv2.resize(frame, (1280, 720))
                    n_frame += 1
                    elapsed = n_frame * (1/self.fps)
                    # Take out the rectangles coordinates
                    for rect in self.rects:
                        (x, y, w, h) = rect.coords
                        if rect.isDynamic:
                            timekey = list(rect.movingBox.keys())
                            timekey_offset = np.array([elapsed - i for i in rect.movingBox.keys()])
                            # Extract the smallest positive index
                            try:
                                min(timekey_offset[timekey_offset >= 0])
                            except ValueError:
                                subframe = frame[y:y+h, x:x+w, :]
                                rect.framelist.append(subframe)
                            else:
                                smallest_posidx = list(timekey_offset).index(min(timekey_offset[timekey_offset >= 0]))
                                current_key = timekey[smallest_posidx]
                                offset_x, offset_y = rect.movingBox[current_key]
                                subframe = frame[y+offset_y:y+h+offset_y, x+offset_x:x+offset_x+w, :]
                                rect.framelist.append(subframe)
                        else:
                            subframe = frame[y:y+h, x:x+w, :]
                            rect.framelist.append(subframe)

            cap.release()
            cv2.destroyAllWindows()

    def merge_videos(self):
        """
        This method is the main merge of the frame lists of the rectangles that are configured above.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for rect in self.rects:
            print("[INFO] The processing video is: {}".format(rect.name))
            output = os.path.join('./videos', '{}.mp4'.format(rect.name))
            h, w = rect.framelist[0].shape[:2]
            wannabe_size = (w, h)
            out = cv2.VideoWriter(output, fourcc, self.fps, wannabe_size)

            for image in tqdm(rect.framelist):
                out.write(image)

            out.release()
            cv2.destroyAllWindows()
    
    def run(self):
        print("[INFO] Load frames...")
        self.load_frames()
        print("[INFO] Merge video...")
        self.merge_videos()

def main(inputpath, inifilepath):
    ini_parser = CustomParser()
    ini_parser.read(inifilepath)
    rects = ini_parser.as_class()
    data_tool = DataSourceTool(inputpath, rects)
    data_tool.run()

if __name__ == "__main__":
    mainparser = argparse.ArgumentParser(description='Specify the input video and the .ini file path used.')
    mainparser.add_argument('--input', default='./videos/LF.mp4', help='The input video file path.')
    mainparser.add_argument('--ini', default='test.ini', help='The ini file name/path.')
    args = mainparser.parse_args()

    input_path = args.input
    inifile_path = args.ini
    main(input_path, inifile_path)
    # Run the tool by: 
    # python dataSourceTool_test.py --input ./videos/LF.mp4 --ini test.ini
