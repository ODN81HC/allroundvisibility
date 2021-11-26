import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

videoDir = './videos/NewYork.mp4'
cap = cv2.VideoCapture(videoDir)

if not cap.isOpened():
    raise RuntimeError("Cannot open the video")
else:
    left, right, top, bottom = [], [], [], []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        height, width = frame.shape[:2]
        # Take out the subframes for videos
        # left.append(frame[100:620, 100:500, :])
        # right.append(frame[100:620, 780:1180, :])
        # top.append(frame[0:200, 150:1130, :])
        # bottom.append(frame[520:720, 150:1130, :])

        # Sketch the rectangles
        cv2.rectangle(frame, (100, 0), (1180, 300), (0, 0, 255), 2) # Top rectangle
        cv2.rectangle(frame, (100, 420), (1180, 720), (0, 0, 255), 2) # Bottom rectangle
        cv2.rectangle(frame, (100, 100), (600, 620), (0, 255, 0), 2) # Left rectangle
        cv2.rectangle(frame, (680, 100), (1180, 620), (0, 255, 0), 2) # Right rectangle
        # Show video
        cv2.putText(frame, "FPS: {}".format(fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Video", frame)
        
        # Exit the video
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # for name, subvideo in zip(['left', 'right', 'top', 'bottom'], [left, right, top, bottom]):
    #     print('[INFO] Current processing video: {}'.format(name))
    #     output = os.path.join('./videos', '{}_subvideo.mp4'.format(name))
    #     h, w = subvideo[0].shape[:2]
    #     wannabe_size = (w, h)
    #     out = cv2.VideoWriter(output, fourcc, fps, wannabe_size)
        
    #     for n, image in tqdm(enumerate(subvideo)):
    #         out.write(image)

    #     out.release()
    #     cv2.destroyAllWindows()
