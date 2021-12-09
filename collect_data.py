import cv2
import numpy as np
import os
from datetime import datetime

'''
Collect data from screenshots of boards in their starting positions.
This will allow me to collect a lot of data quickly but there will be a couple problems:
1. The white queen will always appear on a white square (similar problems for all chess royalty).
2. The algorithm might pick up on the superscript coordinates (e.g. "1" on the 1st rank).

Therefore, I need a bit more data. But this is a good starting point.
'''

paths = {
    "TRAIN" : os.path.join('Data', 'train'),
    "TEST" : os.path.join('Data', 'test'),
    "PREDICT" : os.path.join('Data', 'predict'),
    "SET_BOARDS" : os.path.join('Data', 'boards', 'set_boards'),
    "CROPPED_BOARDS" : os.path.join('Data', 'boards', 'cropped_boards')
}

pieces = ['BP', 'BR', 'BN', 'BB', 'BQ', 'BK', 'Empty', 'WP', 'WR', 'WN', 'WB', 'WQ', 'WK']

# Starting locations for pieces. Used to collect data quickly
training_folders = {n:'empty' for n in range(64)}

# Black pieces
for n in range(8, 16):
    training_folders[n] = 'BP'
training_folders[0] = training_folders[7] = 'BR'
training_folders[1] = training_folders[6] = 'BN'
training_folders[2] = training_folders[5] = 'BB'
training_folders[3] = 'BQ'
training_folders[4] = 'BK'

# White pieces
for n in range(48, 56):
    training_folders[n] = 'WP'
training_folders[56] = training_folders[63] = 'WR'
training_folders[57] = training_folders[62] = 'WN'
training_folders[58] = training_folders[61] = 'WB'
training_folders[59] = 'WQ'
training_folders[60] = 'WK'

coords = range(0, 512, 64)
points = [(x,y) for x in coords for y in coords]
points = sorted(points, key=lambda x: [x[1], x[0]])

top_left_x, top_left_y = 0, 0
bottom_right_x, bottom_right_y = 0, 0
top_left_set, bottom_right_set = False, False


def mark_corner(event,x,y,flags, param):
    global top_left_x, top_left_y, bottom_right_x, bottom_right_y, top_left_set, bottom_right_set
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),1,(255,0,0),-1)
        if not top_left_set:
            top_left_set = True
            top_left_x, top_left_y = x,y
        else:
            bottom_right_set = True
            bottom_right_x, bottom_right_y = x, y


for board in os.listdir(paths['SET_BOARDS']):
    if board.startswith('board'):
        img = cv2.imread(os.path.join(paths['SET_BOARDS'], board))
        top_left_x, top_left_y = 0, 0
        bottom_right_x, bottom_right_y = 0, 0
        top_left_set, bottom_right_set = False, False
        
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', mark_corner)

        while(1):
            cv2.imshow('image', img)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                break

        if top_left_set and bottom_right_set:
            cropped = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            resized = cv2.resize(cropped, (512,512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(paths['CROPPED_BOARDS'] + f'/board{datetime.now().microsecond}.jpeg', resized)
        cv2.destroyAllWindows()


for board in os.listdir(paths['CROPPED_BOARDS']):
    img_count = 0
    if board.startswith('board'):
        img = cv2.imread(os.path.join(paths['CROPPED_BOARDS'], board))
        for (x,y) in points:
            cropped = img[y:y+64, x:x+64]
            path = f'Data/train/{training_folders[img_count]}/'

            if not os.path.exists(path):
                os.makedirs(path)
            cv2.imwrite(path + f'data_image{datetime.now().microsecond}.jpeg', cropped)
            img_count += 1