import cv2
import numpy as np
import keras
import os
import shutil
from datetime import datetime
from cv_chess_functions import (grab_cell_files,
                                classify_cells,
                                fen_to_image)
                                
paths = {
    "TRAIN" : os.path.join('Data', 'train'),
    "TEST" : os.path.join('Data', 'test'),
    "PREDICT" : os.path.join('Data', 'predict'),
    "SET_BOARDS" : os.path.join('Data', 'boards', 'set_boards'),
    "CROPPED_BOARDS" : os.path.join('Data', 'boards', 'cropped_boards')
}

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

current_turn = 'w'
w_kingside = ''
w_queenside = ''
b_kingside = ''
b_queenside = ''

def set_turn(x):
    global current_turn
    if x == 1:
        current_turn = 'w'
    else:
        current_turn = 'b'

def white_king_side(x):
    global w_kingside
    if x == 1:
        w_kingside = 'K'
    else:
        w_kingside = ''

def white_queen_side(x):
    global w_queenside
    if x == 1:
        w_queenside = 'Q'
    else:
        w_queenside = ''

def black_king_side(x):
    global b_kingside
    if x == 1:
        b_kingside = 'k'
    else:
        b_kingside = ''

def black_queen_side(x):
    global b_queenside
    if x == 1:
        b_queenside = 'q'
    else:
        b_queenside = ''


model = keras.models.load_model('piece_classifier')
img = cv2.imread(f'{input("what is the board filename?")}.png')

cv2.namedWindow('image')
cv2.setMouseCallback('image', mark_corner)
cv2.createTrackbar('White to move', 'image', 1, 1, set_turn)
cv2.createTrackbar('White King side', 'image', 0, 1, white_king_side)
cv2.createTrackbar('White Queen side', 'image', 0, 1, white_queen_side)
cv2.createTrackbar('Black King side', 'image', 0, 1, black_king_side)
cv2.createTrackbar('White Queen side', 'image', 0, 1, black_queen_side)


while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break

if top_left_set and bottom_right_set:
    cropped = img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    resized = cv2.resize(cropped, (512,512), interpolation=cv2.INTER_AREA)
    for i, (x,y) in enumerate(points):
        square = resized[y:y+64, x:x+64]
        if i < 10:
            path = os.path.join(paths['PREDICT'], f"square0{i}.jpeg")
        else:
            path = os.path.join(paths['PREDICT'], f"square{i}.jpeg")

        cv2.imwrite(path, square)

    img_filename_list = grab_cell_files()
    fen = classify_cells(model, img_filename_list)
    castling_rights = ''.join([w_kingside, w_queenside, b_kingside, b_queenside])
    if len(castling_rights) == 0: castling_rights = '-'
    fen = f"{fen} {current_turn} {castling_rights} - 0 1"
    print(fen)
    

cv2.destroyAllWindows()