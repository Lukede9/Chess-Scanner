import cv2
import numpy as np
import keras
import os
import shutil
from datetime import datetime
from cv_chess_functions import (read_img,
                               canny_edge,
                               hough_line,
                               h_v_lines,
                               line_intersections,
                               cluster_points,
                               augment_points,
                               write_crop_images,
                               grab_cell_files,
                               classify_cells,
                               fen_to_image,
                               atoi)


font = cv2.FONT_HERSHEY_SIMPLEX

training_folders = {n:'empty_square' for n in range(64)}

# Black pieces
for n in range(8, 16):
    training_folders[n] = 'black_p'
training_folders[0] = training_folders[7] = 'black_r'
training_folders[1] = training_folders[6] = 'black_n'
training_folders[2] = training_folders[5] = 'black_b'
training_folders[3] = 'black_q'
training_folders[4] = 'black_k'

# White pieces
for n in range(48, 56):
    training_folders[n] = 'white_P'
training_folders[56] = training_folders[63] = 'white_R'
training_folders[57] = training_folders[62] = 'white_N'
training_folders[58] = training_folders[61] = 'white_B'
training_folders[59] = 'white_Q'
training_folders[60] = 'white_K'


def rescale_frame(frame):
    dim = (750, 750)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def mark_corner(event,x,y,flags,param):
    global top_left_x, top_left_y, bottom_right_x, bottom_right_y, top_left_set, bottom_right_set
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),1,(255,0,0),-1)
        if not top_left_set:
            top_left_set = True
            top_left_x, top_left_y = x,y
        else:
            bottom_right_set = True
            bottom_right_x, bottom_right_y = x, y


def crop_and_save(points, square_width, square_height, mode):
    img_count = 0
    for (x,y) in points:
        cropped = img[y:y+square_height, x:x+square_width]
        resized = cv2.resize(cropped, (64,64), interpolation = cv2.INTER_AREA)
        if mode == 'predict':
            path = f'Data/{mode}/'
        else:
            path = f'Data/{mode}/{training_folders[img_count]}'
        
        if not os.path.exists(path):
            os.makedirs(path)
        if mode == 'predict':
            if img_count < 10:
                cv2.imwrite(path + f'data_image0{img_count}.jpeg', resized)
            else:
                cv2.imwrite(path + f'data_image{img_count}.jpeg', resized)
        else:
            cv2.imwrite(path + f'data_image{datetime.now().microsecond}.jpeg', resized)
        img_count += 1


def get_squares(top_left_x, top_left_y, bottom_right_x, bottom_right_y, mode):
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    
    if abs(width - height) > 40:
        print("The board is not a square.")
        return

    square_width = (width // 8) + 1
    square_height = (height // 8) + 1

    widths = range(top_left_x, bottom_right_x, square_width)
    heights = range(top_left_y, bottom_right_y, square_height)

    points = [[x, y] for x in widths for y in heights]
    points = sorted(points, key=lambda x: [x[1], x[0]])
    crop_and_save(points, square_width, square_height, mode)
    return points


purpose = input('Are we collecting data (c) or making predictions (p)?')

if purpose.startswith('c'):
    for board in os.listdir('Data/boards'):
        if board.startswith('board'):
            img, gray_blur = read_img(os.path.join('Data', 'boards', board))

            img = rescale_frame(img)
            gray_blur = rescale_frame(gray_blur)

            top_left_x, top_left_y = 0, 0
            bottom_right_x, bottom_right_y = 0, 0
            top_left_set, bottom_right_set = False, False
            
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', mark_corner)

            while(1):
                cv2.imshow('image',img)
                k = cv2.waitKey(20) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('a'):
                    print([[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]])
                    break


            if top_left_set and bottom_right_set:
                points = get_squares(top_left_x, top_left_y, bottom_right_x, bottom_right_y, 'train')

            cv2.destroyAllWindows()

elif purpose.startswith('p'):
    model = keras.models.load_model('piece_classifier')
    board = f'{input("what is the board filename?")}.png'
    img, gray_blur = read_img(os.path.join('Data', 'test_boards', board))
    img = rescale_frame(img)
    gray_blur = rescale_frame(gray_blur)

    top_left_x, top_left_y = 0, 0
    bottom_right_x, bottom_right_y = 0, 0
    top_left_set, bottom_right_set = False, False

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mark_corner)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('a'):
            print([[top_left_x, top_left_y], [bottom_right_x, bottom_right_y]])
            break

    if top_left_set and bottom_right_set:
        points = get_squares(top_left_x, top_left_y, bottom_right_x, bottom_right_y, 'predict')
        img_filename_list = grab_cell_files()
        fen = classify_cells(model, img_filename_list)
        print(fen)
        board = fen_to_image(fen)
        print(board)