import math
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import re
import glob
import PIL
import os


# Convert image from RGB to BGR
def convert_image_to_bgr_numpy_array(image_path, size=(224, 224)):
    image = PIL.Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    img_data = np.flip(img_data, axis=2)
    return img_data


# Adjust image into (1, 224, 224, 3)
def prepare_image(image_path):
    im = convert_image_to_bgr_numpy_array(image_path)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = np.expand_dims(im, axis=0)
    return im


# Reads in the cropped images to a list
def grab_cell_files(folder_name='./Data/predict/'):  
    return sorted([os.path.join(folder_name, x) for x in os.listdir(folder_name) if x.startswith('square')])


# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def classify_cells(model, img_filename_list):
    category_reference = {0: 'b', 1: 'k', 2: 'n', 3: 'p', 4: 'q', 5: 'r', 6: '1', 7: 'B', 8: 'K', 9: 'N', 10: 'P',
                          11: 'Q', 12: 'R'}
    pred_list = []
    for filename in img_filename_list:
        img = prepare_image(filename)
        out = model.predict(img)
        top_pred = np.argmax(out)
        pred = category_reference[top_pred]
        pred_list.append(pred)

    fen = ''.join(pred_list)
    fen = fen[::-1]
    fen = '/'.join(fen[i:i + 8] for i in range(0, len(fen), 8))
    sum_digits = 0
    for i, p in enumerate(fen):
        if p.isdigit():
            sum_digits += 1
        elif p.isdigit() is False and (fen[i - 1].isdigit() or i == len(fen)):
            fen = fen[:(i - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1)) + fen[i:]
            sum_digits = 0
    if sum_digits > 1:
        fen = fen[:(len(fen) - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1))
    fen = fen.replace('D', '')
    return fen


# Converts the FEN into a PNG file
def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")
    return board
