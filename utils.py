import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR = (0,0, 255)

def load_pngs_dict(path):
    '''
    use path and create dict of id and images
    '''
    imgs_ids = {}
    
    for file in os.listdir(path):
        img_file = os.path.join(path, file)
        id = file.split('.')[0]

        img = cv2.imread(img_file)
        imgs_ids[id] = img

    print('Dict created!')
    print('Loaded', len(imgs_ids), 'images.')

    return imgs_ids

def get_crop_image(img, h, w):
    '''
    crop img to fit tiling requirement
    '''
    height, width = img.shape[:2]

    new_width = width - (width % w)
    new_height = height - (height % h)

    img = img[0:new_height, 0:new_width]

    return img

def get_tiles_by_grid(img, r, c, ignore_excess=True):
    '''
    Parameters
    ----------
    img: shape=(h,w,ch), dtype=np.uint8
        input image
    r: int
        number of rows in grid
    c: int
        number of columns in grid
    ignore_excess: bool
        true if image should be cropped to ignore smaller tiles

    Returns
    -------
    Cropped image, shape = (h,w,ch)
    Dict of tiles generated, {id: tile}
    Image masked with tiles and ids, shape = (h,w,ch)
    '''

    if ignore_excess:
        img = get_crop_image(img, r, c)
    height, width = img.shape[:2]

    tile_w = int(width/c)
    tile_h = int(height/r)

    mask = img.copy()
    id = 0
    tiles = {}

    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            cv2.rectangle(mask, pt1=(x,y), pt2=(x+tile_w-1,y+tile_h-1), color=COLOR, thickness=1)
            cv2.putText(mask, str(id), (x,y+tile_h), FONT, 1, COLOR, 3, cv2.LINE_AA)
            tile = img[y:y+tile_h, x:x+tile_w]
            tiles[id] = tile
            id = id + 1

    return img, tiles, mask

def get_tiles_by_pixels(img, h, w, ignore_excess=True):
    '''
    Parameters
    ----------
    img: shape=(h,w,ch), dtype=np.uint8
        input image
    h: int
        height of tile in pixels
    w: int
        width of tile in pixels
    ignore_excess: bool
        true if image should be cropped to ignore smaller tiles

    Returns
    -------
    Cropped image, shape = (h,w,ch)
    Dict of tiles generated, {id: tile}
    Image masked with tiles and ids, shape = (h,w,ch)
    '''

    if ignore_excess:
        img = get_crop_image(img, h, w)
    height, width = img.shape[:2]
    mask = img.copy()

    id = 0
    tiles = {}

    for y in range(0, height, h):
        for x in range(0, width, w):
            cv2.rectangle(mask, pt1=(x,y), pt2=(x+w-1,y+h-1), color=COLOR, thickness=1)
            cv2.putText(mask, str(id), (x,y+h), FONT, 1, COLOR, 3, cv2.LINE_AA)
            tile = img[y:y+h, x:x+w]
            tiles[id] = tile
            id = id + 1

    return img, tiles, mask