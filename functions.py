import cv2
import numpy as np


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, 
                          image.shape[1::-1], 
                          flags=cv2.INTER_LINEAR)
  return result

def draw_shapes_on(img,dots,form):
    dimg = deepcopy(img)
    if form == 'r':
        # draw rectangles
        for (x,y,w,h) in dots:
            dimg = cv2.rectangle(dimg,(x,y),(x+w,y+h),(255,0,0),2)
    if form == 'o':
        # draw rectangles
        for (x,y,w,h) in dots:
            dimg = cv2.circle(dimg, center=(x+(w//2),y+(h//2)), radius =1, thickness=2, color=1)
    return dimg;



def draw_grid_lines(img, pxvals, endval, indictator):
    
    if indictator == 'r':
        for value in pxvals:
            cv2.line(img, (0, value), (endval, value), 128, thickness=1)
            
    if indictator == 'c':
        for value in pxvals:
            cv2.line(img, (value, 0), (value, endval), 128, thickness=1)

    return img;


def draw_grid(img, grid):
    for square in grid:
        boxy = np.ones(img[square[0]:square[2],square[1]:square[3]].shape)
        boxy[:,:,0] = boxy[:,:,0] * random.randint(0,255);
        boxy[:,:,1] = boxy[:,:,1] * random.randint(0,255);
        boxy[:,:,2] = boxy[:,:,2] * random.randint(0,255);
        img[square[0]:square[2],square[1]:square[3]] = boxy;
    return img;

