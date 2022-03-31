import cv2, os, random, time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import MeanShift
from scipy.stats import linregress
from math import atan
from skimage.feature import hog
from joblib import load
from copy import deepcopy


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, 
                          image.shape[1::-1], 
                          flags=cv2.INTER_LINEAR)
  return result


def svm_row_check(img, grid_cols, row):
    linecount = 0;
    for col in grid_cols[:,0]:
        img = rot_img[ row:(row+24),col:(col+22)]
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
        lbpi = getLBPimage(img)
        arr = np.append(fd,lbpi.flatten())
        
        linecount += clf.predict(arr.reshape(1,-1))
    return linecount


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


def getLBPimage(gray_image):

    ### Step 0: Step 0: Convert an image to grayscale
    imgLBP = np.zeros_like(gray_image)
    neighboor = 3 
    for ih in range(0,gray_image.shape[0] - neighboor):
        for iw in range(0,gray_image.shape[1] - neighboor):
            ### Step 1: 3 by 3 pixel
            img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]
            center       = img[1,1]
            img01        = (img >= center)*1.0
            img01_vector = img01.T.flatten()
            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()
            ### Step 2: **Binary operation**:
            img01_vector = np.delete(img01_vector,4)
            ### Step 3: Decimal: Convert the binary operated values to a digit.
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2**where_img01_vector)
            else:
                num = 0
            imgLBP[ih+1,iw+1] = num
    return(imgLBP)




start_time = time.time()

## Define DATA
#HAAR_FIL = 'C:/Users/SebastianG/Desktop/Train_full/cascades/cascade.xml'
HAAR_FIL = 'C:/Users/SebastianG/Desktop/train_final/cascades/cascade.xml'
SOURCE_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/'
GRID_BORDR = [0,0,0,0];     # Area where candidates can be found
Y_TOL = 5   # Search tolerance (+/-) in y direction, px values.
X_TOL = 5   # Search tolerance (+/-) in x direction, px values.

Haar_par1 = 2 # With train_full it is 1.3
Haar_par2 = 0   # With train_full it is 5

# Load DATA
#os.chdir(SOURCE_DIR);
img = cv2.imread(SOURCE_DIR+'number2.jpg',0)
dimg = deepcopy(img)
haar_cascade = cv2.CascadeClassifier(HAAR_FIL)
binary_grid = np.zeros((img.shape))
x_px, y_px = img.shape;

# detect the dots, USING HAAR CLASSIFIR
dots = haar_cascade.detectMultiScale(dimg, Haar_par1, Haar_par2)

# Get the average box size of the haar classifier
# and assume this as the box size;
bb = np.mean(dots, axis=0)[2:].astype('int')
row_height, col_width = bb[0], bb[1]


# CLUSTR TH UNIQU LIN POINTS
y_unique = np.unique(dots[:,1]);
X = np.reshape(y_unique, (-1, 1))
ms = MeanShift(bandwidth=Y_TOL, bin_seeding=True)
ms.fit(X)


###                                         ###
###  estimate rotation and the line groups  ###
###               for rows                  ###

## take one line (defined by a shared label), and estimate the slope
y_vals = y_unique[ms.labels_==0];
ref_line=dots[np.isin(dots[:,1],y_vals)][:,:2]
slope, intercept, r_value, p_value, std_err = linregress(ref_line[:,0],ref_line[:,1])
rotation = np.rad2deg(atan(slope));

## rotate image
rot_img = rotate_image(img,rotation)
rot_imgd = deepcopy(rot_img)

# Get Dots from rotated image
dotsr = haar_cascade.detectMultiScale(rot_img, Haar_par1, Haar_par2)

# calculate the clustering once more, look for clusters.
y_unique, y_counts = np.unique(dotsr[:,1], return_counts=True);
X = np.reshape(y_unique, (-1, 1))
ms = MeanShift(bandwidth=Y_TOL, bin_seeding=True)
ms.fit(X)


# get center of the rows
row_centers = ms.cluster_centers_.astype('int').flatten()
row_centers.sort()

# get the difference, and find the one that happens
# most often
y_diff = np.diff(row_centers);
labels = np.zeros((len(y_diff)+1))
y_diff = np.insert(y_diff, 0, 0, axis=0)

cur_label = 1;
for i in range(0,len(y_diff)-1):
    if np.logical_and( ( y_diff[i+1] > row_height - Y_TOL ) , ( y_diff[i+1] < row_height + Y_TOL )  ):
        labels[i] = cur_label;
        labels[i+1] = cur_label;
    else:
        cur_label = cur_label+1;
        labels[i+1] = cur_label;
        cur_label = cur_label+1;



grid_rows = [];
cur_label = 1;
for ulabel in np.unique(labels):
    cur_pack = np.array(np.where(labels==ulabel))[0]
    if len(cur_pack) == 1:
        grid_rows.append([row_centers[cur_pack][0], cur_label ])
        grid_rows.append([row_centers[cur_pack].max() + row_height, cur_label ])
        cur_label += 1;
    if len(cur_pack) == 2:
        grid_rows.append([row_centers[cur_pack][0], cur_label ])
        grid_rows.append([row_centers[cur_pack][1], cur_label ])
        grid_rows.append([row_centers[cur_pack][1] + row_height, cur_label ])
        cur_label += 1;
    if len(cur_pack) == 3:
        grid_rows.append([row_centers[cur_pack][0], cur_label ])
        grid_rows.append([row_centers[cur_pack][1], cur_label ])
        grid_rows.append([row_centers[cur_pack][2], cur_label ])
        grid_rows.append([row_centers[cur_pack][2] + row_height, cur_label ])
        cur_label += 1;

grid_rows = np.array(grid_rows);


rot_imgd = deepcopy(rot_img)
for label in np.unique(grid_rows[:,1]):
    rows = grid_rows[grid_rows[:,1]==label][:,0]
    a = random.randint(0,255);
    # GT rAndom ColOR
    for row in rows:
        cv2.line(rot_imgd, (0, row), (y_px, row),  a, 1)





###                                         ###
###  estimate rotation and the line groups  ###
###               for columns               ###
rot_imgd = deepcopy(rot_img)

# Get Dots from rotated image
dotsr = haar_cascade.detectMultiScale(rot_img, Haar_par1, Haar_par2)

# calculate the clustering once more, look for clusters.
x_unique, x_counts = np.unique(dotsr[:,0], return_counts=True);
X = np.reshape(x_unique, (-1, 1))
ms = MeanShift(bandwidth=X_TOL, bin_seeding=True)
ms.fit(X)

column_centers = ms.cluster_centers_.astype('int').flatten();
column_centers.sort();


x_diff = np.diff(column_centers);
labels = np.zeros((len(x_diff)+1))
x_diff = np.insert(x_diff, 0, 0, axis=0)



cur_label = 1;
for i in range(0,len(x_diff)-1):
    if np.logical_and( ( x_diff[i+1] > col_width - X_TOL ) , ( x_diff[i+1] < col_width + X_TOL )  ):
        labels[i] = cur_label;
        labels[i+1] = cur_label;
    else:
        cur_label = cur_label+1;
        labels[i+1] = cur_label;
        cur_label = cur_label+1;


grid_cols = [];
cur_label = 1;
for ulabel in np.unique(labels):
    cur_pack = np.array(np.where(labels==ulabel))[0]
    if len(cur_pack) == 1:
        grid_cols.append([column_centers[cur_pack][0], cur_label ])
        grid_cols.append([column_centers[cur_pack].max() + col_width, cur_label ])
        cur_label += 1;
    if len(cur_pack) == 2:
        grid_cols.append([column_centers[cur_pack][0], cur_label ])
        grid_cols.append([column_centers[cur_pack][1], cur_label ])
        grid_cols.append([column_centers[cur_pack][1] + col_width, cur_label ])
        cur_label += 1;

grid_cols = np.array(grid_cols);


rot_imgd = deepcopy(rot_img)
for label in np.unique(grid_cols[:,1]):
    columns = grid_cols[grid_cols[:,1]==label][:,0]
    a = random.randint(0,255);
    for column in columns:
        cv2.line(rot_imgd, (column, 0), (column, x_px),  a, 1)


GRID_BORDR[0] = dotsr[:,0].min();
GRID_BORDR[1] = dotsr[:,0].max() + row_height;
GRID_BORDR[2] = dotsr[:,1].min();
GRID_BORDR[3] = dotsr[:,1].max() + col_width;


rot_imgd = deepcopy(rot_img)
rot_imgd = cv2.cvtColor(rot_imgd,cv2.COLOR_GRAY2RGB)
for label in np.unique(grid_cols[:,1]):
    columns = grid_cols[grid_cols[:,1]==label][:,0]
    a,b,c, = random.randint(0,255),random.randint(0,255),random.randint(0,255);
    # GT rAndom ColOR
    for column in columns:
        cv2.line(rot_imgd, (column, GRID_BORDR[2]), (column, GRID_BORDR[3]),  (a,b,c), 1)
        #grid_rows.append(line)

for label in np.unique(grid_rows[:,1]):
    rows = grid_rows[grid_rows[:,1]==label][:,0]
    a,b,c, = random.randint(0,255),random.randint(0,255),random.randint(0,255);
    # GT rAndom ColOR
    for row in rows:
        cv2.line(rot_imgd, (0, row), (y_px, row),  (a,b,c), 1)
        #grid_rows.append(line)



binary_grid = np.zeros((img.shape))
for dot in dotsr:
    yc = dot[1] + (dot[3] // 2 )
    xc = dot[0] + (dot[2] // 2 )
    
    binary_grid = cv2.circle(binary_grid, center=(xc,yc), radius =1, thickness=2, color=1)





## extract grid structure, pretty greedy
grid_grid = [];
grid_rows = np.sort(grid_rows, axis=0)
grid_cols = np.sort(grid_cols, axis=0)
ru, rc = np.unique(grid_rows[:,1], return_counts=True)
gu, gc = np.unique(grid_cols[:,1], return_counts=True)

for row_group in np.unique(grid_rows[:,1]):
    for row in grid_rows[grid_rows[:,1]==row_group][:,0][:-1]:
        for col_group in np.unique(grid_cols[:,1]):
            for col in grid_cols[grid_cols[:,1]==col_group][:,0][:-1]:
                ishit = 0
                if (np.sum(binary_grid[row:(row+row_height), col:(col+col_width)]) > 0):
                    ishit = 1;
                if( rc[ru == row_group] + gc[gu == col_group]) != 7:
                    grid_grid.append([row, col, row+row_height, col+col_width, ishit, 0])
                if ( rc[ru == row_group] + gc[gu == col_group]) == 7:
                    grid_grid.append([row, col, row+row_height, col+col_width, ishit, 1])

grid_grid = np.array(grid_grid)
pos_grid = grid_grid[grid_grid[:,4] == 1];

# These are all the values that are groups of six
n_six = int(grid_grid[grid_grid[:,5] == 1, 5].shape[0] / 6);
grid_grid[grid_grid[:,5] == 1, 5] = np.repeat(np.arange(1,n_six+1), 6)

colors = np.random.randint(0,255, size=(n_six,3));



rot_imgd = deepcopy(rot_img)
rot_imgd = cv2.cvtColor(rot_imgd,cv2.COLOR_GRAY2RGB)
rot_imgd = draw_grid(rot_imgd, pos_grid)#grid_grid[grid_grid[:,4] == 1])

img1 = deepcopy(rot_img)
img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)

result = cv2.addWeighted(img1, 0.5, rot_imgd, 0.5, 0)
plt.imshow(result);
plt.show()





























## CHCK GRID ROWS ; THIS SHOULD B UP HIGHR !!! IN TH ND !!! ###
## TODO : CHCK IF THR AR OVRLAPS IN TH NW LIN;
## A NW LIN SHOULD NOT HAV A HIGHR VALU THAN TH NXT LIN
## (THR SHOULD NOT B A LIN WITH A HIGHR NUBMR BUT LOWR PX VAL)

clf = load( 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/_GITHUB/HOBraille/svm_dots') 




ugr, cgr = np.unique(grid_rows[:,1],return_counts=True)
rowgroups = ugr[cgr<4];


dimg = deepcopy(rot_img)

for rowg in rowgroups:
    rows   = grid_rows[grid_rows[:,1] == rowg]
    toprow = rows[0]
    botrow = rows[rows.shape[0]-1]
    nrows  = rows.shape[0]
    potrow = [];
    linec  = [];
    
    for row in rows[:,0]:
        cv2.line(dimg, (0, row), (x_px, row), 0, thickness=1)        
    if nrows == 3:
        potrow.append(toprow[0]-row_height);
        potrow.append(botrow[0]);
        linec.append(int(svm_row_check(img, grid_cols, potrow[0])))
        linec.append(int(svm_row_check(img, grid_cols, potrow[1])))
        if linec[0] > 0:
            cv2.line(dimg, (0, potrow[0]), (x_px, potrow[0]), 128, thickness=1)
            grid_rows = np.vstack((grid_rows,[potrow[0],rowg]))
        if linec[1] > 0:
            cv2.line(dimg, (0, potrow[1]+row_height), (x_px, potrow[1]+row_height), 128, thickness=1)
            grid_rows = np.vstack((grid_rows,[potrow[1],rowg]))
    if nrows == 2:
        potrow.append(toprow[0]-row_height+2)
        potrow.append(botrow[0])
        potrow.append(toprow[0]-(row_height*2)+2)
        potrow.append(botrow[0]+row_height)

        linec.append(int(svm_row_check(img, grid_cols, potrow[0])))
        linec.append(int(svm_row_check(img, grid_cols, potrow[1])))
        linec.append(int(svm_row_check(img, grid_cols, potrow[2])))
        linec.append(int(svm_row_check(img, grid_cols, potrow[3])))

        if linec[0] > 0:
            cv2.line(dimg, (0, potrow[0]), (x_px, potrow[0]), 128, thickness=1)
            grid_rows = np.vstack((grid_rows,[potrow[0],rowg]))
        if linec[1] > 0:
            cv2.line(dimg, (0, potrow[1]+row_height), (x_px, potrow[1]+row_height), 128, thickness=1)
            grid_rows = np.vstack((grid_rows,[potrow[1],rowg]))
        if linec[2] > 0:
            cv2.line(dimg, (0, potrow[2]+1), (x_px, potrow[2]+1), 128, thickness=1)
            grid_rows = np.vstack((grid_rows,[potrow[2],rowg]))
        if linec[3] > 0:
            cv2.line(dimg, (0, potrow[3]+row_height*2), (x_px, potrow[3]+row_height*2), 128, thickness=1)
            grid_rows = np.vstack((grid_rows,[potrow[3],rowg]))


plt.imshow(dimg);
plt.show()





## Line Check, never implemented, all lines consinstent this far

ugr, cgr = np.unique(grid_cols[:,1],return_counts=True)
colgroups = ugr[cgr<3];
dimg = deepcopy(rot_img)


for colg in colgroups:
    cols   = grid_cols[grid_cols[:,1] == colg]
    topcol = cols[0]
    botcol = cols[cols.shape[0]-1]
    ncols  = cols.shape[0]
    potcol = [];
    linec  = [];
    
    for col in cols[:,0]:
        cv2.line(dimg, (0, col), (x_px, col), 0, thickness=1)        
    if ncols == 3:
        potcol.append(topcol[0]-col_height);
        potcol.append(botcol[0]);
        linec.append(int(svm_col_check(img, grid_cols, potcol[0])))
        linec.append(int(svm_col_check(img, grid_cols, potcol[1])))
        if linec[0] > 0:
            cv2.line(dimg, (0, potcol[0]), (x_px, potcol[0]), 128, thickness=1)
            grid_cols = np.vstack((grid_cols,[potcol[0],colg]))
        if linec[1] > 0:
            cv2.line(dimg, (0, potcol[1]+col_height), (x_px, potcol[1]+col_height), 128, thickness=1)
            grid_cols = np.vstack((grid_cols,[potcol[1],colg]))
    if ncols == 2:
        potcol.append(topcol[0]-col_height+2)
        potcol.append(botcol[0])
        potcol.append(topcol[0]-(col_height*2)+2)
        potcol.append(botcol[0]+col_height)

        linec.append(int(svm_col_check(img, grid_cols, potcol[0])))
        linec.append(int(svm_col_check(img, grid_cols, potcol[1])))
        linec.append(int(svm_col_check(img, grid_cols, potcol[2])))
        linec.append(int(svm_col_check(img, grid_cols, potcol[3])))

        if linec[0] > 0:
            cv2.line(dimg, (0, potcol[0]), (x_px, potcol[0]), 128, thickness=1)
            grid_cols = np.vstack((grid_cols,[potcol[0],colg]))
        if linec[1] > 0:
            cv2.line(dimg, (0, potcol[1]+col_height), (x_px, potcol[1]+col_height), 128, thickness=1)
            grid_cols = np.vstack((grid_cols,[potcol[1],colg]))
        if linec[2] > 0:
            cv2.line(dimg, (0, potcol[2]+1), (x_px, potcol[2]+1), 128, thickness=1)
            grid_cols = np.vstack((grid_cols,[potcol[2],colg]))
        if linec[3] > 0:
            cv2.line(dimg, (0, potcol[3]+col_height*2), (x_px, potcol[3]+col_height*2), 128, thickness=1)
            grid_cols = np.vstack((grid_cols,[potcol[3],colg]))


plt.imshow(dimg);
plt.show()




## extract grid structure, pretty greedy
grid_grid = [];
grid_rows = np.sort(grid_rows, axis=0)
grid_cols = np.sort(grid_cols, axis=0)
ru, rc = np.unique(grid_rows[:,1], return_counts=True)
gu, gc = np.unique(grid_cols[:,1], return_counts=True)

for row_group in np.unique(grid_rows[:,1]):
    for row in grid_rows[grid_rows[:,1]==row_group][:,0][:-1]:
        for col_group in np.unique(grid_cols[:,1]):
            for col in grid_cols[grid_cols[:,1]==col_group][:,0][:-1]:
                ishit = 0
                if (np.sum(binary_grid[row:(row+row_height), col:(col+col_width)]) > 0):
                    ishit = 1;
                if( rc[ru == row_group] + gc[gu == col_group]) != 7:
                    grid_grid.append([row, col, row+row_height, col+col_width, ishit, 0])
                if ( rc[ru == row_group] + gc[gu == col_group]) == 7:
                    grid_grid.append([row, col, row+row_height, col+col_width, ishit, 1])



grid_grid = np.array(grid_grid)

clf = load( 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/_GITHUB/HOBraille/svm_dots') 

for j, grid_square in enumerate(grid_grid):
    if grid_square[4] == 0:
        img = rot_img[ grid_square[0]:(grid_square[0]+24),grid_square[1]:(grid_square[1]+22)]
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
        lbpi = getLBPimage(img)
        arr = np.append(fd,lbpi.flatten())
        if int(clf.predict(arr.reshape(1,-1))) == 1:
            #plt.imshow(img)
            #plt.show()
            grid_grid[j][4] = 1
            
pos_grid = grid_grid[grid_grid[:,4] == 1];

# These are all the values that are groups of six
n_six = int(grid_grid[grid_grid[:,5] == 1, 5].shape[0] / 6);
grid_grid[grid_grid[:,5] == 1, 5] = np.repeat(np.arange(1,n_six+1), 6)

colors = np.random.randint(0,255, size=(n_six,3));



rot_imgd = deepcopy(rot_img)
rot_imgd = cv2.cvtColor(rot_imgd,cv2.COLOR_GRAY2RGB)
rot_imgd = draw_grid(rot_imgd, pos_grid)#grid_grid[grid_grid[:,4] == 1])

img1 = deepcopy(rot_img)
img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)

result = cv2.addWeighted(img1, 0.5, rot_imgd, 0.5, 0)
plt.imshow(result);
plt.show()




