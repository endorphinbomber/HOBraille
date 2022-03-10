from copy import deepcopy
from xml.etree import cElementTree
import cv2, os, random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.stats import linregress
from math import atan
import time

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

## One Possibility is to use the bounding box by the haar trainer
## as difference between cells. This is size variant, and allows 
## to already approximate the size of cells

start_time = time.time()


## LOAD HAAR CASCAD
HAAR_FIL = 'C:/Users/SebastianG/Desktop/Train_full/cascades/cascade.xml'
SOURCE_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/'
GRID_BORDR = [0,0,0,0];



# LOAD DATA
os.chdir(SOURCE_DIR);
img = cv2.imread('number1.jpg',0)
#img = cv2.imread('number2.jpg',0)
dimg = deepcopy(img)
haar_cascade = cv2.CascadeClassifier(HAAR_FIL)
x_px, y_px = img.shape;

# DTCT TH DOTS, USING HAAR CLASSIFIR
dots = haar_cascade.detectMultiScale(dimg, 1.3, 5)

binary_grid = np.zeros((img.shape))
# draw rectangles
for (x,y,w,h) in dots:
    dimg = cv2.rectangle(dimg,(x,y),(x+w,y+h),(255,0,0),2)
    binary_grid = cv2.circle(binary_grid, center=(x+(w//2),y+(h//2)), radius =1, thickness=2, color=1)

plt.imshow(binary_grid);
plt.show()

# GT BOUNDING BOX OF RCTANGLS
bb = np.mean(dots, axis=0)[2:].astype('int')
row_height, col_width = bb[0], bb[1]


## MAK TH 
# DFIN TOLRANC FOR PXLS
Y_TOL = 5 # 3 px tolerance
X_TOL = 5;

# CLUSTR TH UNIQU LIN POINTS
y_unique = np.unique(dots[:,1]);
X = np.reshape(y_unique, (-1, 1))
ms = MeanShift(bandwidth=Y_TOL, bin_seeding=True)
ms.fit(X)


for label in np.unique(ms.labels_):
    print(y_unique[ms.labels_==label])
    y = int(np.round(np.median(y_unique[ms.labels_==label])));
    cv2.line(dimg, (0, y), (y_px, y),  255, 1)

#plt.imshow(dimg);
#plt.show()






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
dotsr = haar_cascade.detectMultiScale(rot_img, 1.3, 5)




# calculate the clustering once more, look for clusters.
y_unique, y_counts = np.unique(dotsr[:,1], return_counts=True);
X = np.reshape(y_unique, (-1, 1))
ms = MeanShift(bandwidth=Y_TOL, bin_seeding=True)
ms.fit(X)



for label in np.unique(ms.labels_):
    print(y_unique[ms.labels_==label])
    y = int(np.round(np.median(y_unique[ms.labels_==label])));
    cv2.line(rot_imgd, (0, y), (y_px, y),  255, 1)

#plt.imshow(rot_imgd);
#plt.show()


# get center of the rows
row_centers = ms.cluster_centers_.astype('int').flatten()
row_centers.sort()

# get the difference, and find the one that happens
# most often
y_diff = np.diff(row_centers);
#y_diffu, y_f = np.unique(y_diff,return_counts=True);
#row_height = int(y_diffu[y_f == y_f.max()].flatten());

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
        #grid_rows.append(line)

##plt.imshow(rot_imgd,cmap='gray');
#plt.show()







###                                         ###
###  estimate rotation and the line groups  ###
###               for columns               ###
rot_imgd = deepcopy(rot_img)

# Get Dots from rotated image
dotsr = haar_cascade.detectMultiScale(rot_img, 1.3, 5)

# calculate the clustering once more, look for clusters.
x_unique, x_counts = np.unique(dotsr[:,0], return_counts=True);
X = np.reshape(x_unique, (-1, 1))
ms = MeanShift(bandwidth=X_TOL, bin_seeding=True)
ms.fit(X)

column_centers = ms.cluster_centers_.astype('int').flatten();
column_centers.sort();

x_diff = np.diff(column_centers);
#x_diffu, x_f = np.unique(x_diff,return_counts=True);
#col_width = int(x_diffu[x_f == x_f.max()].flatten());

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
        #temp.append(column_centers[cur_pack].max() + col_width)
        cur_label += 1;

grid_cols = np.array(grid_cols);



rot_imgd = deepcopy(rot_img)
for label in np.unique(grid_cols[:,1]):
    columns = grid_cols[grid_cols[:,1]==label][:,0]
    a = random.randint(0,255);
    # GT rAndom ColOR
    for column in columns:
        cv2.line(rot_imgd, (column, 0), (column, x_px),  a, 1)
        #grid_rows.append(line)

#plt.imshow(rot_imgd,cmap='gray');
#plt.show()



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

#plt.imshow(rot_imgd,cmap='gray');
#plt.show()



## extract grid structure, pretty greedy
grid_grid = [];
for row_group in np.unique(grid_rows[:,1]):
    for row in grid_rows[grid_rows[:,1]==row_group][:,0][:-1]:
        
        for col_group in np.unique(grid_cols[:,1]):
            for col in grid_cols[grid_cols[:,1]==col_group][:,0][:-1]:
                ishit = 0
                if (np.sum(binary_grid[row:(row+row_height), col:(col+col_width)]) > 0):
                    ishit = 1;
                grid_grid.append([row, col, row+row_height, col+col_width, ishit])
    
grid_grid = np.array(grid_grid)
pos_grid = grid_grid[grid_grid[:,4] == 1];


rot_imgd = deepcopy(rot_img)
rot_imgd = cv2.cvtColor(rot_imgd,cv2.COLOR_GRAY2RGB)
for square in grid_grid:
    boxy = np.ones(rot_imgd[square[0]:square[2],square[1]:square[3]].shape)
    boxy[:,:,0] = boxy[:,:,0] * random.randint(0,255);
    boxy[:,:,1] = boxy[:,:,1] * random.randint(0,255);
    boxy[:,:,2] = boxy[:,:,2] * random.randint(0,255);
    rot_imgd[square[0]:square[2],square[1]:square[3]] = boxy;

img1 = deepcopy(rot_img)
img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
result = cv2.addWeighted(img1, 0.5, rot_imgd, 0.5, 0)
plt.imshow(result);
plt.show()



















## LOOK AT POSITIVS


rot_imgd = deepcopy(rot_img)
rot_imgd = cv2.cvtColor(rot_imgd,cv2.COLOR_GRAY2RGB)
var_grid = []

for square in pos_grid:
    boxy = np.ones(rot_imgd[square[0]:square[2],square[1]:square[3]].shape)
    boxy[:,:,0] = boxy[:,:,0] * random.randint(0,255);
    boxy[:,:,1] = boxy[:,:,1] * random.randint(0,255);
    boxy[:,:,2] = boxy[:,:,2] * random.randint(0,255);
    rot_imgd[square[0]:square[2],square[1]:square[3]] = boxy;
    var_grid.append(rot_img[square[0]:square[2],square[1]:square[3]])

var_grid = np.array(var_grid);
thealpha = np.mean(var_grid,axis=0).astype('int');





rot_imgd = deepcopy(rot_img)
rot_imgd = cv2.cvtColor(rot_imgd,cv2.COLOR_GRAY2RGB)

from sklearn.metrics import mutual_info_score
score_hist = [];

for square in grid_grid:
    a = rot_img[square[0]:square[2],square[1]:square[3]].flatten();
    b = thealpha.flatten();
    score = mutual_info_score(a,b)
    score_hist.append(score);
    if (score>0.03):
        boxy = np.ones(rot_imgd[square[0]:square[2],square[1]:square[3]].shape)
        boxy[:,:,0] = boxy[:,:,0] * random.randint(0,255);
        boxy[:,:,1] = boxy[:,:,1] * random.randint(0,255);
        boxy[:,:,2] = boxy[:,:,2] * random.randint(0,255);
        rot_imgd[square[0]:square[2],square[1]:square[3]] = boxy;

np.mean(score_hist)

img1 = deepcopy(rot_img)
img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
result = cv2.addWeighted(img1, 0.5, rot_imgd, 0.5, 0)
plt.imshow(result);
plt.show()