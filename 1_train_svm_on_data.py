import os, cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

def getLBPimage(gray_image):
    '''
    == Input ==
    gray_image  : color image of shape (height, width)
    
    == Output ==  
    imgLBP : LBP converted image of the same shape as 
    '''
    
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


def get_feature_length(img_location):
    img = cv2.imread(img_location,0)
    img = cv2.resize(img,(xs,ys))
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    lbpi = getLBPimage(img)
    lfd = int(fd.shape[0]);
    lbp = int(lbpi.flatten().shape[0]);
    return lfd+lbp




## Set the main properties ##
SVM_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/SVM/'
xs, ys = 22, 22;







##                              ##
##   Get the data properties    ##
##                              ##

POS_DIR = SVM_DIR+'train_pos/';
NEG_DIR = SVM_DIR+'train_pos/';

pos_files = os.listdir(POS_DIR)
neg_files = os.listdir(NEG_DIR)

feature_length = get_feature_length(POS_DIR+pos_files[0])



##                              ##
##      Prepare the data        ##
##                              ##

X_train = np.zeros((len(pos_files)+len(neg_files),feature_length))
y_train = np.zeros((len(pos_files)+len(neg_files)))

print("Train_Pos")
os.chdir(POS_DIR)
for ind,file in enumerate(pos_files):
    img = cv2.imread(file,0)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    lbpi = getLBPimage(img)
    arr = np.append(fd,lbpi.flatten())
    X_train[ind,:] = arr;
    y_train[ind] = 1;

print("Train_Neg")
os.chdir(NEG_DIR)
for indn,file in enumerate(neg_files):
    ind = indn + len(pos_files)
    
    img = cv2.imread(file,0)
    img = cv2.resize(img,(xs,ys))
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    lbpi = getLBPimage(img)
    arr = np.append(fd,lbpi.flatten())
    X_train[ind,:] = arr
    y_train[ind] = 0




##                              ##
##     Train the algorihtm      ##
##                              ##

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

print("Train SVM")
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)


from joblib import dump
dump(clf, SVM_DIR+'svm_dots') 






##                                      ##
##   Create Test Data and see how       ## 
##   well the alogrithm performs        ##
##                                      ##

POS_DIR = SVM_DIR+'test_pos/';
NEG_DIR = SVM_DIR+'test_neg/';

pos_files = os.listdir(POS_DIR)
neg_files = os.listdir(NEG_DIR)

X_test = np.zeros((len(pos_files)+len(neg_files),feature_length))
y_test = np.zeros((len(pos_files)+len(neg_files)))

os.chdir(POS_DIR)
for ind,file in enumerate(pos_files):
    img = cv2.imread(file,0)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    lbpi = getLBPimage(img)
    arr = np.append(fd,lbpi.flatten())
    X_test[ind,:,] = arr;
    y_test[ind] = 1

os.chdir(NEG_DIR)
for indn,file in enumerate(neg_files):
    ind = indn + len(pos_files)
    
    img = cv2.imread(file,0)
    img = cv2.resize(img,(xs,ys))
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)
    lbpi = getLBPimage(img)
    arr = np.append(fd,lbpi.flatten())
    X_test[ind,:,] = arr;
    y_test[ind] = 0

import time
start = time.time()
Y_pred = clf.predict(X_test)
end = time.time()
print("Time is :"+str(end - start)+'for '+str(len(Y_pred))+' samples.')


from joblib import dump
dump(clf, SVM_DIR+'svm_dots') 






