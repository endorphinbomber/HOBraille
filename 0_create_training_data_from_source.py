from copy import deepcopy
from xml.etree import cElementTree
import cv2, os, random, string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.stats import linregress
from math import atan


DATA_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/data';
RSULT_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/TRUE/';
os.chdir(DATA_DIR)

try:
    os.mkdir(RSULT_DIR)
except:
    print('folder exists')


yn = 2;
xn = 3;
letters = string.ascii_lowercase
height = 22;
width = 22;


folders = subfolders = [ f.path for f in os.scandir(DATA_DIR) if f.is_dir() ];

for folder in folders:
    os.chdir(folder)
    files = os.listdir('.');
    pfiles =  [ x for x in files if "recto"  in x ];
    pfiles =  [ x for x in pfiles if ".jpg" not in x ];



    for cpic in pfiles:
        with open(cpic, encoding="utf8") as file:
            lines = file.readlines()

        rows = np.fromstring(lines[1][:-1], dtype=int, sep= " ");
        cols = np.fromstring(lines[2][:-1], dtype=int, sep= " ");
        
        #width = cols[1] - cols[0];
        
        boxes =  [ x[:-1] for x in lines[3:] ];
        
        cimg = cpic[:-3]+'jpg';
        img = cv2.imread(cimg)       

        if os.path.isfile(cpic[:-3]+'jpg'):


            for box in boxes:
                box = np.fromstring(box, dtype=int, sep= " ");
                xref = rows[box[1]*yn-yn]  - (width//2);
                yref = cols[box[0]*xn-xn]  - (width//2);

                if box[2] == 1:
                    fname = ''.join(random.choice(letters) for i in range(10));
                    cb = img[yref:(yref+height),xref:(xref+width),]
                    cv2.imwrite(RSULT_DIR+fname+cpic[:-3]+'1.jpg',cb)
                    
                if box[3] == 1:
                    cb = img[(yref+height):(yref+(height*2)),xref:(xref+width),]
                    cv2.imwrite(RSULT_DIR+fname+cpic[:-3]+'2.jpg',cb)
                    
                if box[4] == 1:
                    cb = img[(yref+height*2):(yref+(height*3)),xref:(xref+width),]
                    cv2.imwrite(RSULT_DIR+fname+cpic[:-3]+'3.jpg',cb)
                
                
                if box[5] == 1:
                    cb = img[yref:(yref+height),(xref+width):(xref+width*2),]
                    cv2.imwrite(RSULT_DIR+fname+cpic[:-3]+'4.jpg',cb)
                    
                if box[6] == 1:
                    cb = img[(yref+height):(yref+(height*2)),(xref+width):(xref+width*2),]
                    cv2.imwrite(RSULT_DIR+fname+cpic[:-3]+'5.jpg',cb)
                    
                if box[7] == 1:
                    cb = img[(yref+height*2):(yref+(height*3)),(xref+width):(xref+width*2),]
                    cv2.imwrite(RSULT_DIR+fname+cpic[:-3]+'6.jpg',cb)
                
        
            



ENDRES = 30;
border = ENDRES/2;
n_centers = 9;
image_cur = 0;
START_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/FALSE_SOURCES/';
RSULTG_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/neg/';

os.chdir(START_DIR);
files = os.listdir(START_DIR);

random.shuffle(files)

try:
    os.mkdir(RSULTG_DIR)
except:
    print('folder exists')


for file in files:
    cur_file = START_DIR+'/'+file
    cur_img = cv2.imread(cur_file,0)
    seg_center = np.zeros((n_centers,2))
    seg_center[:,0] = np.array([random.randint(0, cur_img.shape[1]) for p in range(0, n_centers)])
    seg_center[:,1] =  np.array([random.randint(0, cur_img.shape[0]) for p in range(0, n_centers)])

    for dot in seg_center:    
        coords_crop =  int(dot[1]-border),  int(dot[1]+border),  int(dot[0]-border),  int(dot[0]+border);
        res = any(ele < 0 for ele in coords_crop)

        if res == False:
            crop = cur_img[coords_crop[0]:coords_crop[1],coords_crop[2]:coords_crop[3]]
            crop = cv2.resize(crop,(ENDRES,ENDRES))
            
            fname = ''.join(random.choice(letters) for i in range(10));
            cv2.imwrite(RSULTG_DIR+str(image_cur)+fname+ '.jpg',crop)
            image_cur += 1





## Convert Gray
RSULT_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/TRUE/';
RSULTG_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/pos/';
try:
    os.mkdir(RSULTG_DIR)
except:
    print('folder exists')

os.chdir(RSULT_DIR)
files = os.listdir('.');
img_cur = 0;
for i,file in enumerate(files):
    cimg = cv2.imread(file,0)
    cv2.imwrite(RSULTG_DIR+str(img_cur)+'.jpg',cimg)
    img_cur += 1;
    
    
    
## A good idea might be to search for the mean image,
## and afterwards use the mean image to find images similar to it
## this is usefull for mean centering, and eliminating outliers
## which in turns makes the haar classifier better centered,
## but at the same time less acceptable of variance in the images
RSULT_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/TRUE/';
RSULTG_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/pos_cc/';

try:
    os.mkdir(RSULTG_DIR)
except:
    print('folder exists')

os.chdir(RSULT_DIR)
files = os.listdir('.');
img_cur = 0;


for i,file in enumerate(files):
    cimg = cv2.imread(file,0)
    cv2.imwrite(RSULTG_DIR+str(img_cur)+'.jpg',cimg)
    img_cur += 1;
    
    
    
    
    
    
## Create text files for haar classification
## 

folder_files = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_FINAL/' 
os.chdir(folder_files);

def create_pos_n_neg():
 
    fpos = open('pos.lst','w')
    fneg = open('neg.lst','w')
 
    for file_type in ['neg', 'pos']:
         
        for img in os.listdir(file_type):
 
            if file_type == 'pos':
                try:
                    image = cv2.imread(file_type+'/'+img)
                    h, w, channels = image.shape
                    line = file_type+'/'+img+' 1 0 0 '+str(w)+' '+str(h)+'\n'
                    fpos.write(line)
                except:
                    continue
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                fneg.write(line)
 
    fpos.close()
    fneg.close()
 
create_pos_n_neg()
