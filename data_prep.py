import numpy as np
import cv2, os


def cut_off_2px_bottom(STRT_DIR, ND_DIR):
    
    try:
        os.mkdir(ND_DIR)
    except:
        print('folder exists')

    os.chdir(STRT_DIR)
    files = os.listdir('.')
    
    for file in files:
        cur_file = STRT_DIR+'/'+file
        cur_img = cv2.imread(cur_file,0)
        cv2.imwrite(ND_DIR+file,cur_img[:22,:])












STRT_DIR = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_ULTIMATE_MOFO/SVM/pos_train'
ND_DIR   = 'C:/Users/SebastianG/Nextcloud/_SEBASTIAN/Forschung/Braille/_GOOD_PIPELINES/CREATE_ULTIMATE_MOFO/SVM/pos_train-crop/'
cut_off_2px_bottom(STRT_DIR,ND_DIR)