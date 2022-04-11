import cv2, os
base_path = "C:/Users/SebastianG/Pictures/scans/"
new_path = "C:/Users/SebastianG/Pictures/scansout/"
for infile in os.listdir(base_path):
    print ("file : " + infile)
    read = cv2.imread(base_path + infile)
    
    scale_percent = 10 # percent of original size
    width = int(read.shape[1] * scale_percent / 100)
    height = int(read.shape[0] * scale_percent / 100)
    dim = (width, height)

    readr = cv2.resize(read, dim);
    outfile = infile.split('.')[0] + '.jpg'
    cv2.imwrite(new_path+outfile,readr,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
