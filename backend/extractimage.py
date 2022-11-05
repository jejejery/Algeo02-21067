import os
import cv2

# folder path
dir_path = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\train\\5\\RGB'
dir_des = r'C:\\Users\\ASUS\\Documents\\Programming\\Python\\Algeo02-21067\\dataset\\train\\5\\BnW'

# list file and directories
res = os.listdir(dir_path)
dim = (256, 256)

for k in res:
    image = cv2.imread(dir_path+'\\'+k)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    k = dir_des + '\\' + k
    cv2.imwrite(k, resized)