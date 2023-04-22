import os, shutil, random

# using best.pt to detect
def detect():
    os.system('python detect.py --weights best.pt --source 0 --img 640 --conf 0.25')
    