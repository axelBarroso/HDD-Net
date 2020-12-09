from skimage import io
from os import path, mkdir

def check_directory(dir):
    if not path.isdir(dir):
        mkdir(dir)

def create_result_dir(path):
    directories = path.split('/')
    tmp = ''
    for idx, dir in enumerate(directories):
        tmp += (dir + '/')
        if idx == len(directories)-1:
            continue
        check_directory(tmp)

def read_bw_image(path):
    img = io.imread(path, as_gray=True)
    return img