import os

def CheckSavePath(dir_path):
    if not os.path.isdir(dir_path):
        try:
            os.makedirs(dir_path)
        except Exception as ex:
            print("Create dir error: {}".format(ex))
    return dir_path

def GetDataPath(data_dir, data_fmt=[".png", ".bmp", ".jpg", ".JPG", ".JPEG"]):
    """
    data_dir(str): images directory path
    data_fmt(list): the data format want to get. default: [".png", ".bmp", ".jpg", ".JPG", ".JPEG"]
    """
    data_path = list()
    for dirPath, dirNames, fileNames in os.walk(data_dir):
        for f in fileNames:
            if os.path.splitext(f)[1] in data_fmt:
                data_path.append(os.path.join(dirPath, f))
    return data_path