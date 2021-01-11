import os
def rename():
    path = './datasets/out/test_depth'
    filelist = os.listdir(path)
    for files in filelist:
        Olddir = os.path.join(path,files)
        filename = os.path.splitext(files)[0]
        filename = filename.replace('depth','')
        filetype = os.path.splitext(files)[1]
        Newdir = os.path.join(path,filename+filetype)
        os.rename(Olddir,Newdir)

rename()