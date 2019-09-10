import os, shutil

# First, create a list and populate it with the files
# you want to find (1 file per row in myfiles.txt)
files_to_find = []
src=r'Y:\DICOMEXPORT_H&N'
dst=r'Y:\DICOMEXPORT_H&N_100'
dirList = []


def dir_list_folder(head_dir, dir_name):
    """Return a list of the full paths of the subdirectories
    under directory 'head_dir' named 'dir_name'"""
    dirList = []
    for fn in os.listdir(head_dir):
        dirfile = os.path.join(head_dir, fn)
        if os.path.isdir(dirfile):
            if fn.upper() == dir_name.upper():
                dirList.append(dirfile)
            else:
                # print "Accessing directory %s" % dirfile
                dirList += dir_list_folder(dirfile, dir_name)
    return dirList
 
if __name__ == '__main__':
    symlinks=False
    ignore=None
    with open(r'C:\patientID.txt') as fh:
        for row in fh:
            files_to_find.append(row.strip)
            #dir_name=row.strip
            item=row[0:-1]
            s=os.path.join(src,item)

            if os.path.isdir(s)==True:
                print('yes')
                #shutil.copytree(foldername, dstSrcDir)
                #print('1 copied')

                d = os.path.join(dst, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, symlinks, ignore)
                else:
                    shutil.copy2(s, d)
