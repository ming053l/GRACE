import os
import glob
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# celeb-df

pth='/hdd1/DeepFakes/celeb-df-video/Celeb-synthesis/'
outdir = '/hdd1/DeepFakes_may/celeb-df/Celeb-synthesis/'
# pth='/hdd1/DeepFakes/celeb-df-video/Celeb-real/'
base_dir = '/ssd5/DeepFakes_may/celeb-df/'
outdir = 'Celeb-synthesis/'

dx = glob.glob(pth+'*.mp4')
train_data, test = train_test_split(dx, random_state=777, train_size=0.8)
val_data, test_data = train_test_split(test, random_state=777, train_size=0.5)
print(len(train_data))
print(len(val_data))
print(len(test_data))
data_name = ['train', 'val', 'test']
data_use = [train_data, val_data, test_data]

for num, dx in enumerate(data_use):
    for filename in dx:
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        print(filename,v_len)
        frame_interval = 1
        path = '/ssd6/DeepFakes_may/celeb-df/{}.txt'.format(data_name[num])

        # if v_len >600:
        #     v_len = 600

        fn = os.path.basename(filename)[:-4]
        count = 0
        if not os.path.exists(base_dir + data_name[num]+'/'+outdir+'/'+fn):
            os.makedirs(base_dir + data_name[num]+'/'+outdir+'/'+fn)

        with open(path, 'a') as f:
            f.write(os.path.join(data_name[num], outdir, fn)+' 1'+'\n')
        
        path_ =  os.path.join(base_dir, data_name[num], outdir, fn, 'list.txt')

        for j in range(v_len):
            # Load frame
            success, frame = v_cap.read()

            if not success:
                continue
            if j % frame_interval==0:
                cv2.imwrite(os.path.join(base_dir, data_name[num], outdir, fn, '%03d.png' % (count)),frame)
                with open(path_, 'a') as f:
                        f.write('%03d.png' % (count)+'\n')
                
                count+=1
                # break
f.close()