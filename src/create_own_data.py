import sys
sys.path.insert(0, '../tool')
import create_dataset as create

w=open('/root/train.txt')
w1=open('/root/train_label.txt')
image_list = []
label_list = []

for ele in w.readlines():
  image_list.append(ele.split('\n')[0])
for ele in w1.readlines():
  label_list.append(ele.split('\n')[0])

create.createDataset('./lmdb_syn90_train',image_list,label_list)

