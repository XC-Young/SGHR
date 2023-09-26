import pickle
import numpy as np

# rewrite to txt
def rewrite():
  f = open('/main/00_MINE/SGHR/train/pkls/test_whu.pkl','rb')
  data = pickle.load(f)
  output = open('1.txt','a')
  print(data,file=output)

# read
def read():
  with open('/main/00_MINE/SGHR/train/pkls/test_whu.pkl','rb') as f:
    d_list = pickle.load(f)
    print(len(d_list))
    for i in range(len(d_list)):
      d_tuple = d_list[i]
      print(len(d_tuple))
      print(d_tuple[0])
      print(d_tuple[1].shape)

# save WHU-TLS
def save_whu():
  scenes=['Park','Mountain','Campus','RiverBank','UndergroundExcavation','Tunnel']
  stationnums=[32,6,10,7,12,7]
  d_list = []
  for i in range(len(scenes)):
    tup = (scenes[i],np.zeros((stationnums[i],stationnums[i])))
    d_list.append(tup)
  pkl = open('./train/pkls/test_whu.pkl','wb')
  pickle.dump(d_list,pkl)
  pkl.close

# rewrite()
# read()
# save_whu()