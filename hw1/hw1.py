import numpy as np
import csv
import sys

w = np.load('model.npy')

f = open(sys.argv[1], 'r', encoding='big5')
lines = csv.reader(f, delimiter=",")    
    
test_data = []
count = 0
tmp = []
tmp.append( float(1) )

for line in lines:
    if count < 18:
        for i in range(2, 11):
            if line[i] == "NR":
                tmp.append( float(0) )
            else:
                tmp.append( float(line[i]) )
        count += 1
    if count == 18:
        count = 0
        test_data.append(tmp)
        tmp = []
        tmp.append( float(1) ) 

        
test_data = np.array(test_data)
#print(test_data.shape[0], test_data.shape[1], len(test_data[0]))
    
num = test_data.shape[0]    
   
f_out = open(sys.argv[2], 'w', newline='',encoding='big5')
writer = csv.writer(f_out)
ans = []
 
#print('id,value')
for i in range(num):
    v = np.dot(test_data[i], w)
    #print('id_',i,',',v, sep = '')
    ans.append(['id_'+str(i)])
    ans[i].append(v)

writer.writerow(['id','value'])
for i in range(num):
    writer.writerow(ans[i])
f_out.close()
    
    