import os,re

with os.popen("kubectl get pod -o wide") as p:
    res = p.read()
    lines = res.split("\n")
    for i in range(1,len(lines)-1):
        #print(lines[i])
        idx,j = re.search(" ",lines[i]).span()
        print('{0}_s\n'.format(lines[i][0:idx]))
        
#print(lines)
