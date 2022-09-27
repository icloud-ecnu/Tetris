import os
import pandas as pd
import subprocess
def getPodNodNum():
    with os.popen("kubectl get node -o wide") as po:
        res = po.read()
        df = 0
        lines = res.split("\n")
    print(len(lines))
    for i in lines:
        print(i)
def sub():
    res = subprocess.Popen("kubectl get node -o wide",shell=True)
    # lines = res.split("\n")
    #print(type(res))
getPodNodNum()

# sub()