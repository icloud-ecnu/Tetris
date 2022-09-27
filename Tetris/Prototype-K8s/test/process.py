from multiprocessing import Pool,Manager,Process
import time

a = Manager().dict()
def func(i):
    a[i]=i*i
    return  i*i
 

if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    
    results = []
    length = 5000
    for i in range(8):
        pool = Process(target=func,args=(i,))
        results.append(pool)
    for pool in results:
        pool.start()
    for pool in results:
        pool.join()
    print ("Sub-process(es) done.")
    print(a)
