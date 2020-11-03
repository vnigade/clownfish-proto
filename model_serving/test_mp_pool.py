import multiprocessing
from multiprocessing import Pool
import numpy as np
import time


def tester(num):
    return np.cos(num)


if __name__ == '__main__':

    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size,
                                )
    starttime1 = time.time()
    pool_outputs = pool.map(tester, range(50000000))
    pool.close()
    pool.join()
    endtime1 = time.time()
    timetaken = endtime1 - starttime1

    starttime2 = time.time()
    for i in range(50000000):
        tester(i)
    endtime2 = time.time()
    timetaken2 = endtime2 - starttime2

    print('The time taken with multiple processes:', timetaken)
    print('The time taken the usual way:', timetaken2)
