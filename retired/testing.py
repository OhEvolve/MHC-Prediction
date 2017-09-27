
"""
This is a building ground for new ideas and optimization 
NOT FOR RUNNING REAL SIMULATION
"""


'''

from models import * # libraries: kNN,cNN,spiNN

for i in xrange(1,5):
    params = {
             'sw_depth':2,
             'pw_depth':i,
             'silent':True,
             }


    m = spiNN.BuildModel(params)
    m.fold_data()
    m.fold_pick(0)
    m.network_initialization()


    print 'SW depth:',m.sw_depth
    print 'PW depth:',m.pw_depth

    print 'Parameters:',m.count_trainable_parameters()

#'''

from multiprocessing.dummy import Pool as ThreadPool 
import time

def my_function(val):
    for i in xrange(400000):
        val += 1
    return val


my_array = [10*i for i in xrange(32)]


for threads in [1,2,4,8,16]:
    start = time.time()
    pool = ThreadPool(threads) 
    results = pool.map(my_function, my_array)
    pool.close()
    pool.join()
    
    print 'Thread count: {} -> Elapsed time: {}'.format(threads,time.time() - start)
