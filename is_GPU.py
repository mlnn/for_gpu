import os
os.environ['THEANO_FLAGS']="device=gpu1,lib.cnmem=0.5"
path=os.path.expanduser('~\.theanorc')

from theano import function, config, shared, sandbox
#import theano.sandbox.cuda
#theano.sandbox.cuda.use("gpu1")
import theano.tensor as T

import time
import numpy


#print(path)

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 10000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')