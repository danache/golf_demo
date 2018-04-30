import multiprocessing as mp
import numpy as np

from .serialize import loads, dumps

# reduce_method
LIST = 0
ITEM = 1
ITEMS = 2
ITEMSLIST = 3

# func type
FUNC = 0
# ITER = 1

class Worker(mp.Process):
    def __init__(self, id, queue, func, func_type, *args, **kwargs):
        super(Worker, self).__init__()
        self.id = id
        self._func = func
        self._queue = queue
        self.args = args
        self.kwargs = kwargs
        self._func_type = func_type

    def run(self):
        if self._func_type == FUNC:
            msg = self._func(self.id, *self.args, **self.kwargs)
            self._queue.put( dumps([self.id, msg]) )
        # elif self._func_type == ITER:
        #     for i, msg in enumerate(func(self.id, *self.args, **self.kwargs)):
        #         self._queue.put([self.id, i, dumps(msg)])
        else:
            raise ValueError('Invalid func type.')

class MultiProc(object):
    def __init__(self, nr_proc, func, func_type=FUNC, reduce_method=ITEMSLIST, *args, **kwargs):
        self._queue = mp.Queue()
        self.nr_proc = nr_proc
        self._proc_ids = [i for i in range(self.nr_proc)]
        self._func_type = func_type
        self._reduce_method = reduce_method

        self._procs = []
        for i in range(self.nr_proc):
            w = Worker(self._proc_ids[i], self._queue, func, func_type, *args, **kwargs)
            w.deamon = True
            self._procs.append( w )

    def work(self):
        for p in self._procs:
            p.start()

        ret = [[] for i in range(self.nr_proc)]
        for i in range(self.nr_proc):
            id, msg = loads( self._queue.get(block=True, timeout=None) )
            ret[id] = msg

        for p in self._procs:
            p.join()

        result = []
        if self._reduce_method == LIST:
            for i in range(len(ret)):
                result.extend(ret[i])
        elif self._reduce_method == ITEM:
            result = ret
        elif self._reduce_method == ITEMS:
            for i in range(len(ret[0])):
                result.append( [ret[j][i] for j in range(len(ret))] )
        elif self._reduce_method == ITEMSLIST:
            for i in range(len(ret[0])):
                tmp_res = []
                for j in range(len(ret)):
                    tmp_res.extend(ret[j][i])
                result.append( tmp_res )
        else:
            raise ValueError('Invalid reduce method.')

        return result

if __name__ == '__main__':
    test_ranges = [0, 100, 200, 300, 400, 500]
    def test_net(id):
        test_range = [test_ranges[id], test_ranges[id+1]]
        x = []
        for i in range(*test_range):
            x.append(np.ones((10, 10)) * i)
        print('finish {}'.format(id))
        return x

    x = MultiProc(5, test_net, reduce_method=LIST)
    res = x.work()
    from IPython import embed; embed()
