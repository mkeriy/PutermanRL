import numpy as np

class Replay_buffer(object):
    default_elem = ('state', 'next_state', 'action', 'reward', 'done')
    
    def __init__(self, max_size=1000000, top_perc=1.0):
        self.dt = None
        self.storage = None
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.sortedinds = None
        self.is_sorted = False
        self.top_perc = top_perc
        
    def push(self, data):
        self.is_sorted = False

        if self.dt is None:
            s_shape, a_shape = data[0].shape, data[2].shape
            self.dt = np.dtype([('state', np.float64, s_shape), ('next_state', np.float64, s_shape), ('action', np.float32, a_shape), ('reward', np.float64, (1,)), ('done', np.bool_, (1,))])
            self.storage = np.empty(shape=(self.max_size,),dtype=self.dt)
            
        data = np.array([data], dtype=self.dt)
        
        self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size != self.max_size: self.size += 1

    def sample_by_inds(self, inds, batch_size=None, elem=default_elem):
        if batch_size is not None and len(inds) > batch_size:
            inds = np.random.choice(inds, batch_size)
        sampled = self.storage[inds]
        return [np.array(sampled[i]) for i in elem]    
    
    def sample(self, batch_size, elem=default_elem):
        return self.sample_by_inds(
                np.random.randint(0, self.size, size=batch_size), batch_size, elem)

    def sample_r_sorted(self, batch_size, elem=default_elem):
        if self.top_perc == 1.0: return self.sample(batch_size, elem)
        if not self.is_sorted: self.sort_r()
        cur_sorted_inds = self.sortedinds[:max(batch_size, int(self.size*(1 - self.top_perc)))]
        return self.sample_by_inds(cur_sorted_inds, batch_size, elem)
    
    def sample_last(self, batch_size, last = None, elem = default_elem):
        return self.sample_by_inds(np.arange(max(-self.size, self.ptr - last), self.ptr), batch_size, elem)

    def sort_r(self):
        self.sortedinds = np.argpartition(self.storage[:self.size]['reward'].flatten(), int(self.size*(1 - self.top_perc)))[::-1]
        self.is_sorted = True