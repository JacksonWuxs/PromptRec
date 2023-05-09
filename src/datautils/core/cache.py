import time
import random
import collections


class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.

    """

    __slots__ = 'hashvalue'

    def __init__(self, tup):
        self[:] = tup
        self.hashvalue = None

    def __hash__(self):
        if self.hashvalue is None:
            flatten = []
            for val in self:
                if not isinstance(val, collections.Hashable):
                    val = _HashedSeq(val)
                flatten.append(val)
            self.hashvalue = hash(tuple(flatten))
        return self.hashvalue


def _make_key(args, kwds, 
             kwd_mark = (object(),),
             fasttypes = {int, str}):
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)


class HashCache:
    def __init__(self, func, size, order):
        assert callable(func)
        assert isinstance(size, int) or size is None
        self._func = func
        self._storage = {}
        self._score = []
        self._mapping = []
        self._size = size if size else float('inf')
        self._order = {"time": 0, "freq": 1, "random": 2}[order]
        self._timer = time.time
        self._sampler = random.randint

    def collect(self, *args, **kwrds):
        key = _make_key(args, kwrds)
        if key in self._storage:
            rslt, idx = self._storage[key]
            if self._order:
                self._score[idx] += 1
            else:
                self._score[idx] = self._timer()
            return rslt

        rslt = self._func(*args, **kwrds)
        init = 0.0 if self._order else self._timer()
        if len(self._storage) < self._size:
            self._storage[key] = (rslt, len(self._score))
            self._score.append(init)
            self._mapping.append(key)

        else:
            if self._order == 2:
                idx = min(enumerate(a), key=lambda pair: pair[1])[0]
            else:
                idx = self._sampler(0, len(self._score))
            del self._storage[self._score[idx][0]]
            self._score[idx] = init
            self._mapping[idx] = key
        return rslt


if __name__ == "__main__":
    def func(x):
        return x + 1
    c = HashCache(func, 5, "time")
