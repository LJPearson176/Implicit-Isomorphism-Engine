import random
import copy

class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = self._get_shape(data)

    def _get_shape(self, data):
        if isinstance(data, list):
            if not data:
                return (0,)
            if isinstance(data[0], list):
                return (len(data), len(data[0]))
            return (len(data),)
        return ()

    def size(self, dim=None):
        if dim is None:
            return self.shape
        if dim < len(self.shape):
            return self.shape[dim]
        raise IndexError("Dimension out of range")

    def clone(self):
        return Tensor(copy.deepcopy(self.data))

    def __getitem__(self, key):
        # Support [:, mask]
        # key is a tuple (slice(None), Tensor(mask))
        if isinstance(key, tuple):
            if len(key) == 2:
                # check for slice(None) which is :
                if isinstance(key[0], slice) and key[0].start is None and key[0].stop is None:
                    if isinstance(key[1], Tensor):
                        mask = key[1].data
                        # self.data is [[...], [...]]
                        new_data = []
                        for row in self.data:
                            new_row = [x for x, m in zip(row, mask) if m]
                            new_data.append(new_row)
                        return Tensor(new_data)
        raise IndexError(f"Unsupported index type: {key}")

    def sum(self):
        # For boolean tensor (list of bools), sum is count of True
        return Tensor(sum(self.data))

    def item(self):
        return self.data

    def __gt__(self, other):
        # vector > scalar
        return Tensor([x > other for x in self.data])

    def __repr__(self):
        return f"Tensor({self.data})"

def tensor(data):
    return Tensor(data)

def rand(size):
    # size is int
    return Tensor([random.random() for _ in range(size)])

def randint(low, high, size):
    # size is tuple (rows, cols)
    rows, cols = size
    data = [[random.randint(low, high-1) for _ in range(cols)] for _ in range(rows)]
    return Tensor(data)

def manual_seed(seed):
    random.seed(seed)

def equal(t1, t2):
    return t1.data == t2.data
