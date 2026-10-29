# %%
import matplotlib.pyplot as plt
import numpy as np


def y(x:np.ndarray):
    return x*np.sin(4*np.pi*x)+2
# %%
class SFactory:
    def __init__(self, min, max, n, size=50):
        self.min = min
        self.max = max

        self.n = n
        self.bit_width = 1
        while 1<<self.bit_width < n:
            self.bit_width += 1
        self.mask = (1<<self.bit_width)-1
        self.k = (max-min) / self.mask

        self.data = np.random.randint(n, size=size)
        self.size = size

    def decode(self):
        return self.min+self.k*self.data

    def y(self):
        return y(self.decode())

    def select(self):
        """轮盘赌"""
        score = self.y()
        self.data = self.data[((score/score.sum()).cumsum()[:, np.newaxis] < np.random.random((1, self.size))).sum(axis=0)]

    def exchange(self, p=.9):
        index = np.random.permutation(np.arange(10)).reshape(2,-1)[:,np.random.random(5) < p]
        mask = self.mask >> np.random.randint(1, self.bit_width, size=index.shape[1])

        self.data[index[0]] ^= self.data[index[1]] & mask
        self.data[index[1]] ^= self.data[index[0]] & mask
        self.data[index[0]] ^= self.data[index[1]] & mask

        # temp = self.data[index[0]] & mask
        # self.data[index[0]] &= ~mask
        # self.data[index[0]] |= self.data[index[1]] & mask
        # self.data[index[1]] &= ~mask
        # self.data[index[1]] |= temp


    def mutate(self, p=.05):
        self.data ^= np.where(
            np.random.random(self.size) < p,
            1<<np.random.randint(self.bit_width, size=self.size),
            np.int8(0)
        )

fac = SFactory(-1, 2, 3_000_000)
# fac.decode()
# fac.decode(2**22-1)

# fac.select()
# fac.mutate()
fac.exchange()
# %%
fac = SFactory(-1, 2, 3_000_000)

fig, (ax0, ax1) = plt.subplots(ncols=2)
results = np.zeros((100, fac.size))
for each in results:
    each += fac.decode()
    fac.select()
    fac.exchange()
    fac.mutate()

x = np.arange(100)

ax0.plot(x, y(results).max(axis=1))
ax0.plot(x, y(results).mean(axis=1))

ax1.plot(x, results[*np.vstack((x,y(results).argmax(axis=1)))])

plt.show()