import numpy as np
import time
import collections

N = 100000
np.random.seed(0)

p_list = list(np.random.uniform(0, 0.3, int(N / 5))) + list(np.random.uniform(0.01, 0.1, int(N / 10))) + \
         list(np.random.uniform(0.5, 0.6, int(N / 2))) + list(np.random.uniform(0.3, 0.35, int(N / 5)))
# print(p_list)

# reset random seed
t = 1000 * time.time()  # current time in milliseconds
np.random.seed(int(t) % 2 ** 32)

sample_list = []

for i in range(N):
    p = [p_list[i], 1 - p_list[i]]
    sample = np.random.choice(a=[0, 1], size=1, p=p)[0]
    sample_list.append(sample)

frequency = collections.Counter(sample_list)
print(dict(frequency))
print(np.mean(p_list))