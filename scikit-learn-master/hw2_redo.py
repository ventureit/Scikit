#Hoeffding Inequality
import numpy as np
import matplotlib.pyplot as plt

ncoins = 10
nflips = 1000
iterations = 100000
flip_lambda = lambda ncoins, p, nflips: np.random.binomial(ncoins, .5, size=nflips)
flips = flip_lambda(ncoins, .5, nflips)
# print(flips)
v_1, v_rand, v_min = [np.zeros(iterations), np.zeros(iterations), np.zeros(iterations)]
# print(c_1)

for times in range(iterations):
    c_1 = flips[0]
    c_rand = flips[np.random.randint(1, ncoins)]
    c_min = flips.min()
    v_1[times] = c_1/ncoins
    v_rand[times] = c_rand/ncoins
    v_min[times] = c_min/ncoins

#print(v_min.mean())
# plt.hist(v_rand, bins=11, normed=False, facecolor='green', alpha=0.5)
# plt.show(True)

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import scipy.stats as stats

data = [v_1, v_rand, v_min]
titles = ["nu_1", "nu_rands", "nu_mins"]
for i in range(3):
    plt.figure(i)
    # example data
    x = sorted(data[i])  # distribution
    mu = np.mean(x)  # mean of distribution
    sigma = np.std(x)  # standard deviation of distribution

    num_bins = 11
    # the histogram of the data
    weights = np.ones_like(x) / len(x)
    n, bins, patches = plt.hist(x, weights=weights, bins=num_bins, normed=False, facecolor='green', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma) / num_bins

    plt.plot(bins, y, 'r--')
    plt.xlabel('{}'.format(titles[i]))
    plt.ylabel('Probability')
    plt.title(r'Histogram of {}: $\mu={}$, $\sigma={}$'.format(titles[i],
                                                               round(mu, 2),
                                                               round(sigma, 2)
                                                               )
              )

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
plt.show()
