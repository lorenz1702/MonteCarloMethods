import math
import matplotlib.pyplot as plt
import random
import numpy as np

import math
import random

def box_muller():
    u1 = random.random()
    u2 = random.random()

    z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

    return z1, z2

def gaussian_distribution(mean, std_dev, num_samples):
  samples = []
  for _ in range(num_samples // 2):
    z1, z2 = box_muller()
    samples.append(z1 * std_dev + mean)
    samples.append(z2 * std_dev + mean)

  return samples


def exponiation_function():
    u1 = random.random()

    x1 = - math.log(1-u1)

    return x1

def gaussian_distribution_exp(mean, std_dev, num_samples):
  samples = []
  for _ in range(num_samples // 2):
    x1 = exponiation_function()
    samples.append(x1 * std_dev + mean)

  return samples




mean = 0.0
std_dev = 1.0
num_samples = 1000

gaussian_samples = gaussian_distribution(mean, std_dev, num_samples)
gaussian_samples_exp = gaussian_distribution_exp(mean, std_dev, num_samples*2)

print(gaussian_samples[0:10])





plt.hist(gaussian_samples, bins = 50)
plt.hist(gaussian_samples_exp, bins = 50)
plt.show()