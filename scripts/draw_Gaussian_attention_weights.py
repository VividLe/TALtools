# reference: CSDN blog

import numpy as np
import matplotlib.pyplot as plt
import cv2


res_file = '/data1/yangle/ShortActions/weight.png'
total_temporal_length = 30


x = np.linspace(0, total_temporal_length)

# Gaussian function 1
mu = 9
sigma = 3.5
y_sig_1 = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# # Gaussian function 2
mu = 16
sigma = 4
y_sig_2 = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# add noise to the figure
y_sig = y_sig_1 + y_sig_2
noise = np.random.randn(50) * 0.2
y_sig = y_sig + noise

plt.plot(x, y_sig, "r-", linewidth=2)

y_sig = np.expand_dims(y_sig, axis=0)

# save mask size
y_data = cv2.resize(y_sig, (600, 20), cv2.INTER_LINEAR)
y_data = y_data / y_data.max() * 255
y_data = y_data.astype(int)

cv2.imwrite(res_file, y_data)
plt.xlabel('Frequecy')
plt.ylabel('Latent Trait')
plt.title('Normal Distribution: $\mu = %.2f, $sigma=%.2f' % (mu, sigma))
plt.grid(True)
plt.show()


