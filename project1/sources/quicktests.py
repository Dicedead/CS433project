from helpers import *
from preprocessing import *

from additional_implementations import *
from datetime import datetime
import matplotlib.pyplot as plt


x = lambdas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1]
y = [0.39114789, 0.39115315, 0.3912603, 0.39460472, 0.41996818, 0.46430481, 0.49739291]
plt.plot(x, y, label="Test loss", marker=".")
plt.xscale("log")
plt.legend()
plt.title("Cross validation of lambda parameter for jet num 0")
plt.show()
