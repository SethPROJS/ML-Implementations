from linearRegression import simpleLinearRegression
from matplotlib import pyplot as plt
import random

x = []
y = []

y_offline = []

for i in range(200):
    x.append(i)

for i in range(len(x)):
    y.append(x[i]*5*(random.randint(1,30)*1/10)+23.123)

plot = plt.scatter(x, y)

Model = simpleLinearRegression(x, y)

for i in range(len(x)):
    y_offline.append(Model.run_prediction(x[i]))

plot2 = plt.plot(x, y_offline, color = 'red')

plt.show()
