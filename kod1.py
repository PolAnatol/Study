import numpy as np
import matplotlib.pyplot as plt
f= open('result.txt', 'w');
l=np.linspace(-1000, 1000, 2001);
x=l/100;
y=-np.cos(x)*np.cos(np.pi)*np.exp(-(x-np.pi)**2);
plt.plot(x,y)
plt.savefig('graph.png')
plt.show()
for i in range(2001):
    f.write(str(x[i]))
    f.write('\t ')
    f.write(str(y[i]))
    f.write('\n')
f.close

