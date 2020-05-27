import requests as rq
import re
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import os
response=rq.get('https://jenyay.net/uploads/Student/Modelling/task_02.txt')
x=re.search(r'5\W.+',response.text)
z=x.group().split(';')
D=float(z[0].split('=')[1])
fmin=float(z[1].split('=')[1])
fmax=float(z[2].split('=')[1])
res=open('result2.txt', 'w')
print('Диаметр сферы D =',D, 'м\n',
      'Минимальная частота fmin = ',fmin/1e6,'МГц\n',
      'Максимальная частота fmin = ',fmax/1e9,'ГГц')
c = 3e8
r = D / 2
f = np.linspace(fmin, fmax + 1, 500)
sigma1 = []
for fx in f:
    lamda = c / fx
    k = 2 * np.pi / lamda
    h = []
    b = [0]
    a = []
    sigmaN = []
    n = 0
    while n < 70:
        h.append(
            sp.spherical_jn(n,k * r) + 1j * sp.spherical_yn(n, k * r))
        a.append(sp.spherical_jn(n, k * r) / h[n])
        n += 1
    n = 1
    while n < 70:
        b.append((k * r * sp.spherical_jn(n - 1, k * r) - n * \
                    sp.spherical_jn(n, k * r)) / (k * r * h[n - 1] - n * h[n]))
        n += 1
    n = 1
    while n < 70:
        sigmaN.append(((-1)**n) * (n + 1 / 2) * (b[n] - a[n]))
        n += 1
    sigma = (lamda**2 / np.pi) * (abs(np.sum(sigmaN)))**2
    print(str(fx * 1e-9) + '            ' + str(sigma), file=res)
    sigma1.append(sigma / (np.pi * (r**2)))
res.close()

# Построение графика
plt.figure()
plt.plot(2 * np.pi * f * r / c, sigma1)
plt.grid()
plt.ylabel('ЭПР')
plt.xlabel('длина волны')
plt.savefig('graph2.png')
plt.show()

