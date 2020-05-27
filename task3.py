
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

import tools


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi

    # Шаг по пространству
    dx = 5e-3

    # Скорость света
    c=3e8

    #
    x=1.8

    # Число Куранта
    Sc = 1
    
    #
    dt=Sc*dx/c
  
    # Время расчета в отсчетах
    maxTime = 300

    # Размер области моделирования в отсчетах
    maxSize = int(x/dx)

    # Положение источника в отсчетах
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [100]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize)

    for probe in probes:
        probe.addData(Ez, Hy)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dt)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])


    # Параметры гауссова импульса
    A0 = 1 
    Am = 100 
    Fm = 2.5e9*dt
    w_g = np.sqrt(np.log(Am)) / (np.pi * Fm)
    d_g = w_g * np.sqrt(np.log(A0))

                        

    for t in range(1, maxTime):
        # Граничные условия для поля H
        Hy[-1] = Hy[-2]

        # Расчет компоненты поля H
        Ez_shift = Ez[1:]
        Hy[:-1] = Hy[:-1] + (Ez_shift - Ez[:-1]) * Sc / W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        #Hy[sourcePos - 1] -= (Sc / W0) * np.exp(-(t - 30.0 - sourcePos) ** 2 / 100.0)
        #Hy[sourcePos - 1] -= (Sc / W0) * np.exp(-(t - 30.0) ** 2 / 100.0)
        Hy[sourcePos - 1] -= (Sc / W0) * np.exp(-((t - d_g - sourcePos) / w_g) ** 2) 
        # Граничные условия для поля E
        Ez[0] = Ez[1]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:] = Ez[1:] + (Hy[1:] - Hy_shift) * Sc * W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        #Ez[sourcePos] += Sc * np.exp(-((t + 0.5) - (sourcePos - 0.5) - 30.0) ** 2 / 100.0)
        #Ez[sourcePos] += Sc * np.exp(-((t + 0.5) - (-0.5) - 30.0) ** 2 / 100.0)
        Ez[sourcePos] += Sc * np.exp(-(((t + 0.5) - (sourcePos - 0.5) - d_g) / w_g) ** 2)

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % 2 == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1)
    t=np.arange(0, len(probes[0].E))*dt
    df = 1.0 / (maxTime * dt)
    gauss=probe.E
    spectrum = np.abs(fft(gauss))
    spectrum = fftshift(spectrum)
    freq = np.arange(-maxTime / 2 * df, maxTime / 2 * df, df)
    # Отображение спектра
plt.figure
plt.plot(freq, spectrum / np.max(spectrum))
plt.grid()
plt.xlabel('Частота, Гц')
plt.ylabel('|S| / |Smax|')
plt.xlim(0, 5e9)

plt.subplots_adjust(wspace=0.4)
plt.show()
    
 
