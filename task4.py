
import numpy as np
import tools
import matplotlib.pyplot as plt
import numpy.fft as fft


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * np.pi
    dx=5e-3
    c=3e8

    # Число Куранта
    Sc = 1.0
    dt=Sc*dx/c

    # Время расчета в отсчетах
    maxTime = 600

    # Размер области моделирования в отсчетах
    maxSize = 400

    # Первый слой
    d1=0.15
    eps1=2.4
    layer1_start = int(1 / dx)
    layer1_end = layer1_start + int(d1 / dx)

    # Первый слой
    d2=0.4
    eps2=4.4
    layer2_start = layer1_end
    layer2_end = layer2_start + int(d2 / dx)

    # Первый слой
    d3=0.06
    eps3=6
    layer3_start = layer2_end 
    layer3_end = layer3_start + int(d3 / dx)

    # Первый слой\
    eps4=5.2
    layer4_start = layer3_end
    layer4_end = layer4_start + int(maxSize-layer4_start)

    # Датчики для регистрации поля
    probesPos = [25,75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Левая граница TFSF
    tfsf_left = 50
    eps=np.ones(maxSize)
    eps = np.ones(maxSize)
    eps[layer1_start:] = eps1
    eps[layer2_start:] = eps2
    eps[layer3_start:] = eps3
    eps[layer4_start:] = eps4
    mu = np.ones(maxSize - 1)
    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)
   

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([tfsf_left])
    display.drawBoundary(layer1_start)
    display.drawBoundary(layer2_start)
    display.drawBoundary(layer3_start)
    display.drawBoundary(layer4_start)

      # Параметры гауссова импульса
    A0 = 1 
    Am = 100 
    Fm = 2e9*dt
    w_g = np.sqrt(np.log(Am)) / (np.pi * Fm)
    d_g = w_g * np.sqrt(np.log(A0))
    

    for t in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[tfsf_left - 1] -= (Sc / (W0 * mu[tfsf_left - 1])) * np.exp(-((t - d_g - tfsf_left)/w_g) ** 2)
        # Граничные условия для поля E
        Ez[0] = Ez[1]
        Ez[-1] = Ez[-2]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[tfsf_left] += Sc / (np.sqrt(eps[tfsf_left] * mu[tfsf_left])) *np.exp(-(((t + 0.5) - (tfsf_left - 0.5) - d_g) / w_g) ** 2) 
       

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)
        if t % 2 == 0:
            display.updateData(display_field, t)
    display.stop()
 # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)
 # Размера БПФ
    size = 4096
    FallField = np.zeros(maxTime)
    FallField[0:150] = probes[1].E[0:150] 
 # Нахождение БПФ падающего поля
    FallSpectr = abs(fft.fft(FallField, size))
    FallSpectr = fft.fftshift(FallSpectr)
 # Нахождение БПФ отраженного поля
    ScatteredSpectr = abs(fft.fft(probes[0].E, size))
    ScatteredSpectr = fft.fftshift(ScatteredSpectr)
 # шаг по частоте и определение частотной оси
    df = 1 / (size * dt)
    f = np.arange(-(size / 2) * df, (size / 2) * df, df)
 # Построение спектра падающего и рассеянного поля
    plt.figure()
    plt.plot(f * 1e-9, FallSpectr / np.max(FallSpectr))
    plt.plot(f * 1e-9, ScatteredSpectr / np.max(ScatteredSpectr))
    plt.grid()
    plt.xlim(0, 2e9 * 1e-9)
    plt.xlabel('f, ГГц')
    plt.ylabel('|S/Smax|')
    plt.legend(['Спектр падающего поля', 'Спектр отраженного поля'])
 # Определение коэффициента отражения и построения графика
    plt.figure()
    plt.plot(f * 1e-9, ScatteredSpectr / FallSpectr)
    plt.xlim(0, 2e9 * 1e-9)
    plt.ylim(0, 0.4)
    plt.grid() 
    plt.xlabel('f, ГГц')
    plt.ylabel('|Г|')
    plt.show() 
