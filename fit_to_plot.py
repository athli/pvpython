import datetime as dt
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.optimize as optimize

def modelFunc(t, T_inf, deltaT, t_0, tau):
    # expected exponential decay model for solar cell temperature
    return T_inf + deltaT*np.exp(-(dates.date2num(t)-t_0)/tau)

fig = plt.figure()
ax = fig.add_subplot()
times = []
temps = []
deltat = 1000

name = input("What file should we read from? Include .txt ")

def fitToData(z):
    xs = []
    ys = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    temp = 0
    shading = ""
    f = open(name, "r")
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        fields = line.strip().split('\t')
        if fields[1] != '':
            temp = float(fields[1])
            ys.append(temp)
            xs.append(dt.datetime.strptime(fields[0][0:15], "%H:%M:%S.%f"))
        if len(fields) > 2:
            if fields[2] == "Shading":
                shading = "Shading"
            elif fields[2] == "Removed shading":
                shading = "Removed shading"
            elif fields[2] == "Open circuit":
                shading == ''
        if shading == "Shading":
            x1.append(dt.datetime.strptime(fields[0][0:15], "%H:%M:%S.%f"))
            y1.append(temp)
        elif shading == "Removed shading":
            x2.append(dt.datetime.strptime(fields[0][0:15], "%H:%M:%S.%f"))
            y2.append(temp)
    ax.clear()
    ax.plot(xs, ys, 'bo')
    f.close()
    popt1, pcov1 = optimize.curve_fit(modelFunc, x1, y1, 
        bounds = ([-np.inf, -np.inf, dates.date2num(x1[1]) - .5, -np.inf], [np.inf, np.inf, dates.date2num(x1[1]) + .5, np.inf]))
    ax.plot(x1, modelFunc(x1, *popt1), 'r-', label = 'fit1')
    result = "       fit1\n T_inf: " + str(popt1[0]) + "\n tau:   " + str(popt1[3])
    if len(x2) != 0:
        popt2, pcov2 = optimize.curve_fit(modelFunc, x2, y2, 
            bounds = ([-np.inf, -np.inf, dates.date2num(x2[1]) - .5, -np.inf], [np.inf, np.inf, dates.date2num(x2[1]) + .5, np.inf]))
        ax.plot(x2, modelFunc(x2, *popt2), 'r-', label = 'fit2')
        result = "       fit1, fit2 \n T_inf: " + str(popt1[0]) + ", " + str(popt2[0]) + "\n tau:   " + str(popt1[3]) + ", " + str(popt2[3])
    return result

ani = animation.FuncAnimation(fig, fitToData, interval = deltat)
plt.show()