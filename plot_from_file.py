import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot()
times = []
temps = []
deltat = 1000

name = input("What file should we read from? Include .txt ")

def realTimePlot(j, xs, ys):
    f = open(name, "r")
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        fields = line.strip().split('\t')
        if fields[1] != '':
            temp = float(fields[1])
            ys.append(temp)
            xs.append(dt.datetime.strptime(fields[0][0:15], "%H:%M:%S.%f"))
    ax.clear()
    ax.plot(xs[0:2], ys[0:2])
    ax.plot(xs[2:(len(xs)-1)], ys[2:(len(ys)-1)])
    f.close()

ani = animation.FuncAnimation(fig, realTimePlot, fargs = (times,temps), interval = deltat)
plt.show()