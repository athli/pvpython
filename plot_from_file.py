import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.add_subplot()
times = []
temps = []
deltat = 1000

name = input("What file should we read from? Include .txt ")

def realTimePlot(j):
    # reads data from a file and plots in real time as the file is edited
    xs = []
    ys = []
    f = open(name, "r")
    lines = [line.rstrip('\n') for line in f]
    for line in lines:
        fields = line.strip().split('\t') # split into "columns"
        if fields[1] != '': # if not a special case (see datetime.py)
            temp = float(fields[1])
            ys.append(temp)
            xs.append(dt.datetime.strptime(fields[0][0:15], "%H:%M:%S.%f"))
    ax.clear()
    ax.plot(xs, ys)
    plt.yscale('log') # looking for something linear with log scale
    f.close()

# call realTimePlot every deltat microseconds
ani = animation.FuncAnimation(fig, realTimePlot, interval = deltat)
plt.show()