import datetime as dt
import matplotlib as plt
import pygtail as tail

fig = plt.figure()
ax = fig.add_subplot()

def testWatchdog():

when a file is changed, open it and read its contents and plot them