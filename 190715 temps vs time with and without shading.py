import matplotlib
import datetime
data = [68.6, 68, 69, 70, 73, 76, 78, 79, 80, 81, 83, 85, 88, 90, 92, 93, 95, 96, 97, 98, 100, 101, 
 102, 103, 104, 105, 106, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87,
 ] 
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '86',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '83',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '82',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '81',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '80',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '79',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '78',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '77',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '76',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '75',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '74',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 '73',
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717)]

times = [datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 datetime.datetime(2019, 7, 15, 14, 12, 47, 363717),
 ]
temps = []

for i in range(len(data)):
    if i % 2 == 0:
        temps += data[i]
    else:
        times += data[i]

plt = matplotlib.pyplot.plot_date(times, temps)

plt.show()
