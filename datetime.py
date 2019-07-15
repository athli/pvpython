import datetime
import matplotlib.pyplot

def recordTimedData():
    name = input("What should this file be called? Do not include .txt ")
    f = open(name + ".txt", "a+")
    while(True):
        temp = input("Cell temp: ")
        if temp == 'done':
            f.close()
            break
        elif temp[0:7] == "Comment":
            f.write(datetime.datetime.now().strftime("%Y-%m-%d \t %H:%M:%S"))
            f.write('\t')
            f.write('\t')
            f.write(temp)
            f.write('\n')
        else:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d \t %H:%M:%S"))
            f.write('\t')
            f.write(temp)
            f.write('\n')
