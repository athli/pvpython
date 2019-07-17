import datetime
import matplotlib.pyplot

def recordTimedData():
    name = input("What should this file be called? Include .txt ")
    while(True):
        f = open(name, "a+")
        temp = input("Cell temp: ")
        if temp == '':
            continue
        elif temp == 'done':
            f.close()
            break
        elif temp[0:7] == "Comment":
            f.write(datetime.datetime.now().strftime("%H:%M:%S.%f"))
            f.write('\t')
            f.write('\t')
            f.write(temp)
            f.write('\n')
            f.close()
        else:
            f.write(datetime.datetime.now().strftime("%H:%M:%S.%f"))
            f.write('\t')
            f.write(temp)
            f.write('\n')
            f.close()
