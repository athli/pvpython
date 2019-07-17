import datetime
import matplotlib.pyplot

def recordTimedData():
    # Writes data in real time to a .txt file
    name = input("What should this file be called? Include .txt ")
    while(True):
        f = open(name, "a+")
        temp = input("Cell temp: ")
        if temp == '':
            continue
        elif temp == 'done': # trigger this when finished
            f.close()
            break
        # special cases go in a third column
        elif temp[0:7] == "Comment" or temp[0:7] == "Shading" or temp[0:15] == "Removed shading" or temp[0:12] == "Open circuit":
            f.write(datetime.datetime.now().strftime("%H:%M:%S.%f"))
            f.write('\t')
            f.write('\t')
            f.write(temp)
            f.write('\n')
            f.close()
        # assume everything else is a number
        # TODO: should probably implement something to actually check this, and then have 
        #   a case that does nothing if it's not a number
        else:
            f.write(datetime.datetime.now().strftime("%H:%M:%S.%f"))
            f.write('\t')
            f.write(temp)
            f.write('\n')
            f.close()
