from collections import OrderedDict
import numpy as np
import re
import json
from tempfile import TemporaryFile

# Find by > < to decided if reading
def keep(linelist):
    if 1 in linelist:
        return True
    return False

def numpyLine(line):
    list = []
    start = False
    end = False
    for x in line:
        if x == '>':
            start = True
            list.append(0)
        elif x == '<':
            start = False
            end = False
            list.append(0)
        elif start and not end and re.search("[a-zA-Z0-9]", x) != None:
            end = True
            list.append(1)
        elif start and end:
            list.append(1)
        else:
            list.append(0)
    print(line)
    print(list)
    return list


f = open("wikipediaFiles\input\en.wikipedia.orgwiki%C3%86lfheah_of_Canterbury.html.html", encoding='UTF8')
d = OrderedDict()
arr = np.ndarray(shape = (), dtype = object)
oldlist = []
oldClass = []
#arr = np.array([], dtype = object)
classArr = np.ndarray(shape = (1,))
line = f.readline()
while "</head>" not in line:
    line = f.readline()

#line = f.readline()
#while '<ol class="references">' not in line:
for x in f:
    newline = numpyLine(x)
    if keep(newline):
        #arr = np.vstack([arr, np.array(newline, dtpe=object)])
        oldlist.append([newline])
        oldClass.append(1)
        #classArr = np.append(1, classArr)
    else:
        #arr = np.vstack([arr, np.array(newline, dtype=object)])
        oldlist.append([newline])
        oldClass.append(1)
        #classArr = np.append(0, classArr)
    #line = f.readline()

#for x in f:
#    d[x] = 0

f.close()
print(oldlist[-1])
arr = np.array(oldlist, dtype=object)
classArr = np.array(oldClass)
#print(oldlist)
print(arr[-1][-1])
print(type(arr[-1][-1]))
print(np.shape(arr))
print(np.shape(arr[-1]))
print(np.shape(arr[-1][-1]))
print(np.shape(classArr))
#print(classArr)
#print(d)

with open('trainning.npy', 'wb') as n:
    np.save(n, arr)
with open('trainningresults.npy', 'wb') as c:
    np.save(c, classArr)
#with open("wikipediaFiles\output\en.wikipedia.orgwiki%C3%86lfheah_of_Canterbury.html.txt", 'w') as outfile:
#    json.dump(d, outfile)
