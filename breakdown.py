from collections import OrderedDict
import numpy as np
import json

# Find by > < to decided if reading
def keep(line):
    return True

def numpyLine(line):
    list = []
    start = False
    end = False
    for x in line:
        if x == '>':
            start = True
        elif x == '<'

f = open("wikipediaFiles\input\en.wikipedia.orgwiki%C3%86lfheah_of_Canterbury.html.html", encoding='UTF8')
d = OrderedDict()
line = f.readline()
while "</head>" not in line:
    line = f.readline()

line = f.readline()
while '<ol class="references">' not in line:
    if keep(line):
        d[line] = 1
    else:
        d[line] = 0
    line = f.readline()

for x in f:
    d[x] = 0

f.close()

print(d)

with open("wikipediaFiles\output\en.wikipedia.orgwiki%C3%86lfheah_of_Canterbury.html.txt", 'w') as outfile:
    json.dump(d, outfile)
