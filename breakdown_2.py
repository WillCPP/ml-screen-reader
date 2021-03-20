from os import listdir
from os.path import isfile, join, split
from html.parser import HTMLParser
import string
import numpy as np

def pad_list(l):
    s = len(l)
    if s < 200:
        for i in range(s,200):
            l.append(2)

class LineHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.data = []
    
    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        if len(self.data) < 200:
            self.data.append(0)

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        if len(self.data) < 200:
            self.data.append(0)

    def handle_data(self, data):
        print("Encountered some data  :", data)
        if len(self.data) < 200 and data[0] != '(' and data[0] != '{':
            s = data.split(' ')
            for w in s[:200 - len(self.data)]:
                self.data.append(1)
            print(f'DONE| {len(self.data)}')

path = 'wikipediaFiles/output/'
files = [f for f in listdir(path) if isfile(join(path, f))]

arr_d = np.empty(shape=(0,200), dtype=int)
arr_c = np.empty(shape=(0,1), dtype=int)

i = 0
count = 0
for f in files:
    atStart = True
    atEnd = False
    classList = []
    with open(path+f, encoding='utf-8', errors='ignore') as file:
        line = file.readline()
        while line:
            parser = LineHTMLParser()
            # Do stuff
            while 'mw-headline' not in line and atStart:
                line = file.readline()
                if 'mw-headline' in line:
                    atStart = False

            if 'id="Notes"' in line:
                atEnd = True

            if not atEnd:
                classification = 1
            else:
                classification = 0

            parser.feed(line)
            classList.append(classification)
            
            if len(parser.data) == 200:
                print('FULL')
                print(parser.data)
            pad_list(parser.data)
            print(f'LEN: {len(parser.data)}')

            # arr_d = np.append(arr_d, [parser.data])
            arr_d = np.vstack((arr_d, parser.data))

            line = file.readline()
            count += 1
    
    print(arr_d)

    np.save('r', arr_d)
    np.savetxt('r.csv', arr_d, fmt='%d', delimiter=',')
    
    # print(classList)
    print(count)
    i += 1
    if i > 0: # Early break for testing
        break

