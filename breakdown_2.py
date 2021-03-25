from os import listdir
from os.path import isfile, join, split
from html.parser import HTMLParser
import string
import numpy as np
import multiprocessing

def pad_list(l):
    s = len(l)
    if s < 200:
        for i in range(s,200):
            l.append(0)

class LineHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.data = []
    
    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        if len(self.data) < 200:
            self.data.append(1)

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        if len(self.data) < 200:
            self.data.append(1)

    def handle_data(self, data):
        # print("Encountered some data  :", data)
        if len(self.data) < 200 and data[0] != '(' and data[0] != '{':
            s = data.split(' ')
            for w in s[:200 - len(self.data)]:
                self.data.append(2)
            # print(f'DONE| {len(self.data)}')

path = 'wikipediaFiles/output/'

# ==================================================


def process(files, i):
    count = 0
    for f in files:
        d = np.empty(shape=(0,200), dtype=np.float)
        c = np.array([], dtype=np.float)
        atStart = True
        atEnd = False
        classList = []
        with open(path+f, encoding='utf-8', errors='ignore') as file:
            line = file.readline()
            while line:
                parser = LineHTMLParser()
                while 'mw-headline' not in line and atStart:
                    line = file.readline()
                    if 'mw-headline' in line:
                        atStart = False
                # if 'id="Notes"' in line:
                #     atEnd = True
                # if not atEnd:
                #     classification = 1
                # else:
                #     classification = 0
                if 2 in parser.data:
                    classification = 1
                else:
                    classification = 0
                parser.feed(line)
                classList.append(classification)
                pad_list(parser.data)
                d = np.vstack((d, parser.data))
                line = file.readline()
        c = np.append(c, np.array(classList, dtype=np.float))
        np.save(f'dataset/data/{i}_{count}_data', d)
        np.save(f'dataset/labels/{i}_{count}_labels', c)
        count += 1

def run():
    
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print(len(files))

    f0 = files[:739-1]
    f1 = files[739:(2*739-1)]
    f2 = files[(2*739):(3*739-1)]
    f3 = files[(3*739):(4*739-1)]
    f4 = files[(4*739):(5*739-1)]
    f5 = files[(5*739):(6*739-1)]
    f6 = files[(6*739):(7*739-1)]
    f7 = files[(7*739):]

    t0 = multiprocessing.Process(target=process, name=f'=process_0', args=(f0, 0,))
    t0.start()
    t1 = multiprocessing.Process(target=process, name=f'=process_1', args=(f1, 1,))
    t1.start()
    t2 = multiprocessing.Process(target=process, name=f'=process_2', args=(f2, 2,))
    t2.start()
    t3 = multiprocessing.Process(target=process, name=f'=process_3', args=(f3, 3,))
    t3.start()
    t4 = multiprocessing.Process(target=process, name=f'=process_4', args=(f4, 4,))
    t4.start()
    t5 = multiprocessing.Process(target=process, name=f'=process_5', args=(f5, 5,))
    t5.start()
    t6 = multiprocessing.Process(target=process, name=f'=process_6', args=(f6, 6,))
    t6.start()
    t7 = multiprocessing.Process(target=process, name=f'=process_7', args=(f7, 7,))
    t7.start()

if __name__ == '__main__':
    run()
# ==================================================



# arr_d = np.empty(shape=(0,200), dtype=int)
# # arr_c = np.empty(shape=(0,), dtype=int)
# arr_c = np.array([], dtype=int)

# i = 0
# # count = 0
# for f in files:
#     atStart = True
#     atEnd = False
#     classList = []
#     with open(path+f, encoding='utf-8', errors='ignore') as file:
#         line = file.readline()
#         while line:
#             parser = LineHTMLParser()
#             # Do stuff
#             while 'mw-headline' not in line and atStart:
#                 line = file.readline()
#                 if 'mw-headline' in line:
#                     atStart = False

#             if 'id="Notes"' in line:
#                 atEnd = True

#             if not atEnd:
#                 classification = 1
#             else:
#                 classification = 0

#             parser.feed(line)
#             classList.append(classification)
            
#             # if len(parser.data) == 200:
#             #     print('FULL')
#             #     print(parser.data)
#             pad_list(parser.data)
#             # print(f'LEN: {len(parser.data)}')

#             arr_d = np.vstack((arr_d, parser.data))

#             line = file.readline()
#             # count += 1
    
#     # print(arr_d)
#     # print(arr_d.shape)
#     arr_c = np.append(arr_c, np.array(classList, dtype=int))
#     # print(arr_c)
#     # print(arr_c.shape)

#     # np.save('r', arr_d)
#     # np.savetxt('r.csv', arr_d, fmt='%d', delimiter=',')
    
#     i += 1
#     if i % 10 == 0:
#         print(f'FILES PROCESSED: {i}')
#     # if i > 0: # Early break for testing
#     #     break