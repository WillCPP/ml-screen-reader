from os import listdir
from os.path import isfile, join
from html.parser import HTMLParser
import string

class LineHTMLParser(HTMLParser):
    data = []
    # classification = 0
    def handle_starttag(self, tag, attrs):
        # print("Encountered a start tag:", tag)
        self.data.append(0)

    def handle_endtag(self, tag):
        # print("Encountered an end tag :", tag)
        self.data.append(0)

    def handle_data(self, data):
        print("Encountered some data  :", data)
        self.data.append(1)

path = 'wikipediaFiles/output/'
files = [f for f in listdir(path) if isfile(join(path, f))]

parser = LineHTMLParser()
printable = set(string.printable)


i = 0
count = 0
for f in files:
    atStart = True
    atEnd = False
    classList = []
    with open(path+f, encoding='utf-8', errors='ignore') as file:
        line = file.readline()
        while line:
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
            # print(parser.data)
            # End loop
            line = file.readline()
            count += 1
    print(classList)
    print(count)
    i += 1
    if i > 0:
        break
