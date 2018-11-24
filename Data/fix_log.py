import csv
import pandas as pd

with open('driving_log.csv', 'r') as readFile:
    with open('driving_log_fixed.csv', 'w') as writeFile:
        reader = csv.reader(readFile)
        writer = csv.writer(writeFile)
        for rowix, row in enumerate(reader):
            for colix in range(3):
                add = row[colix]
                ix = add.rfind('a\\')
                new_val = row[colix][ix+2:]
                new_val = new_val.replace('\\', '/')
                row[colix] = 'Data/' + new_val
            writer.writerow(row)
