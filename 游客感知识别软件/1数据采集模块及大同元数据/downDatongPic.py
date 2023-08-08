import csv
import requests
import os
with open('datacsvdatongrawfinal.csv', 'r',encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        id  = row[0]
        url = row[2]
        if url == "None":
            continue
        response=requests.get(url)
        f = open('E:\datong\{0}{1}'.format(id,'.jpg'),"wb")
        f.write(response.content)
        f.close()
