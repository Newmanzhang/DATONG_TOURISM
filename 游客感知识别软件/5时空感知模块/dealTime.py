import csv
from datetime import datetime
with open("datacsvdatongrawfinal1.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    sum_1=0
    sum_2=0
    sum_3=0
    sum_4=0
    for row in reader:
        time=datetime.strptime(row[3], "%Y/%m/%d %H:%M")
        if str(time.month) in ['1','2','3']:
            sum_1=sum_1+1
        elif str(time.month) in ['4','5','6']:
            sum_2=sum_2 + 1
        elif str(time.month) in ['7','8','9']:
            sum_3=sum_3 + 1
        else:
            sum_4 = sum_4 + 1

print('一季度',sum_1)
print('二季度',sum_2)
print('三季度',sum_3)
print('四季度',sum_4)