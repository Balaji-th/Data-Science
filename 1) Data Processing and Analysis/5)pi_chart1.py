import matplotlib.pyplot as plt

f=open(r"E:\Machine learning\spyder\agedata2.csv","r")
agefile=f.readlines()

city_list=[]

for record in agefile:
    age,city=record.split(sep=",")
    city_list.append(city)
    
from collections import Counter
city_count=Counter(city_list)

city_names=list(city_count.keys())
city_values=list(city_count.values())

plt.pie(city_values,labels=city_names,autopct="%.2f%%")
plt.show()
