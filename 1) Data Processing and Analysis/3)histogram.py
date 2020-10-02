import matplotlib.pyplot as plt

f=open(r"E:\Machine learning\spyder\agedata.csv","r")
agefile=f.readlines()

age_list=[]

for record in agefile:
    age_list.append(int(record))

bins=[0,10,20,30,40,50,60,70,80,90,100]


plt.title("age_Histrogram")
plt.xlabel("Group")
plt.ylabel("Age")

plt.hist(age_list,bins,histtype="bar",rwidth=0.9)
plt.show()