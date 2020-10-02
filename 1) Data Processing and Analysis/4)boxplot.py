import matplotlib.pyplot as plt

f=open(r"E:\Machine learning\spyder\saledata.csv","r")
salefile=f.readlines()

sale_list=[]

for record in salefile:
    sale_list.append(int(record))
    
plt.title("Box plot of sale")

plt.boxplot(sale_list)
plt.show()