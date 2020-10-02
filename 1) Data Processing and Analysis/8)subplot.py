# ------------------------------------------------------------
# Create the scatter plot of Sales Vs cost from the file data
# ------------------------------------------------------------

# Import pyplot
import matplotlib.pyplot as plt

# Open the file in read mode and read lines
f = open('salesdata2.csv','r')
salefile = f.readlines()

# Create the sales List
sale_list=[]
s_list = []
c_list = []

# Append all the records from the file to the saleslist
for records in salefile:
    sale, cost = records.split(sep=',')
    s_list.append(int(sale))
    c_list.append(int(cost))
    
sale_list.append(s_list)
sale_list.append(c_list)

# Sctter Plot
plt.subplot(2,2,1)
plt.title("Sales Vs Cost")
plt.xlabel("Sale")
plt.ylabel("Cost")
plt.scatter(s_list,c_list)
#plt.savefig()

#Box Plot
plt.subplot(2,2,2)
plt.title("Box plot of sales")
plt.ylabel("USD")
plt.boxplot(sale_list)
#plt.savefig()

#plot Histrogram
plt.subplot(2,2,3)
plt.title("Histogram of Sales")
plt.ylabel("USD")
plt.hist(s_list,bins=5,rwidth=0.9)

#Plot LinePlot
x_days=[1,2,3,4,5]
y_prices1=[9,9.5,10.1,10,12]
plt.subplot(2,2,4)
plt.title("Stock Movement")
plt.xlabel("Days")
plt.ylabel("Price_in_USD")
plt.plot(x_days,y_prices1,label="stock1")

plt.tight_layout()
plt.show()






