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
#plt.figure("Scatter_plot")
plt.subplot(2,1,1)
plt.title("Sales Vs Cost")
plt.xlabel("Sale")
plt.ylabel("Cost")
plt.scatter(s_list,c_list)
#plt.savefig()

#Box Plot
#plt.figure("Box_Plot")
plt.subplot(2,1,2)
plt.title("Box plot of sales")
plt.ylabel("USD")
plt.boxplot(sale_list)
#plt.savefig()

plt.show()





