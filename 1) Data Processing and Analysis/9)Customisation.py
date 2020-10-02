# Create the scatter plot of Sales Vs cost from the file data

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
#marker- 0,s,^,+,X,D
#c-color b,g,r,c,m,y,k,w
#c-float in quotes for greyscales "0.80"
#c-Html hex code such as #FF5733
plt.subplot(2,3,1)
plt.title("Sales Vs Cost")
plt.xlabel("Sale")
plt.ylabel("Cost")
plt.scatter(s_list,c_list,
            marker="*",
            s=100,c="#FF5733")
#plt.savefig()

#Box Plot
plt.subplot(2,3,2)
plt.title("Box plot of sales")
plt.ylabel("USD")
plt.boxplot(sale_list,                                           \
            patch_artist=True,                                   \
            boxprops=dict(facecolor="g",color="r",linewidth=2),  \
            whiskerprops=dict(color="r",linewidth=2),            \
            medianprops=dict(color="w",linewidth=1),             \
            capprops=dict(color="k",linewidth=2),                \
            flierprops=dict(markerfacecolor="r",marker="o",markersize=5))
#plt.savefig()

#plot Histrogram
plt.subplot(2,3,3)
plt.title("Histogram of Sales")
plt.ylabel("USD")
plt.hist(s_list,bins=5,rwidth=0.9,color="b")

#Plot LinePlot
x_days=[1,2,3,4,5]
y_prices1=[9,9.5,10.1,10,12]
plt.subplot(2,3,4)
plt.title("Stock Movement")
plt.xlabel("Days")
plt.ylabel("Price_in_USD")
plt.plot(x_days,y_prices1,label="stock1",color="g",
         marker="o",markersize=5,linewidth=3,linestyle="--")

#Plot bar_chart
plt.subplot(2,3,5)
x_city=["New_york","london","Dubai","New_Delhi","Tokyo"]
y_temp=[75,65,105,98,90]
plt.title("Temp_variation")
plt.xlabel("cities")
plt.ylabel("Temperature")
plt.xticks(rotation="45")
plt.bar(x_city,y_temp,color=["r","g","c","y","m"])

plt.tight_layout()
plt.show()






