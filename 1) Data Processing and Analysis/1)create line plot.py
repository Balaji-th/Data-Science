import matplotlib.pyplot as plt

x_days=[1,2,3,4,5]
y_prices1=[9,9.5,10.1,10,12]
y_prices2=[11,12,10.5,11.5,12.5]

plt.title("Stock Movement")
plt.xlabel("Days")
plt.ylabel("Price_in_USD")

plt.plot(x_days,y_prices1,label="stock1")
plt.plot(x_days,y_prices2,label="stock2")

plt.legend(loc=2,fontsize=12)
plt.show()
