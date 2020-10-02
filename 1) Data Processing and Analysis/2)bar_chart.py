import matplotlib.pyplot as plt
x_city=["New_york","london","Dubai","New_Delhi","Tokyo"]
y_temp=[75,65,105,98,90]


plt.title("Temp_variation")
plt.xlabel("cities")
plt.ylabel("Temperature")

plt.bar(x_city,y_temp)
plt.show()