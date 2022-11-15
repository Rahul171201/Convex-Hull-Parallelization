import numpy as np
import matplotlib.pyplot as plt

numbers = []

#Open the file
with open('plot.txt') as fp:
    #Iterate through each line
    counter = 0
    for line in fp:
        if(counter==0):
            numbers.extend( #Append the list of numbers to the result array
            [int(item) #Convert each number to an integer
             for item in line.split() #Split each line of whitespace
             ])
        else:
            numbers.extend( #Append the list of numbers to the result array
            [float(item) #Convert each number to a float
             for item in line.split() #Split each line of whitespace
             ])
        counter = counter+1
             
n = []
time_n = []
max_val = []
time_max_val = []

# Partitioning the numbers into x and y coordinates
N1 = numbers[0]
N2 = numbers[1]
print(N1, N2)

for i in range(2, 2*N1+2, 2):
    n.append(numbers[i])
    time_n.append(numbers[i+1])

for i in range(N1*2+2, 2+2*N1+2*N2, 2):
    max_val.append(numbers[i])
    time_max_val.append(numbers[i+1])
    print(max_val)

plt.scatter(n, time_n, color='blue')
plt.plot(n, time_n, 'b-',label='Plot of number of elements in the plane to speedup')

plt.scatter(max_val, time_max_val, color='green')
plt.plot(max_val, time_max_val, 'g-',label='Plot of maximum coordinate values to speedup')
plt.legend()
plt.show()