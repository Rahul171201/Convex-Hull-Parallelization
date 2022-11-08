import numpy as np
import matplotlib.pyplot as plt

numbers = []

#Open the file
with open('points.txt') as fp:
    #Iterate through each line
    for line in fp:

        numbers.extend( #Append the list of numbers to the result array
            [int(item) #Convert each number to an integer
             for item in line.split() #Split each line of whitespace
             ])
             
x = []
y = []

# Partitioning the numbers into x and y coordinates
n = numbers[0]

for i in range(1, 2*n, 2):
    x.append(numbers[i])
    y.append(numbers[i+1])

plt.scatter(x, y, color='red')

# Reading into convex hull points
n_convex_hull = numbers[2*n+1]
convex_hull = []

for i in range(2*n+2, len(numbers), 2):
    convex_hull.append((numbers[i], numbers[i+1]))

print("The points in the convex hull are :\n" ,convex_hull)

convex_hull_x = []
convex_hull_y = []

for i in range(0, len(convex_hull)):
    convex_hull_x.append(convex_hull[i][0])
    convex_hull_y.append(convex_hull[i][1])

# Plotting the convex Hull
convex_hull_x.append(convex_hull_x[0])
convex_hull_y.append(convex_hull_y[0])

plt.plot(convex_hull_x, convex_hull_y)

plt.show()