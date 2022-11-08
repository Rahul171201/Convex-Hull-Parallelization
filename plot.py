import numpy as np

numbers = []

#Open the file
with open('points.txt') as fp:
    #Iterate through each line
    for line in fp:

        numbers.extend( #Append the list of numbers to the result array
            [int(item) #Convert each number to an integer
             for item in line.split() #Split each line of whitespace
             ])

print(numbers)

x = []
y = []
