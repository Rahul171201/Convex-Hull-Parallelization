import numpy as np
import matplotlib.pyplot as plt

input_data = []
input_x = []
input_y = []
with open('input_points.txt') as fp:
    for line in fp:
        input_data.extend([int(item) for item in line.split()])
input_n = input_data[0]
for i in range(1, 2*input_n+1, 2):
    input_x.append(input_data[i])
    input_y.append(input_data[i+1])

output_data = []
output_x = []
output_y = []
with open('output_points.txt') as fp:
    for line in fp:
        output_data.extend([int(item) for item in line.split()])
output_n = output_data[0]
for i in range(1, 2*output_n+1, 2):
    output_x.append(output_data[i])
    output_y.append(output_data[i+1])

output_x.append(output_x[0])
output_y.append(output_y[0])

plt.scatter(input_x,input_y,color='red')
plt.plot(output_x,output_y)
plt.show()