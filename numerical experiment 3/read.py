import csv
import numpy as np


input_file = open("input_wheat.csv")
with input_file:
    reader1 = csv.reader(input_file)
    rows1 = [row for row in reader1]
    rows1 = rows1[1:]
input_data = np.zeros((100,701))
for i in range(100):
    for j in range(701):
        input_data[i][j] = float(rows1[i][j])


output_file = open("output_wheat.csv")
with output_file:
    reader2 = csv.reader(output_file)
    rows2 = [row for row in reader2]
    rows2 = rows2[1:]
output_data = np.zeros(100)
for i in range(100):
    output_data[i] = float(rows2[i][0])

np.save('input_training_wheat.npy', input_data[:80])
np.save('output_training_wheat.npy', output_data[:80])
np.save('input_testing_wheat.npy', input_data[80:])
np.save('output_testing_wheat.npy', output_data[80:])
