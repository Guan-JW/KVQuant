import re
import numpy as np
import matplotlib.pyplot as plt


# Open the file
with open("rst_comp_details_k_512.txt", "r") as file:
    content = file.read()

# Define regular expressions to extract the desired information
pattern_ratio = r"compressed ratio: (\d+\.\d+)"
pattern_time = r"compression time \(ms\): (\d+\.\d+)"

# Extract all compressed ratios and compression times
ratios = np.array([float(ratio) for ratio in re.findall(pattern_ratio, content)])
times = np.array([float(time) for time in re.findall(pattern_time, content)])

print(len(ratios)/22, len(times)/22)
y_values = []
for i in range(int(len(ratios)/22)):
    part_ratios = ratios[i*22: (i+1)*22]
    print(np.mean(part_ratios))
    y_values.append(np.mean(part_ratios))
# Print the results
print("Avg. Compressed Ratios:", np.average(ratios))
# print("Compression Times (ms):", times)

# Create the line chart
x_values = range(1, int(len(ratios)/22)+1)
plt.plot(x_values, y_values)

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Avg. LZ4 Compression Ratio')

plt.savefig('lz4_comp.png')