import matplotlib.pyplot as plt

# Lists to store x and y values
x_values = []
y_values = []

# Read data from the text file
with open('/home/musong/Desktop/paper/drugfinder/validation_log.txt', 'r') as file:
    for line in file:
        elements = line.split()
        x = float(elements[0])
        y = float(elements[1])
        x_values.append(x)
        y_values.append(y)

# Create a plot
plt.plot(x_values, y_values, marker='o', linestyle='-')

# Set labels for x and y axes
plt.xlabel('x')
plt.ylabel('y')

# Set a title for the plot
plt.title('Plot of x and y')

# Show the plot
plt.show()