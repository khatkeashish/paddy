from sys import platform

if platform == "linux" or platform == "linux2":
    # linux
    data_path = "/home/ash/ak/data"
else:
    data_path = "/Users/ash/ak/data"

# competition name
competition = "paddy-disease-classification"