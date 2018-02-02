import math
import pickle
import matplotlib.pyplot as plt

with open("housing.txt") as house:
    housing = house.readlines()


def calculations(l):
    # Find the mean of the list
    mean = 0
    sd = 0
    for num in l:
        mean += num
    mean = mean/len(l)

    # Find the standard deviation of the list
    for num in l:
        sd += (num-mean) ** 2
    sd = sd/len(l)
    sd = math.sqrt(sd)

    return mean, sd

# normalize the features of input text and create new file for output
def extraction_normalization(original):
    f = open("normalized.txt", "w")
    # Pickling normalized list
    norm_list_obj = open("norm_list.pkl", "wb")

    temp_list = []
    for l in housing:
        l = l.strip()
        temp_list.append(l.split(","))

    areas = []
    rooms = []
    for i in range(len(temp_list)-1):
        areas.append(int(temp_list[i][0]))
        rooms.append(int(temp_list[i][1]))

    mean_area, sd_area = calculations(areas)
    mean_room, sd_room = calculations(rooms)

    result = []

    for i in range(len(temp_list)):
        a = (int(temp_list[i][0]) - mean_area) / sd_area
        b = (int(temp_list[i][1]) - mean_room) / sd_room
        # List of normalized area, room, and original price
        c = [a, b, int(temp_list[i][2])]
        f.write(str(c))
        result.append(c)
    pickle.dump(result, norm_list_obj)
    norm_list_obj.close()
    f.close()

extraction_normalization(housing)
