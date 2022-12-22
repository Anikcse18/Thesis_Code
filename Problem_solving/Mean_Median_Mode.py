dataset = []
print("Please enter your Data: ", end="")
dataset = list(map(int, input().split()))
dataset.sort()
length_dataset = len(dataset)

print("Data Is: ",end="")
for i in range(0,length_dataset):
    print(dataset[i],end=" ")
print()


def median_val(dataset):

    mid = int(length_dataset / 2)

    if length_dataset%2==0:

        min_index_1 = dataset[mid]

        min_index_2 = dataset[mid-1]


        median = (min_index_1+min_index_2)/2


    else:

       median = dataset[mid]



    return median


def mean_val(dataset):
    tem_sum = 0
    for i in dataset:
        tem_sum = tem_sum+i

    mean = (tem_sum/length_dataset)

    return mean




prev_val = 0
temp_loc = 0
mode = False
for i in dataset:
        f_val = i
        coun = 0
        for j in dataset:

            if f_val == j:
                coun = coun+1
                temp_loc = j

        if coun > 1 and  coun>prev_val and coun<length_dataset:

            prev_val = coun
            location = temp_loc
            mode = True



median = median_val(dataset)
mean = mean_val(dataset)

if mode:

    mode_val = location
    if median == mean and median == mode_val:

        print("Median: ",median)
        print("Mean: ",mean)
        print("Mode: ",mode_val)
        print("Here, all three values are same, so the dataset is symmetric.")

    else:

        print("Median: ", median)
        print("Mean: ", mean)
        print("Mode: ", mode_val)
        print("Here the values are not equal. So, the dataset is asymmetric.")


else:

        print("Median: ", median)
        print("Mean: ", mean)
        print("Mode: No Mode")
        print("There is no mode here. So, the dataset is neither symmetric nor asymmetric")














