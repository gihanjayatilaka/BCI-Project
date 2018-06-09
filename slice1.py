import random



n_test_files = 100
n_train_files = 1000

n_test = 2000
n_min = 1000

file_name = 'sample'
file = open(file_name + '.txt', 'r')

data = []

while(1):
    line = file.readline()
    dat = line.strip().split()
    try:
        num = int(dat[0].strip(','))
        data.append(line)
        # index =
    except:
        pass
    if (line == ""):
        break
    # data.append(file.readline())

print(data)

test = data[-n_test:]
train = data[:-n_test]

print('train', train)
print('test', test)

n_len = len(data)
len_train = len(train)
len_test = len(test)

print(len(data))

for i in range(n_test_files):
    start = random.randint(0, len_train - n_min)
    end = start + n_min + random.randint(0, 500)
    end = min(end, len_train)
    # print(start, end)

    fw_name = file_name + str(i) + 'train.txt'

    fw = open(fw_name, 'w')

    for j in range(start, end):
        # print(j)
        fw.write(train[j])

    fw.close()

for i in range(n_train_files):
    start = random.randint(0, len_test - n_min)
    end = start + n_min + random.randint(0, 500)
    end = min(end, len_test)
    # print(start, end)

    fw_name = file_name + str(i) + 'test.txt'

    fw = open(fw_name, 'w')

    for j in range(start, end):
        # print(j)
        fw.write(test[j])

    fw.close()