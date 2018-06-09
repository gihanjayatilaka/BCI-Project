import random



n_files = 100
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

n_len = len(data)

print(len(data))

for i in range(n_files):
    start = random.randint(0, n_len - n_min)
    end = start + n_min + random.randint(0, 500)
    end = min(end, n_len)
    print(start, end)


    fw_name = file_name + str(i) + '.txt'

    fw = open(fw_name, 'w')

    for j in range(start, end):
        # print(j)
        fw.write(data[j])

    fw.close()