import random

def slice(file_name, n_files):
    file = open(file_name + '.txt', 'r')

    data = []
    names = []

    while (1):
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

    # print(data)
    try:
        test = data[1000:]
    except:
        print ("File not big enough")
        return

    start_adj = 1000
    n_min = 1000

    n_len = len(test)

    for i in range(n_files):

        start = start_adj + random.randint(0, n_len - start_adj - n_min)
        end = start + n_min + random.randint(0, 500)
        end = min(end, n_len)
        # print(start, end)

        fw_name = file_name + str(i) + '.txt'
        names.append(fw_name)

        fw = open(fw_name, 'w')

        for j in range(start, end):
            # print(j)
            fw.write(test[j])

        fw.close()

    return names


files = slice('sample', 10)

print(files)
