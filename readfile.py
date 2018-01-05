def fun():
    with open(r'D:\lrb.csv',encoding='UTF-8') as file_object:
        lines = file_object.readlines()
    return lines

def compare_len(str): #15 14
    length = len(str)
    if str.find('-',0,0) == 0: #负数
        if length <= 15:
            return 1
        else:
            return 0
    else:#正数
        if length <= 14:
            return 1
        else:
            return 0

def compare():
    lines = fun()
    num = 0
    result = []
    for line in lines:
        num += 1
        if num != 1:
            #print(line)
            templine = line.split(',')
            #print(len(templine))
            if templine[6].find('-',0,0) == -1 and len(templine[6]) >= 8 and len(templine[6]) <= 14:
                if templine[7].find('-',0,0) == -1 and len(templine[7]) >= 8 and len(templine[7]) <= 14:
                    if templine[8].find('-', 0, 0) == -1 and len(templine[8]) >= 8 and len(templine[8]) <= 14:
                        if templine[9].find('-', 0, 0) == -1 and len(templine[9]) >= 8 and len(templine[9]) <= 14:
                            xxx = len(templine)
                            if compare_len(templine[2]) == 1 and compare_len(templine[3]) == 1 and compare_len(templine[4]) == 1 and compare_len(templine[5]) == 1 and compare_len(templine[10]) == 1 and compare_len(templine[11]) == 1 and compare_len(templine[12]) == 1 and compare_len(templine[13]) == 1 and compare_len(templine[14]) == 1 \
                                    and compare_len(templine[15]) == 1 and compare_len(templine[16]) == 1 and compare_len(templine[17]) == 1 and compare_len(templine[18]) == 1 and compare_len(templine[19]) == 1 \
                                    and compare_len(templine[20]) == 1 and compare_len(templine[21]) == 1 and compare_len(templine[22]) == 1 and compare_len(templine[23]) == 1 \
                                    and compare_len(templine[24]) == 1 and compare_len(templine[25]) == 1:
                                result.append(line)
    return result
jieguo = compare()
print(len(jieguo))
for i in range(len(jieguo)):
    print(jieguo[i])