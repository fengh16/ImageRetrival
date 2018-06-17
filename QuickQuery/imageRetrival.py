from readimage import *
from loadMoment import *
from PIL import Image
import matplotlib.pyplot as plt

testdir = "./ir/"
traindir = "./image/"

def dist(a, b):
    ans = 0
    for i in range(len(a)):
        ans += (a[i] - b[i]) * (a[i] - b[i])
    return ans

def comparef1(a):
    return a[1]

def comparef(a):
    return a[2]

testresultnames = []
def testnow(namePic, show = True):
    global testresultnames
    test_file = namePic.split("\n")[0]
    if not (test_file in predict):
        print("文件名输入错误，没有找到该文件！\n请注意文件名称输入格式，例如2.jpg，请检查后重新输入！")
        return
    test_result=[]
    for sample in predict:
        if sample[0] != 'n':
            continue
        test_result.append([sample, dist(predict[test_file], predict[sample])])
    test_result = sorted(test_result, key = comparef1)
    dis11 = (test_result[10][1])
    i = 10
    trylist = [[test_result[i][0], test_result[i][1], dist(data[test_file], data[test_result[i][0]])]]
    alreadylist = [[test_file]]
    for i in range(10):
        if (test_result[i][1] - dis11 < 0.0000000001 and test_result[i][1] - dis11 > -0.0000000001):
            trylist.append([test_result[i][0], test_result[i][1], dist(data[test_file], data[test_result[i][0]])])
        else:
            alreadylist.append(test_result[i])
    for i in range(len(test_result) - 11):
        if (test_result[11 + i][1] - dis11 < 0.0000000001 and test_result[11 + i][1] - dis11 > -0.0000000001):
            trylist.append([test_result[11 + i][0], test_result[11 + i][1], dist(data[test_file], data[test_result[11 + i][0]])])
        else:
            break
    trylist = sorted(trylist, key = comparef)
    for i in range(11 - len(alreadylist)):
        alreadylist.append(trylist[i])
    if show:
        for i in range(11):
            print(alreadylist[i])
        images_show(alreadylist)
    else:
        testresultnames = []
        for i in range(11):
            if i == 0:
                continue
            testresultnames.append(alreadylist[i][0])

def images_show(images):
    plt.subplots(num='result pictures window',figsize=(8,6))
    for i in range(11):
        if i == 0:
            try:
                image=Image.open(testdir+images[i][0])
            except:
                print("错误！文件：" + testdir+images[i][0] + "无法打开（不存在或者已损坏）")
                continue
        else:
            try:
                image=Image.open(traindir+images[i][0])
            except:
                print("错误！文件：" + traindir+images[i][0] + "无法打开（不存在或者已损坏）")
                continue
        image=image.resize([image.width*5,image.height*5])
        if i == 0:
            plt.subplot(3,5,3)
        else:
            plt.subplot(3,5,5+i)
        plt.axis('off')
        plt.imshow(image)
    plt.suptitle('result pictures')
    plt.show()

while True:
    name = input("请输入要测试的文件名称，输入z退出，输入y生成结果文件：")
    if name[0] == "z":
        break
    elif name[0] == "y":
        resultfile = open("result.txt", "w")
        record = 0
        for sample in predict:
            if sample[0] == 'n':
                continue
            record += 1
            print(str(record) + "/2000")
            resultfile.write(sample.split(".")[0] + ":")
            testnow(sample, False)
            for i in range(9):
                resultfile.write(testresultnames[i].split(".")[0] + ",")
            resultfile.write(testresultnames[9].split(".")[0] + "\n")
        resultfile.close()
        print("结果文件已经保存在result.txt")
    else:
        testnow(name)