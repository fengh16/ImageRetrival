import json

try:
    file = open("images_tested.txt")
    jsonfile = json.load(file)
    file.close()
    predict = {}
    for jsondata in jsonfile:
        predict[jsondata[0]] = jsondata[1]
except:
    input("未找到images_tested.txt文件或者文件读取出错，这是运行程序所必需的数据文件，请检查后重新打开程序！\n按下回车退出……")
    raise("ERROR_No_images_tested_txt")