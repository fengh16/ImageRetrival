import json

try:
    file = open("moment.txt")
    data = json.load(file)
    file.close()
except:
    input("未找到moment.txt文件或者文件读取出错，这是运行程序所必需的颜色矩文件，请检查后重新打开程序！\n按下回车退出……")
    raise("ERROR_No_moment_txt")