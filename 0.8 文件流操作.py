# IO流：通过流的方式允许计算机程序使用相同的方式来访问不同的输入/出源
# stream是从起源(source)到接受的(sink)的有序数据。
# 文件流：就是源或者目标都是文件的流
"""
文件流的操作
1.打开文件流
文件对象 = open(目标文件，访问方式)
2.读操作
文件对象.read()
文件对象.readlines()
文件对象.readline()
3.写操作
文件对象.write()
4.指针操作
seek(偏移量,起始位置)
tell()函数返回当前指针的位置
5.关闭操作
close()
"""
#打开文件流
f = open('new.txt', mode='r+', encoding='UTF-8')
# print(f.read(100))
# print(f.readline())
# print(f.readline())
# print(f.readline())
print(f.readlines())
f.close()
# 写操作
i = 0
wf = open('new1.txt', 'w+', encoding='UTF-8')
for i in range(3):
    print(f'当前指针位置：{wf.tell()}')
    wf.write('hello world\n')
    print(f'当前指针位置：{wf.tell()}')
wf.close()
# wf1 = open('new1.txt',mode='r+',encoding='UTF-8')
# print(wf1.read())
# wf1.close()
#指针的移动操作
#在第一个hello的后面加一个：liu
wf1 = open('new1.txt', mode='r+', encoding='UTF-8')  #在指定位置写入数据，会导致覆盖
#把指针移动到hello的后面
wf1.seek(6, 0)
#把第一个hello后面的数据读取出来
after = wf1.read()  #读完之后，指针又移动到文件的末尾
wf1.seek(6, 0)
wf1.write('liu ' + after)
wf1.close()
wf1 = open('new1.txt', mode='r+', encoding='UTF-8')  #在指定位置写入数据，会导致覆盖
print(wf1.read())
wf1.close()
#with语句：不用在最后输入命令关闭文件
with open('new1.txt', mode='r', encoding='UTF-8') as f:
    print(f.read())