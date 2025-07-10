# 局部变量：只能在函数内部使用，函数外部不能使用
# 全局变量：在整个python文件中声明，全局范围内都可以使用
a = 100 # 全局变量
print(a)
def test1():
    global a # 全局变量在函数内部声明
    a = 20
    print(a)
    b = 100
    return b**(1/2)

print(test1())
print(a)