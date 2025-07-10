"""
    1.lambda表达式的参数可有可无
    2.函数的参数在lambda表达式中完全适用
    3.只能返回一个表达式的值
    4.直接打印lambda表达式，输出的是此lambda的内存地址
"""
# lambda 参数列表：表达式
# 使用匿名函数计算两个数字的和
print(lambda x, y: x + y)
fn = lambda x, y: x + y
print(fn(1, 2))

# 需要给某个复杂的列表排序
lst = [
    {'name':'张三','age':15},
    {'name':'张四','age':19},
    {'name':'李三','age':100}
]
lst.sort(key=lambda x:x['age'],reverse=True)
print(lst)
