"""
    高阶函数：把函数作为参数传入，或者返回值是另外一个函数，这样的函数称为高阶函数，高阶函数是函数式编程的体现
    函数式编程就是指这种高度抽象的编程范式
    函数式编程大量使用函数，减少了代码的重复，因此程序比较短，开发速度比较快
"""
# 一、函数的参数是函数
# 对任意两个数字，整理之后再求和
def sum_num(a,b,f):
    """

    :param a:
    :param b:
    :param f: 就是对两个数字进行整理的函数
    :return:
    """
    return f(a) + f(b)
print(sum_num(1,-2,lambda x:x))
print(sum_num(1,-2,lambda x:x**2))

# 二、函数的返回值是函数
def test(*args):
    def sum_num():
        sum = 0
        for x in args:
            sum += x
        return sum
    return sum_num


print(test(1, 3, 23, 45, 34)())