# 需求：计算一个正整数n的阶乘
def test(n) -> int:
    """
    计算一个数字n的阶乘
    :param n:
    :return:
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1
    return n * test(n-1)


print(test(0))
print(test(5))