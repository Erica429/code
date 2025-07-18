__all__ = ['my_sum'] #声明当前模块，只能公开my_sum函数
def my_sum(n):
    s = 0
    for i in range(n):
        s = s + i
    return s


def test(n) -> int:
    if n == 0:
        return 1
    elif n == 1:
        return 1
    return n * test(n - 1)
if __name__ == "__main__": # 如果是当前py文件，就是执行入口
    print(my_sum(10))
