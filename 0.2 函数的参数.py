# 形参
def test1(x, y):
    return x + y


result = test1(20, 10)
print(result)


# 默认传参
def test2(x, y, init_sum=10):
    init_sum = init_sum + x + y
    return init_sum
print(test2(10, 20))
print(test2(10, 20,100))

# 不定长传参
def test3(*args, init_sum=10):
    print(type(args))
    for i in args:
        init_sum = init_sum + i
    return init_sum
print(test3(10, 20))
print(test3(10, 20,100))
print(test3(10, 20,init_sum=100))

# 不定长关键字对的传参
def test4(init_sum=10,**kwargs):
    print(type(kwargs))
    for k,v in kwargs.items():
        init_sum = init_sum + v
    return init_sum
print(test4())
# 位置参数-->默认传参-->不定长普通参数-->不定长关键字参数
# def test5(x,y,z=10,*args,**kwargs)