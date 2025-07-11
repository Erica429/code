# map函数
print(list(map(lambda x: x ** 2, [1, 2, 3, 4, 5])))

# reduce函数
from functools import *
print(reduce(lambda x, y: x + y ** 2, [1, 2, 3, 4, 5], 1))

#案例：假如给你一个很长的字符串，统计字符串中每个单词出现的字数
str1 = 'hello world python matlab DNN NNs shallow DNN hello,java'
#一、把单词切开
str2 = str1.replace(',',' ')
lst1 = str2.split(' ')
print(lst1)
#二、把每个单词都记为一次
new_lst1 = list(map(lambda item: {item: 1}, lst1))
print(new_lst1)
#三、调用reduce函数实现相同单词的叠加
def func1(dict1, dict2):
    key = list(dict2.items())[0][0]
    value = list(dict2.items())[0][1]
    dict1[key] = dict1.get(key,0) + value
    return dict1
print(reduce(func1, new_lst1))

#filter函数
lst2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
filtered = list(filter(lambda item: item % 2 != 0, lst2))
print(filtered)

#sorted函数
lst = [
    {'name':'张三','age':15},
    {'name':'张四','age':19},
    {'name':'李三','age':100}
]
lst1 = sorted(lst, key=lambda item: item['age'], reverse=True)
print(lst1)
lst1 = str2.split(' ')
print(sorted(lst1, key=str.upper, reverse=True))
print(sorted(lst1,reverse=True))