"""
    Python 模块(Module)，是一个 Python 文件，以.py 结尾，模块能定义函数，类和变量，模块里也能包含可执行的代码。
    # 1.导入模块
    # 2.自定义模块
"""
# 1.导入模块

# 模块math
# # 第一种
# import math
# # 模块.函数名(对象)
# print(math.log2(8))
# print(math.log(8,2))

# 第二种 ：from 模块名字 import *
# from math import *
#
# print(log(8, 2))

# 第三种：
from math import log2,log10
print(log2(10))
print(log10(10))

# 2.自定义模块
# 调用模块
# import my_module
# print(my_module.my_sum(4))

# 3.注意事项
"""  
    1.如果使用 from ..import..或 from.. import *导入多个模块的时候，且模块内有同名功能。
    当调用这个同名功能的时候，调用到的是后面导入的模块的功能。
    2.当导入一个模块，Python解析器对模块位置的搜索顺序是
    a.当前目录
    b.如果不在当前目录，Python则搜索在shell变量PYTHONPATH下的每个目录。sys.path可以查看
    3.自己的文件名不要和已有模块名重复，否则导致模块功能无法使用   
"""
# 如果一个模块文件中有变量，当使用 from xxx import *导入时，只能导入这个列表中的元素。 但是:
# import 模块名字的方式，不起作用!
from my_module import *

print(my_sum(4))
# print(test(3)) # 报错