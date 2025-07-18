# 类：是对一系列具有相同 特征 和 行为 的事物的统称
# 对象：对象是基于类创建出来的真实存在的事物
class Car():
    # 代表当前对象（实例）本身
    def __init__(self, make, model, color, year):
        print('开始初始化')
        # make,model,color,year 都是对象属性
        self.make = make
        self.model = model
        self.color = color
        self.year = year

    def run(self):
        print(f'{self.make}--{self.model}--{self.color}--{self.year}')
        print("It is running")

    def __new__(cls, *args, **kwargs):
        print('创建Car的对象')
        return super().__new__(cls)

    def __str__(self):
        # str魔法函数，以后只要有print(对象),则会自动调用str函数，并且会打印str函数的返回值
        return f'{self.make}--{self.model}--{self.color}--{self.year}'


# 对象的属性和函数
c1 = Car('BYD', '汉', '黑色', '2022')
print(c1)
# print(c1.make)
# c1.run()
# c2 = Car('一汽大众', '迈腾', '黑色', '2023')
# print(c2.make)
# c2.run()
"""
魔术函数：在Python中，__xx__()的函数叫做魔术函数，指的是具有某种特殊功能 或者有特殊含义的函数
    1.init函数：__init__():对象初始化函数，创建函数时默认被自动调用，不需要手动调用
        def __new__():属于类函数
    2.str函数：当使用print输出对象时，默认打印对象的内存地址。如果类定义了__str__方法，
    那么就会打印从在这个函数中return的数据
    3.del函数：__del__()  删除函数
"""


# 类属性和实例属性
# 1.类属性：就是类所拥有的属性，他被该类的所有实例属性所共有
# 2.实例属性：就是单个对象或者实例才能拥有的属性

class Person:
    species = '人类'  #物种名字，是类属性

    def __init__(self, name, age):
        self.name = name  #对象属性，成员属性，是实例属性
        self.age = age

    def __str__(self):
        return f'{self.name}--{self.age}'


p1 = Person('战三', '18')
p2 = Person('李四', '22')
print(p1)
print(p2)
print(p1.name, p1.age)
print(Person.species, p1.species, p2.species)

# 访问属性（类属性，对象属性）
print(Person.species, p1.species, p2.species)
#修改属性
p1.name = 'zhangsan'
print(p1.name, p1.age)
#修改类属性（只有一种方法）
Person.species = 'human'
print(Person.species, p1.species, p2.species)
p1.species = '人'  #并不是修改类属性，只是增加了一个对象属性
print(Person.species, p1.species, p2.species)

# 类函数和静态函数
"""
    1.类函数：需要用装饰器@classmethod来标识其为类函数，
        对于类函数，第一个参数必须是类（当前类），一般以cls作为第一个参数，也可以命名为其他
    2.静态函数：需要用装饰器@staticmethod来进行修饰，静态方法既不需要传递类对象也不需要传递实例对象
        特点：当方法中不需要使用实例对象，也不需要用类对象时，使用定义静态函数方法
             取消不需要时的参数传递，有利于减少不必要的内存占用和性能消耗
"""


class Person1():
    def __init__(self, name, age):  #成员函数，实例函数
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name}正在吃饭')

    @classmethod
    def work(cls,other):
        print(other)
        print('每个人都要工作')
    @staticmethod
    def work2():
        print('每个人都可以跑起来')

p1 = Person1('zhang san', '18')
p1.eat()

#类函数调用方法
p1.work('abc')
Person1.work('efg')
#静态函数调用方法
p1.work2()
Person1.work2()
