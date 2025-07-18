"""
    一、继承
        Python面对对象的继承指的是多个类之间的所属关系，即子类默认继承父类的所有属性和函数
        在Python中，所有类默认继承object类，object类是顶级类或者基类
    1.单继承：默认值只继承一个父类

"""


class Animals(object):
    name = '动物'

    def say(self):
        print('动物的叫声')


class Dog(Animals):  # 子类只继承一个父类：单继承
    name = '大黄'

    def dogs(self):
        print('my best love')


p1 = Dog()
p1.say()
p1.dogs()
print(p1.name)

# 子类的对象 是子类类型，也是父类类型
print(type(p1))
# isinstance 判断对象，是否是某个类型
print(isinstance(p1, Animals))
print(isinstance(p1, Dog))
# issubclass 判断类是否是继承关系
print(issubclass(Dog, Animals))
print(issubclass(Animals, Dog))
print(issubclass(Dog, object))

"""
    二、函数的重写
    父类的函数都会被子类继承，当父类的某个函数不完全适用于子类时，就需要在子类中重写父类的这个函数，而且函数的名字必须一模一样
    Python中有一个super()函数，通过使用super()函数，在重写父类函数时，让super().调用在父类中封装的函数
"""


class Parent():
    def __init__(self, name):
        self.name = name
        print('parent的init函数被执行了')

    def say_hello(self):
        print(f'Hello, {self.name}')
        print('parent的say_hello被执行了')


print('parent的init函数被执行了')


class Son(Parent):
    def __init__(self, name, age):
        # self.name = name
        super().__init__(name)
        self.age = age
        print('Son的init函数被执行了')

    def say_hello(self):
        print('Hello, son')
        print('Son的say_hello被执行了')


p1 = Son('1', 22)
p1.say_hello()

"""
    三、多继承
    Python中是可以多继承的，继承的先后顺序是有区别的。
    MRO顺序：可以参考数据结构的遍历规则
"""
class Person(Parent):
    def __init__(self, name, *args,**kwargs):
        self.name = name
        print('person的init函数执行了')

    def test(self):
        print('person的test函数被执行了')

class Person1(Person):
    def __init__(self, name, age, *args,**kwargs):
        self.age = age
        super().__init__(name, *args, **kwargs)
        print('person1的init函数执行了')

    def test(self):
        print('person1的test函数被执行了')

class Person2(Person):
    def __init__(self, name, sex, *args,**kwargs):
        self.sex= sex
        super().__init__(name, *args, **kwargs)
        print('person2的init函数执行了')

    def test(self):
        print('person2的test函数被执行了')

class GrandPerson(Person1, Person2):
    def __init__(self, name, age, sex, *args,**kwargs):
        super().__init__(name,age,sex)
        print('GrandPerson的init函数被执行了')

print(f'MRO的序列：{GrandPerson.__mro__}')
gs = GrandPerson('张三', 22,'man')
gs.test()

"""
    四、私有属性和私有函数
    在Python中，可以为属性和函数，即设置某个属性或函数不继承给子类。甚至，不能在类的外部调用或者访问
    设置私有权限的方法:在属性名和函数名前面加上两个下划线__
    如果也想要访问和修改私有属性，在Python中，一般定义函数名get_xx 用来获取私有属性，定义set_xx用来修改私有属性值。
"""
class Person3(object):
    __name = '动物' # 私有属性(类属性)
    def __init__(self, age, *args,**kwargs):
        self.__age = age # 私有属性
    def __run(self):
        print('他十分活泼')
    def say_hello(self):
        print(f'Hello, {self.__name}')
        print(f'年龄{self.__age}')

    def set_age(self,new_age): # 通过set函数来修改私有属性
        self.__age = new_age
    def get_age(self): # get_xx 访问私有函数值
        return self.__age

class Person4(Person3):
    pass

d = Person4('18')
d.say_hello()
d.set_age('15')
d.say_hello()
print(d.get_age())
# print(d.__run)  # 报错
# Person3.__run   # 报错


"""
    五、面对对象的三个特性
    封装
    将属性和方法书写到类的里面的操作即为封装封装可以为属性和方法添加私有权限
    继承
    -子类默认继承父类的所有属性和方法.子类可以重写父类属性和方法
    多态
    传入不同的对象，产生不同的结果

"""








