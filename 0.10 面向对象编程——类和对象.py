# 类：是对一系列具有相同 特征 和 行为 的事物的统称
# 对象：对象是基于类创建出来的真实存在的事物
class Car():
    # 代表当前对象（实例）本身
    def __init__(self, make, model, color, year):
        # make,model,color,year 都是对象属性
        self.make = make
        self.model = model
        self.color = color
        self.year = year

    def run(self):
        print(f'{self.make}--{self.model}--{self.color}--{self.year}')
        print("It is running")


# 对象的属性和函数
c1 = Car('BYD', '汉', '黑色', '2022')
print(c1.make)
c1.run()
c2 = Car('一汽大众', '迈腾', '黑色', '2023')
print(c2.make)
c2.run()

"""
魔术函数：在Python中，__xx__()的函数叫做魔术函数，指的是具有某种特殊功能 或者有特殊含义的函数
    1.init函数：__init__():对象初始化函数，创建函数时默认被自动调用，不需要手动调用
"""