# def my_abs(num):
#     if num >= 0:
#         return num
#     else:
#         return -num
# print(my_abs(-2))
def new_abs(num:int) -> int:
    if num < 0:
        return -1
    else:
        return num
print(new_abs(9))