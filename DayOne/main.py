# # 条件语句
# score = 85
# if score >= 90:
#     print("A")
# elif score >= 60:
#     print("Pass")
# else:
#     print("Fail")
#
# # 循环语句
# for i in range(5):
#     if i == 3:
#         continue
#     print(i)
#
# # 异常处理
# try:
#     num = int(input("Enter a number: "))
#     print(100 / num)
# except ZeroDivisionError:
#     print("Cannot divide by zero!")
# except ValueError:
#     print("Invalid input!")
# finally:
#     print("Execution completed.")


# # 函数定义
# def greet(name, greeting="Hello"):
#     return f"{greeting}, {name}!"
#
# print(greet("Alice"))  # Hello, Alice!
# print(greet("Bob", "Hi"))  # Hi, Bob!

# # 可变参数
# def sum_numbers(*args):
#     return sum(args)
# print(sum_numbers(1, 2, 3, 4))  # 10

# # 匿名函数
# double = lambda x: x * 2
# print(double(5))  # 10
#
# 高阶函数
def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x ** 2, 8))  # 16

