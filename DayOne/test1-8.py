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
# def apply_func(func, value):
#     return func(value)
# print(apply_func(lambda x: x ** 2, 8))  # 16

# 创建模块 mymodule.py
# mymodule.py
# def say_hello():
#     return "Hello from module!"
#
# # 主程序
# import mymodule
# print(mymodule.say_hello())
#
# # 导入第三方模块
# import requests
# response = requests.get("https://api.github.com")
# print(response.status_code)  # 200
#
# # 包使用示例
# from mypackage import mymodule

# # 定义类
# class Student:
#     def __init__(self, name, age):
#         self.name = name
#         self.age = age
#
#     def introduce(self):
#         return f"I am {self.name}, {self.age} years old."
#
# # 继承
# class GradStudent(Student):
#     def __init__(self, name, age, major):
#         super().__init__(name, age)
#         self.major = major
#
#     def introduce(self):
#         return f"I am {self.name}, a {self.major} student."
#
# # 使用
# student = Student("Alice", 20)
# grad = GradStudent("Bob", 22, "CS")
# print(student.introduce())  # I am Alice, 20 years old.
# print(grad.introduce())     # I am Bob, a CS student.

# # 简单装饰器
# def my_decorator(func):
#     def wrapper():
#         print("Before function")
#         func()
#         print("After function")
#     return wrapper
#
# @my_decorator
# def say_hello():
#     print("Hello!")
#
# say_hello()
#
# # 带参数的装饰器
# def repeat(n):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             for _ in range(n):
#                 func(*args, **kwargs)
#         return wrapper
#     return decorator
#
# @repeat(3)
# def greet(name):
#     print(f"Hi, {name}!")
#
# greet("Alice")

# # 写文件
# with open("example.txt", "w") as f:
#     f.write("Hello, Python!\n")
#
# # 读文件
# with open("example.txt", "r") as f:
#     content = f.read()
#     print(content)
#
# # 处理CSV
# import csv
# with open("data.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Name", "Age"])
#     writer.writerow(["Alice", 20])





