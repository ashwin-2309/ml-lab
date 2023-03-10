import array_initialize as a

# create an instance of the my_reg class
# my_reg_instance = a.my_reg(1, 1, 0, 1, 2)
my_reg_instance = a.my_reg()
my_reg_instance.x1 = 1
my_reg_instance.x2 = 1
my_reg_instance.x3 = 0
my_reg_instance.x4 = 1
my_reg_instance.actual = 2


# call the my_linear method on the instance and print the result
print(my_reg_instance.my_linear())

# call the square_error method on the instance and print the result
print(my_reg_instance.square_error())
