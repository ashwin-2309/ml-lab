def sum(a, b):
    return a+b


def diff(a, b):
    return a-b


def mul(a, b):
    return a*b


def sqrt(a):
    return pow(a, 0.5)


def isPrime(a):
    limit = pow(a, 0.5)
    lim = int(limit)+1
    for i in range(2, lim, 1):
        if (a % i == 0):
            return False
    return True


print("Enter a no. to check if its prime or not\n")
x = int(input())
if (isPrime(int(x))):
    print(f"{x} is a prime number")
else:
    print(f"{x} is not a prime number")
sq = sqrt(x)
print(sq)
# isPrime function in python
