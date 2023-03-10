import array
import time

# Test list
lst = [i for i in range(1000000)]
start = time.time()
lst[1000]  # Accessing an element
end = time.time()
print("List Access Time: ", end - start)

# Test tuple
tpl = tuple(i for i in range(1000000))
start = time.time()
tpl[1000]  # Accessing an element
end = time.time()
print("Tuple Access Time: ", end - start)

# Test array
arr = array.array("i", [i for i in range(1000000)])
start = time.time()
arr[1000]  # Accessing an element
end = time.time()
print("Array Access Time: ", end - start)
