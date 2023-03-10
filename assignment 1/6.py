# # data structures arrays list set dictionary tuple
import array
# inport sys
myarr = array.array("i", [1, 2, 3, 4, 5])
sizeOfArray = myarr.itemsize*len(myarr)
print("Size of array", sizeOfArray, "bytes")
print(myarr[1])
dict = {
    "name": "Ash",
    "age": "20"
}
print(dict["name"])
lst = [1, "hello", 3, 4]
print(lst)
tup = (1, 2, 3)
mySet = {"apple", "mango", "apple"}
print(mySet)
