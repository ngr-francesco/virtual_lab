class A:
    def __init__(self,d):
        self.d = d
    def c(self):
        print("A")
def f(string):
    return exec(string, {"a" : b})
b = 2
a = f("a = 2+2")
print(b)