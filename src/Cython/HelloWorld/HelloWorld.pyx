def say_hello_to(name):
    print(f"Hello {name}, this is running via compiled C code!")

# Example of a typed function for performance (C-level integers)
cpdef int add_numbers(int a, int b):
    return a + b