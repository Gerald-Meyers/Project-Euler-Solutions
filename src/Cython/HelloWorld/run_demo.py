# run_demo.py
import HelloWorld  # This imports the compiled .so file

print("--- Start of Cython Test ---")

# Call the function defined in your .pyx file
HelloWorld.say_hello_to("Gerald")

# Call the typed function
result = HelloWorld.add_numbers(50, 100)
print(f"The result of the C-typed addition is: {result}")

print("--- End of Cython Test ---")
