from dotenv import load_dotenv

load_dotenv()
import os

name = os.getenv("USER_NAME")
print("My name is:", name)

for i in range(10):
    print("this is i:", i)
