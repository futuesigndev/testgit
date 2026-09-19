import random
secret = random.randint(1, 10)
attempts = 3
print("เกมทายเลข 1-10")
while attempts > 0:
    guess = int(input("ทายเลข: "))
    if guess == secret:
        print("ถูกต้อง!")
        break
    elif guess < secret:
        print("มากกว่านี้")
    else:
        print("น้อยกว่านี้")
    attempts -= 1
    print("เหลือโอกาส", attempts, "ครั้ง")
    if attempts == 0:
        print("หมดโอกาส คําตอบคือ", secret)