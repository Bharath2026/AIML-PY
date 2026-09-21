a = 10
b = 3

print(a + b)
print(a - b)
print(a * b)   
print(a / b)   
print(a // b)  
print(a % b)   
print(a ** b)


#comparision operator

a = 10
b = 5

print(a > b)   
print(a < b)   
print(a == b)  
print(a != b)  
print(a >= b)  
print(a <= b)


#Logical operator

a = 10

print(a > 5 and a < 20)  
print(a > 20 or a == 10) 
print(not(a > 5))


#Assignment Operator
a = 10

a += 5
print(a)   

a -= 5
print(a)   

a *= 2
print(a)



#Bitwise Operator
a = 5       
b = 3

print(a & b)   
print(a | b)   
print(a ^ b)   
print(~a)      
print(a << 1)
print(a >> 1)

#Membership Operator

name = "Python"

print("P" in name)  
print("z" not in name)

#Identity Operator

a = [1, 2, 3]
b = a

print(a is b)   
print(a is not b)   
