#!/usr/local/bin/python3 -u

import random
import sympy
p,q = sympy.randprime(0, 2**35), sympy.randprime(0, 2**35)
with open('flag.txt', 'rb') as f:
    flag = f.read().strip()
inp = int(input("Enter your number: "))
prep_flag = inp.to_bytes((inp.bit_length()+7)//8, "big")+flag+b"padding...."
prep_flag = b'\x00'*(7-(len(prep_flag)-1)%8)+prep_flag
ct = [hex((pow(int.from_bytes(prep_flag[i:i+8],"big"),2**16+1,p*q)*(p*q+i//8))%(sympy.nextprime(p*q)))[2:][-16:].zfill(16) for i in range(0, len(prep_flag), 8)]
random.shuffle(ct)
print("".join(ct))
