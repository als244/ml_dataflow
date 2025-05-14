import sys

def u16_to_fp16(u16_val):
    
    u16_val_bin = (bin(u16_val)[2:]).zfill(16)
    
    if u16_val_bin[0] == "0":
        sign = 1
    else:
        sign = -1

    exp = int(u16_val_bin[1:6], 2)

    frac = int(u16_val_bin[6:], 2)

    return sign * (2 ** (exp - 15)) * (1 + frac / 1024)



if __name__ == "__main__":
    u16_val = int(sys.argv[1])
    print(u16_to_fp16(u16_val))


