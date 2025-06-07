import sys
import struct

def u16_to_bf16(u16_val):
    
    """
    Converts an unsigned 16-bit integer, whose bits are interpreted as
    a bfloat16, to its Python float representation.

    Args:
        u16_val (int): An integer. The lower 16 bits of this integer are
                       interpreted as a bfloat16 floating-point number.
                       It's expected to be in the range [0, 65535]
                       for a direct uint16_t interpretation.

    Returns:
        float: The Python float value represented by the bfloat16 bits.
    """
    # Ensure that we are only considering the lower 16 bits of the input.
    # This handles cases where u16_val might be outside the typical uint16 range.
    u16_val &= 0xFFFF

    # The bfloat16 format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    # To convert this to a standard float, we can treat its bits as the
    # most significant bits of a 32-bit single-precision float (float32).
    # This involves left-shifting the 16 bits of the bfloat16 by 16 positions,
    # effectively padding the 7-bit bfloat16 mantissa with 16 zeros to form
    # the 23-bit mantissa of a float32.
    # The sign and exponent bits of bfloat16 align with those of float32.
    #
    # bfloat16 bit pattern: S EEEEEEEE MMMMMMM
    # float32 bit pattern:  S EEEEEEEE MMMMMMM0000000000000000
    u32_val = u16_val << 16

    # Pack the 32-bit integer into 4 bytes.
    # '!I' specifies big-endian (network byte order) for an unsigned int (4 bytes).
    packed_bytes = struct.pack('!I', u32_val)

    # Unpack these 4 bytes as a single-precision float.
    # '!f' specifies big-endian for a float (4 bytes).
    # struct.unpack returns a tuple, so we take the first (and only) element.
    float_value = struct.unpack('!f', packed_bytes)[0]

    return float_value

if __name__ == "__main__":
    u16_val = int(sys.argv[1])
    print(u16_to_bf16(u16_val))


