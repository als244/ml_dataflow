#include "dataflow_common.h"

float solo_bf16_to_fp32(uint16_t a) {
	// bf16 is stored in the upper 16 bits of a float.
	uint32_t bits = ((uint32_t)a) << 16;
	float f;
	memcpy(&f, &bits, sizeof(f));
	return f;
}

uint16_t solo_fp32_to_bf16(float f) {
	uint32_t bits;
	memcpy(&bits, &f, sizeof(bits));
	// Round-to-nearest: add 0x8000 (1 << 15) before truncating.
	bits += 0x8000;
	uint16_t b = (uint16_t)(bits >> 16);
	return b;
}

float solo_fp16_to_fp32(uint16_t h) {
	uint32_t sign = (h & 0x8000) << 16;
	uint32_t exp  = (h >> 10) & 0x1F;
	uint32_t mant = h & 0x03FF;
	uint32_t f;

	if(exp == 0) {
		if(mant == 0) {
			// Zero.
			f = sign;
		} else {
			// Subnormal number; normalize it.
			while ((mant & 0x0400) == 0) {
				mant <<= 1;
				exp--;
			}
			exp++;               // Adjust exponent (it was decremented one time too many)
			mant &= ~0x0400;     // Clear the leading 1 that was shifted out
			f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
		}
	} else if(exp == 31) {
		// Inf or NaN.
		f = sign | 0x7F800000 | (mant << 13);
	} else {
		// Normalized number.
		f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
	}

	float ret;
	memcpy(&ret, &f, sizeof(ret));
	return ret;
}

uint16_t solo_fp32_to_fp16(float f) {
	uint32_t x;
	memcpy(&x, &f, sizeof(x));

	uint16_t sign = (x >> 16) & 0x8000;
	int32_t exp   = ((x >> 23) & 0xFF) - 127 + 15;
	uint32_t mant = x & 0x007FFFFF;
	uint16_t h;

	if(exp <= 0) {
		// Handle subnormals and zeros.
		if(exp < -10) {
			// Too small becomes zero.
			h = sign;
		} else {
			// Subnormal: add the implicit 1 and shift right.
			mant = (mant | 0x00800000) >> (1 - exp);
			// Rounding: add 0x00001000 (if bit 12 is set) for round-to-nearest.
			if(mant & 0x00001000)
				mant += 0x00002000;
			h = sign | (mant >> 13);
		}
	} else if(exp >= 31) {
		// Overflow: return infinity.
		h = sign | 0x7C00;
	} else {
		// Normalized number.
		// Rounding.
		if(mant & 0x00001000)
			mant += 0x00002000;
		h = sign | (exp << 10) | (mant >> 13);
	}

	return h;
}

uint16_t solo_fp16_to_bf16(uint16_t h) {
    // Decode fp16: 1 sign bit, 5 exponent bits, 10 mantissa bits.
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    uint32_t fbits;

    if (exp == 0) {
        if (mant == 0) {
            // Zero.
            fbits = sign;
        } else {
            // Subnormal number: normalize it.
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++; // Adjust exponent (one too many decrement)
            mant &= ~0x0400; // Clear the shifted-out leading 1
            fbits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN.
        fbits = sign | 0x7F800000 | (mant << 13);
    } else {
        // Normalized number.
        fbits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    // Rounding: add 0x8000 for round-to-nearest before truncating to 16 bits.
    fbits += 0x8000;
    return (uint16_t)(fbits >> 16);
}


uint16_t solo_bf16_to_fp16(uint16_t a) {
    // Reconstruct the 32-bit float bits from bf16.
    uint32_t x = ((uint32_t)a) << 16;
    uint16_t sign = (x >> 16) & 0x8000;
    // Convert float exponent to fp16 exponent: subtract float bias (127) and add half bias (15)
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x007FFFFF;
    uint16_t h;

    if (exp <= 0) {
        // Handle subnormals (or zero).
        if (exp < -10) {
            // Underflow: too small becomes zero.
            h = sign;
        } else {
            // Subnormal: add the implicit 1 and shift right accordingly.
            mant = (mant | 0x00800000) >> (1 - exp);
            // Rounding: if bit 12 is set, round up.
            if (mant & 0x00001000)
                mant += 0x00002000;
            h = sign | (mant >> 13);
        }
    } else if (exp >= 31) {
        // Overflow: return infinity.
        h = sign | 0x7C00;
    } else {
        // Normalized number.
        if (mant & 0x00001000)
            mant += 0x00002000;
        h = sign | (exp << 10) | (mant >> 13);
    }
    return h;
}