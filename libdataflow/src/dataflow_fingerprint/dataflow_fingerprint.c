#include "dataflow_fingerprint.h"

typedef enum fingerprint_type{
	SHA256_HASH = 0, // 32 bytes
	SHA512_HASH = 1, // 64 byte
	SHA1_HASH = 2, // 20 bytes
	MD5_HASH = 3, // 16 bytes
	BLAKE3_HASH = 4, // 32 bytes
} FingerprintType;

void print_hex(uint8_t * fingerprint, int num_bytes){
	for (int i = 0; i < num_bytes; i++){
		printf("%02x", fingerprint[i]);
	}
	printf("\n");
}

void print_sha256(uint8_t * fingerprint){
	// sha256 = 256 bits = 32 bytes
	int num_bytes = 32;
	for (int i = 0; i < num_bytes; i++){
		printf("%02x", fingerprint[i]);
	}
	printf("\n");
}

uint64_t fingerprint_to_least_sig64(uint8_t * fingerprint, int fingerprint_num_bytes){
	uint8_t * least_sig_start = fingerprint + fingerprint_num_bytes - sizeof(uint64_t);
	uint64_t result = 0;
    for(int i = 0; i < 8; i++){
        result <<= 8;
        result |= (uint64_t)least_sig_start[i];
    }
    return result;
}

uint8_t get_fingerprint_num_bytes(FingerprintType fingerprint_type){
	switch(fingerprint_type){
		case SHA256_HASH:
			return 32;
		case SHA512_HASH:
			return 64;
		case SHA1_HASH:
			return 20;
		case MD5_HASH:
			return 16;
		case BLAKE3_HASH:
			fprintf(stderr, "Error: blake3 is not supported yet\n");
			return 0;
		default:
			fprintf(stderr, "Error: fingerprint_type not supported\n");
			return 0;
	}
}

char * get_fingerprint_type_name(FingerprintType fingerprint_type){
	switch(fingerprint_type){
		case SHA256_HASH:
			return "SHA256";
		case SHA512_HASH:
			return "SHA512";
		case SHA1_HASH:
			return "SHA1";
		case MD5_HASH:
			return "MD5";
		case BLAKE3_HASH:
			return "BLAKE3";
		default:
			return "UNKNOWN";
	}
}


// These are "deprecated" as of OpenSSL 3.0, but they are faster and simpler...

// should figure out how to have global contexts to avoid overhead because doing this repeatedly...
void do_fingerprinting_sha256(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint){
	
	SHA256_CTX ctx;
	SHA256_Init(&ctx);
	SHA256_Update(&ctx, data, num_bytes);
	SHA256_Final(ret_fingerprint, &ctx);

 	return;
}


// should figure out how to have global contexts to avoid overhead because doing this repeatedly...
void do_fingerprinting_sha512(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint){
	
	SHA512_CTX ctx;
	SHA512_Init(&ctx);
	SHA512_Update(&ctx, data, num_bytes);
	SHA512_Final(ret_fingerprint, &ctx);

 	return;
}

// should figure out how to have global contexts to avoid overhead because doing this repeatedly...
void do_fingerprinting_sha1(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint){
	
	SHA_CTX ctx;
	SHA1_Init(&ctx);
	SHA1_Update(&ctx, data, num_bytes);
	SHA1_Final(ret_fingerprint, &ctx);

 	return;
}

// should figure out how to have global contexts to avoid overhead because doing this repeatedly...
void do_fingerprinting_md5(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint){
	
	MD5_CTX ctx;
	MD5_Init(&ctx);
	MD5_Update(&ctx, data, num_bytes);
	MD5_Final(ret_fingerprint, &ctx);

 	return;
}


// should figure out how to have global contexts to avoid overhead because doing this repeatedly...
// THE "New"/"Supported" interface, but 30% slower...
void do_fingerprinting_evp(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint){
	
	// using this function instead of on stack for compatibility...
	EVP_MD_CTX *mdctx = EVP_MD_CTX_new();

 	// can do switching here based on fingerprint type..\n
 	const EVP_MD *md = EVP_sha256();

 	// declare using sha256
 	EVP_DigestInit_ex(mdctx, md, NULL);
 	
 	// acutally perform the hashing stored in the context
 	EVP_DigestUpdate(mdctx, data, num_bytes);
 	
 	// copy from context to destination specified as argument
 	unsigned int fingerprint_len;
 	EVP_DigestFinal_ex(mdctx, ret_fingerprint, &fingerprint_len);
 	
 	// reset context
 	EVP_MD_CTX_free(mdctx);

 	return;
}


// ASSUMING THAT RET_FINGERPRINT HAS PRE-ALLOCATED MEMORY!
void dataflow_do_fingerprinting(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint){

	// using functions from OpenSSL's libcrypto
	switch(DATAFLOW_OP_IDENTIFIER_FINGERPRINT_TYPE){
		case SHA256_HASH:
			do_fingerprinting_sha256(data, num_bytes, ret_fingerprint);
			break;
		case SHA512_HASH:
			do_fingerprinting_sha512(data, num_bytes, ret_fingerprint);
			break;
		case SHA1_HASH:
			do_fingerprinting_sha1(data, num_bytes, ret_fingerprint);
			break;
		case MD5_HASH:
			do_fingerprinting_sha1(data, num_bytes, ret_fingerprint);
			break;
		case BLAKE3_HASH:
			fprintf(stderr, "Error: blake3 is not supported yet\n");
			break;
		default:
			fprintf(stderr, "Error: fingerprint_type not supported\n");
			break;
	}
	return;
}