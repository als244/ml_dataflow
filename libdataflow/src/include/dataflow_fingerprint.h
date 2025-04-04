#ifndef DATAFLOW_FINGERPRINT_H
#define DATAFLOW_FINGERPRINT_H

#include "dataflow_common.h"

#include <openssl/sha.h>
#include <openssl/md5.h>
#include <openssl/evp.h>


int dataflow_do_fingerpinting(void * data, uint64_t num_bytes, uint8_t * ret_fingerprint);

#endif