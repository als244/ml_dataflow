#ifndef DATAFLOW_UTILS_H
#define DATAFLOW_UTILS_H

#include "dataflow_common.h"

void * create_zero_host_matrix(uint64_t M, uint64_t N, DataflowDatatype dt, void * opt_dest);
void * create_rand_host_matrix(uint64_t M, uint64_t N, float mean, float std, DataflowDatatype dt, void * opt_dest);
void * create_identity_host_matrix(uint64_t N, DataflowDatatype dt, void * opt_dest);
void * create_index_identity_host_matrix(uint64_t M, uint64_t N, DataflowDatatype dt, void * opt_dest);

void * load_host_matrix_from_file(char * filepath, uint64_t M, uint64_t N, DataflowDatatype orig_dt, DataflowDatatype new_dt, void * opt_dest);

int save_host_matrix(char * filename, void * mat, uint64_t M, uint64_t N, DataflowDatatype dt);

float get_home_link_speed_bytes_per_sec(unsigned int pcie_link_width, unsigned int pcie_link_gen);


#endif
