#include "host_ops.h"

int set_mem_host(void * _set_mem_host_op_args){
    Set_Mem_Host_Op_Args * args = (Set_Mem_Host_Op_Args *) _set_mem_host_op_args;
    memset(args -> ptr, args -> value, args -> size_bytes);
    return 0;
}
