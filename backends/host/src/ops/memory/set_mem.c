#include "host_ops.h"

int set_mem_host(void * _host_set_mem_args){
    Host_Set_Mem_Args * args = (Host_Set_Mem_Args *) _host_set_mem_args;
    memset(args -> ptr, args -> value, args -> size_bytes);
    return 0;
}

// should have add function here...
