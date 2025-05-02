nsys profile -t cuda,nvtx,cublas,osrt ---gpu-metrics-devices=all --gpu-metrics-devices=gh100 -force-overwrite true -o profiling/$1 ./test_transformer_chunked_fwd
