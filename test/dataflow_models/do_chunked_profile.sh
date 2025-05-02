nsys profile -t cuda,nvtx,cublas,osrt --gpu-metrics-devices=all --force-overwrite true -o profiling/$1 ./test_transformer_chunked_fwd
