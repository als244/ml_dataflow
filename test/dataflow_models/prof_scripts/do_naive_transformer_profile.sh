nsys profile -t cuda,nvtx,cublas,osrt --gpu-metrics-devices=all --force-overwrite true -o profiling/$1 ./naive_transformer
