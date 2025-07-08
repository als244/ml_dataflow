ncu -o $5 --force-overwrite --set detailed --kernel-name default_rms_norm_bf16_kernel --launch-count 1 --section SchedulerStats --section WarpStateStats --kill yes ../../transformerDemo $1 $2 $3 $4
