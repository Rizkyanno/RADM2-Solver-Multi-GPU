# RADM2-Solver-Multi-GPU

Compile : nvv mgpu.cu -o mgpu.out

Generate Profiler : nvprof -f -o test.nvp ./mgpu.out

Execute : ./mgpu.out
