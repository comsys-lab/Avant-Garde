# # FP16/FP32 (같음)
# ./start_simulation.sh ubench_fp16_fp32 A100

# # FP16 (같음)
# ./start_simulation.sh ubench_fp16_fp16 A100_FP16

# # # Bfloat16 (같음)
# ./start_simulation.sh ubench_bfloat16_fp32 A100_FP8

# # FP8 (latency를 절반으로 낮춤, 같음)
# ./start_simulation.sh ubench_fp16_fp16 A100_FP8

# # MXFP8 (Conventional GPU: 명령어 12개 추가, Avant-Garde: Tensor core latency 추가)
# ./start_simulation.sh ubench_fp16_fp16 A100_MXFP8_baseline
# ./start_simulation.sh ubench_fp16_fp16 A100_MXFP8_Avant-Garde

# # INT8
# ./start_simulation.sh ubench_int8_int32 A100_INT8

# # MXINT8 (Conventional GPU: 명령어 12개 추가, Avant-Garde: Tensor core latency 추가)
# ./start_simulation.sh ubench_int8_int32 A100_MXINT8_baseline
# ./start_simulation.sh ubench_int8_int32 A100_MXINT8_Avant-Garde

# MX4 (Conventional GPU: 명령어 (16+12)개 추가, Avant-Garde: Tensor core latency + Pipeline latency 추가)
# ./start_simulation.sh ubench_int8_int32 A100_MX4_baseline
# ./start_simulation.sh ubench_int8_int32 A100_MX4_Avant-Garde

# # MX7 (Conventional GPU: 명령어 (16+12)개 추가, Avant-Garde: Tensor core latency + Pipeline latency 추가)
# ./start_simulation.sh ubench_int8_int32 A100_MX9_baseline
# ./start_simulation.sh ubench_int8_int32 A100_MX9_Avant-Garde

# # MX9 (Conventional GPU: 명령어 (16+12)개 추가, Avant-Garde: Tensor core latency + Pipeline latency 추가)
# ./start_simulation.sh ubench_int8_int32 A100_MX9_baseline
# ./start_simulation.sh ubench_int8_int32 A100_MX9_Avant-Garde

./start_simulation.sh ubench_int8_int32 HBFP4_baseline
./start_simulation.sh ubench_int8_int32 HBFP4_Avant-Garde
