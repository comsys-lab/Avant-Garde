# PLATFORM: 실행할 config 파일의 directory (tested-cfgs 안의 directory를 정확하게 입력해야 됨)
# BENCHMARK: 실행할 TRACE가 저장되어 있는 directory. generate_trace.sh의 BENCHMARK와 동일하게 설정해야 함.

# Simulation 결과는 /workspace/results/simulation_results/${BENCHMARK}/${PLATFORM}.txt에 저장됨

BENCHMARK=bert_inference
PLATFORM=A100


#################################

_BENCHMARK=${1:-${BENCHMARK}}
_PLATFORM=${2:-${PLATFORM}}


cd /workspace/accel-sim-dev

TRACE_FILE="/workspace/results/traces/${_BENCHMARK}/kernelslist.g"
GPGPUSIM_CONFIG_FILE="./gpu-simulator/gpgpu-sim/configs/tested-cfgs/${_PLATFORM}/gpgpusim.config"
TRACE_CONFIG_FILE="./gpu-simulator/configs/tested-cfgs/${_PLATFORM}/trace.config"
RESULT_DIR="/workspace/results/simulation_results/${_BENCHMARK}"
mkdir -p $RESULT_DIR

source ./gpu-simulator/setup_environment.sh
./gpu-simulator/bin/release/accel-sim.out \
    -trace $TRACE_FILE \
    -config $GPGPUSIM_CONFIG_FILE \
    -config $TRACE_CONFIG_FILE > ${RESULT_DIR}/${_PLATFORM}.txt

