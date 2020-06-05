#! /bin/bash
echo "HostName: `hostname`"
echo "GEM5-McPat working directory: ${PWD}"
if [ -f "app.source" ]; then
source app.source
fi
export RUN_DIR=/home/nqx/RISCV
if [[ -z "${GEM5_DIR}" ]]; then
export GEM5_DIR=${RUN_DIR}/gem5
fi
if [[ -z "${APP_BINARY}" ]]; then
echo "[Warning] Environment variable APP_BINARY is not set"
export APP_BINARY=$RUN_DIR/Benchmarks/mt-all-matmul/build.64.newlib/RISCVProject
else
echo "Environment variable set for APP_BINARY:${APP_BINARY}"
fi
if [[ -z "${APP_OPTS}" ]]; then
echo "[Warning] Environment variable APP_OPTS is not set"
export APP_OPTS="cmd_options_file.txt"
else
echo "Environment variable set for APP_OPTS:${APP_OPTS}"
fi
echo "Application binary: ${APP_BINARY}"
echo "Application options: ${APP_OPTS}"

binary=${APP_BINARY}
commandline_option=${APP_OPTS}
if [ -f "cmd_options_file.txt" ]; then
commandline_option=`cat cmd_options_file.txt`
fi
l1d_ways=2
l1i_ways=2
l2_ways=2
l1d_size=16
l1i_size=16
l2_size=128
cacheline=64
$GEM5_DIR/build/RISCV/gem5.opt \
    $GEM5_DIR/configs/example/se.py \
    --cmd=${binary} \
    --cpu-type=TimingSimpleCPU \
    --options="${commandline_option}" \
    --output=sim.out --errout=sim.err \
    --mem-size=8GB --mem-type=DDR4_2400_8x8 \
    --caches --l2cache \
    --l1d_assoc=${l1d_ways} --l1d_size=${l1d_size}kB \
    --l1i_assoc=${l1i_ways} --l1i_size=${l1i_size}kB \
    --l2_assoc=${l2_ways} --l2_size=${l2_size}kB \
    --cacheline_size=${cacheline} | tee output-gem5.log
