SCRIPT_FILE=${BASH_SOURCE[0]}
#SCRIPT=$(readlink -f $SCRIPT_FILE)
SCRIPT=$(python3 -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" ${SCRIPT_FILE})
export TEST_DIR=$(dirname -- ${SCRIPT})/../..
echo "*********************** Deffe *********************"
echo "TEST_DIR: $TEST_DIR"
echo "***************************************************"

param1="1"
param2="1"
options="16"
evaluate_index="20"
python $TEST_DIR/evaluate.py -cwd . -param1 $param1 -param2 $param2 -options $options -evaluate_index ${evaluate_index}

