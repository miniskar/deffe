: "${DEFFE_EXP_DIR:=$PWD}"
: "${DEFFE_CONFIG_DIR:=$PWD}"
echo "*********************** Evaluate.sh *********************"
echo "DEFFE_EXP_DIR    = $DEFFE_EXP_DIR"
echo "DEFFE_CONFIG_DIR = $DEFFE_CONFIG_DIR"
echo "*********************************************************"

param1="1"
param2="1"
options="16"
evaluate_index="20"
python $DEFFE_EXP_DIR/evaluate.py -cwd . -param1 $param1 -param2 $param2 -options $options -evaluate_index ${evaluate_index}
