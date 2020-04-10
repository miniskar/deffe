python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_matmul_tl_samples.json -icp ../../kmeans.hdf5 -only-preloaded-data-exploration -epochs 1000 -batch-size 256 -train-test-split 1.0 -validation-split 0.23 
python3 $DEFFE_DIR/framework/run_deffe.py -model-extract-dir checkpoints -config $DEFFE_DIR/example/config_matmul.json  -only-preloaded-data-exploration -train-test-split 1.0 -validation-split 0.23 -load-train-test -loss custom_mean_abs_exp_loss -model-stats-output test-output-exploss.csv
python3 $DEFFE_DIR/framework/run_deffe.py -model-extract-dir checkpoints -config $DEFFE_DIR/example/config_matmul.json  -only-preloaded-data-exploration -train-test-split 1.0 -validation-split 0.23 -load-train-test -loss custom_mean_abs_log_loss -model-stats-output test-output-logloss.csv
python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_matmul_tl_samples.json -icp matmul.hdf5  -input test-input.csv -output test-output.csv -inference-only
python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_matmul_tl_samples.json -icp matmul.hdf5  -input ../../../../output_matmul_deffe.csv -output test-output-full.csv -inference-only
