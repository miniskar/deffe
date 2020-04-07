python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_kmeans.json -only-preloaded-data-exploration -epochs 100000 -batch-size 4096 -full-exploration -loss custom_mean_abs_exp_loss -train-test-split 0.7 -validation-split 0.23
python3 $DEFFE_DIR/framework/run_deffe.py -model-extract-dir checkpoints -config $DEFFE_DIR/example/config_kmeans.json  -only-preloaded-data-exploration -train-test-split 0.7 -validation-split 0.23 -load-train-test -loss custom_mean_abs_exp_loss
