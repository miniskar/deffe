python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_kmeans.json -only-preloaded-data-exploration -epochs 20000 -batch-size 4096 -full-exploration -train-test-split 0.7 -validation-split 0.23
python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_kmeans.json -icp kmeans.hdf5  -input test-input.csv -output test-output.csv -inference-only
python3 $DEFFE_DIR/framework/run_deffe.py -config $DEFFE_DIR/example/config_kmeans.json -icp kmeans.hdf5  -input ../../../../output_kmeans_deffe.csv -output test-output-full.csv -inference-only
