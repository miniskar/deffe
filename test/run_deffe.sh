python3 $DEFFE_DIR/framework/run_deffe.py -no-slurm -no-train -batch-size 1 -config config.json -init-batch-samples 10 -batch-size 10 $@
python3 $DEFFE_DIR/framework/run_deffe.py -no-slurm -no-train -batch-size 1 -config config_light.json -init-batch-samples 10 -batch-size 10 $@
