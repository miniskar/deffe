#run_deffe -no-slurm -no-train -batch-size 1 -config config.json -init-batch-samples 10 -batch-size 10 $@
run_deffe -no-slurm -no-train -batch-size 1 -config config_light_test1.json -init-batch-samples 10 -batch-size 10 $@ | grep -v seconds | tail -n +4 > test1.txt
run_deffe -no-slurm -no-train -batch-size 1 -config config_light_test2.json -init-batch-samples 10 -batch-size 10 $@ | grep -v seconds | tail -n +4 > test2.txt
run_deffe -no-slurm -no-train -batch-size 1 -config config_light_test3.json -init-batch-samples 10 -batch-size 10 $@ | grep -v seconds | tail -n +4 > test3.txt
