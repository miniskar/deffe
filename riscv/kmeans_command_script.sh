echo "Creating command file"
cat <<EOT >> cmd_options_file.txt
-i /home/nqx/RISCV/Benchmarks/rodinia_3.1/data/kmeans/kdd_cup
-m ${kmeans_objects}
-k 10
EOT
