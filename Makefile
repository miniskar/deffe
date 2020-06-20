pre:
	python3 -m pip install -I keras \
        tensorflow \
        torch \
        doepy \
        scikit-learn \
        xlsxwriter \
        matplotlib \
        pandas \
        pathlib \
        pydot \
        tqdm \
        commentjson \
        torchsummary  --user

test-deffe: 
	export DEFFE_DIR=$(PWD)
	$(info echo "$(DEFFE_DIR)")
	cd test ; \
	rm -f test*.txt ; sh run_deffe.sh ;  \
	diff gold_test1.txt test1.txt ; \
	diff gold_test2.txt test2.txt ; \
	diff gold_test3.txt test3.txt ; \
	cd ..

