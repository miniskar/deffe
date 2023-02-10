pre:
	python3 -m pip install --upgrade pip --user
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
        jsoncomment \
        torchsummary  --user

test-deffe: 
	$(info echo "$(DEFFE_DIR)")
	export DEFFE_DIR=$(PWD) ; \
	cd test ; \
	rm -f test*.txt ; sh run_deffe.sh ;  \
	diff gold_test1.txt test1.txt ; \
	diff gold_test2.txt test2.txt ; \
	diff gold_test3.txt test3.txt ; \
	cd ..

