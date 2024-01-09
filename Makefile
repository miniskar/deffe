pre:
	python3 -m pip install --upgrade pip
	python3 -m pip install -I .

test-deffe: 
	cd test ; \
	rm -f test*.txt ; sh run_deffe.sh ;  \
	diff gold_test1.txt test1.txt ; \
	diff gold_test2.txt test2.txt ; \
	diff gold_test3.txt test3.txt ; \
	cd ..

