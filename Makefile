setup:
	pip install .

test: setup
	python3 ./tests/test.py
	
