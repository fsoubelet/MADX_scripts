# Copyright pyLHC/OMC-team <pylhc@github.com>

# Documentation for most of what you will see here can be found at the following links:
# for the GNU make special targets: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
# for python packaging: https://docs.python.org/3/distutils/introduction.html

# ANSI escape sequences for bold, cyan, dark blue, end, pink and red.
B = \033[1m
C = \033[96m
D = \033[34m
E = \033[0m
P = \033[95m
R = \033[31m

.PHONY : help clean

all: clean

help:
	@echo "Please use 'make $(R)<target>$(E)' where $(R)<target>$(E) is one of:"
	@echo "  $(R) clean $(E)          to recursively remove caches and jupyter checkpoints."

clean:
	@echo "Cleaning up bitecode files and python cache."
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@echo "Cleaning up pytest cache."
	@find . -type d -name '*.pytest_cache' -exec rm -rf {} + -o -type f -name '*.pytest_cache' -exec rm -rf {} + -o -type f -name 'stats.txt' -delete
	@echo "Cleaning up jupyter checkpoints."
	@find . -type d -name '*.ipynb_checkpoints' -exec rm -rf {} + -o -type f -name 'coverage.xml' -delete
	@echo "All cleaned up!\n"

# Catch-all unknow targets without returning an error. This is a POSIX-compliant syntax.
.DEFAULT:
	@echo "Make caught an invalid target! See help output below for available targets."
	@make help