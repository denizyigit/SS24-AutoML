ENV_NAME = automl-vision-env
CONDA_BASE = $(shell conda info --base)

.PHONY: env

env:
	@echo "Checking if conda environment '$(ENV_NAME)' exists..."
	@source "$(CONDA_BASE)/etc/profile.d/conda.sh" && \
	if [ $$(conda info --envs | grep -c "$(ENV_NAME)") -eq 0 ]; then\
		echo "Creating conda environment '$(ENV_NAME)'..."; \
		conda create -n $(ENV_NAME) python=3.11 -y; \
	else \
		echo "Conda environment '$(ENV_NAME)' already exists."; \
	fi
	@echo "Activating conda environment '$(ENV_NAME)'..."
	@source "$(CONDA_BASE)/etc/profile.d/conda.sh" && \
	conda activate $(ENV_NAME) && \
	echo "Environment is ready"

install:
	@echo "Activating conda environment '$(ENV_NAME)'..."
	@source "$(CONDA_BASE)/etc/profile.d/conda.sh" && \
	conda activate $(ENV_NAME) && \
	echo "Environment is ready"

	@echo "Installing packages..."
	pip install -e .

run:
	python run.py --dataset emotions --seed 42 --output-path preds-42-emotions.npy --reduced-dataset-ratio 0.1

SHELL := /bin/bash

# install:
# 	pip install -e .

# test:
# 	python -m pytest tests

# check:
# 	pycodestyle --max-line-length=120 src
# 	@echo "All good!"

# clean-tex:
# 	rm *.log *.aux *.out

# tex:
# 	pdflatex *.tex
# 	make clean-tex


# .PHONY: install test check all clean-tex tex help
# .DEFAULT_GOAL := help