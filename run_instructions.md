# How to run the automation pipeline

There are 2 different ways to run the AutoML pipeline. You can do either one of them.

## 1. Using Makefile

There is a `Makefile` with pre-defined targets.
You can follow the steps:

1. Prepare conda environment

```bash
	make env
```

2. Install dependencies

```bash
	make install
```

3. Run the pipeline
   You need to specify dataset name for this target as follows:

```bash
	make run dataset=<dataset_name>
```

e.g.

```bash
	make run dataset=emotions
	make run dataset=flowers
	make run dataset=fashion
	make run dataset=skin_cancer
```

- Make sure these dataset names are defined in `run.py` and in the `get_dataset_class` method in `src/automl/utils.py`. Otherwise pipeline will raise an exception.

- By running this command you will initiate [neps](https://github.com/automl/neps) 's hyperparameter optimization pipeline.
- This pipeline will pick and evaluate several configurations from the search space and will give you the best configuration it could found.
- Then it will train a pretrained [MobileNetV2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) model and save it.
- And this tuned model will be used for accuracy score calculation over the test data, automatically.
- It is that simple to get a tuned model for your dataset.

## 2. Manual running via command line arguments

You can also run the pipeline using command line arguments.
Plus, you can change default pipeline configurations.

1. Prepare conda environment

```bash
	conda init
	conda create -n automl-vision-env python=3.11 -y
	conda activate automl-vision-env
```

2. Install dependencies

```bash
	pip install -e .
```

3. Run the pipeline

```bash
	python run.py \
		--dataset <dataset-name>  \
		--seed 42 \
		--output-path preds-42-<dataset-name>.npy \
		--reduced-dataset-ratio 1 \
		--max-evaluations-total 18 \
		--max-epochs 15 \
		--num-process 1
```

- Notice that you can play with those options as you wish. Here are their details:

  - **`dataset`**: Name of your dataset. e.g. `emotions`, `fashion`, `flowers`, `skin_cancer`
  - **`seed`**: Seed for randomness
  - **`output-path preds`**: Output path for `.npy` file to be saved after a fine-tuned model is trained and its predictions are evaluated over test dataset.
  - **`reduced-dataset-ratio`**: The factor to reduce your dataset. Use only for development purposes. Otherwise it is `1` by default.
  - **`max-evaluations-total`**: Number of configurations that you want to explore. Default is `10`. We have used 18 most of the time.
  - **`max-epochs`**: Maximum budget that you want to assign to a configuration during the training process. Because the pipeline is using [priorband](https://automl.github.io/neps/0.12.2/api/neps/optimizers/multi_fidelity_prior/priorband/?h=priorband) as an accelerator, this value is important for it to infer to number of brackets. Default is `15`. And `eta` is `3`, hardcoded inside the function call `neps.run()` in `src/automl/automl.py`.
  - **`num-process`**: Number of processes you want to utilize while search space exploration. Neps allows parallelization. However, keep in mind that each individual process will run the same pipeline independently. This means the more processes you have, the more configurations you will explore. Each process' evaluation results will be combined together at the end of the pipeline and will be compared against each other. Default is `1`.

- Even though multiprocessing works well, we don't recommend due to the high utilization of GPU memory. At least on our machines we sometimes had `"CUDA is out of memory"` errors.

### Important Notes

1. There is another pipeline configuration called `max_cost_total` which is given fixed value of `10800` inside the function call `neps.run()` in `src/automl/automl.py`. Its unit is in seconds. This value is equivalent to 3 hours, which means your pipeline will run maximum 3 hours and won't fine tune further. The end model will be used for test prediction and accuracy score calculation.
2. Pipeline will output the configuration evaluations under the folder `result_<dataset_name>` in the root directory. If you cancel your pipeline in the middle or the execution stops due to an error, you will need to delete that folder in order to re-run the pipeline.
3. If the pipeline successfully finishes the execution you will see the folder `result_<dataset_name>` with all the configurations evaluations. You can explore those configurations by going through `.txt` files. If you run the pipeline again, it won't fine-tune a model again. Instead it will fetch the best config from those results, train a model with that config, calculate accuracy score over the test dataset.
