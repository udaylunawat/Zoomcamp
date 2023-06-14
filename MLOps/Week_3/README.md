# Steps

```bash
mkdir data
cd data
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet
```

https://github.com/discdiver/prefect-mlops-zoomcamp/blob/main/requirements.txt

conda create --name prefect-ops python=3.9.12

pip install -r requirements.txt

http://127.0.0.1:4200/api

# 3.3

```
python orchestrate_pre_prefect.py
[99]    validation-rmse:5.19931
```

```
python orchestrate.py
03:33:23.029 | INFO    | Task run 'train_best_model-0' - [99]   validation-rmse:5.19931
/opt/homebrew/Caskroom/miniconda/base/envs/prefect-ops/lib/python3.9/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
03:33:24.389 | INFO    | Task run 'train_best_model-0' - Finished in state Completed()
03:33:24.407 | INFO    | Flow run 'fragrant-carp' - Finished in state Completed('All states completed.')
```

# 3.4

```
prefect project init
```
# Answers

@task(retries=3, retry_delay_seconds=2, name="Read taxi data")

0 9 3 * *

5.19931

