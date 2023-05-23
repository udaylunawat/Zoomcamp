# MLOps Zoomcamp 2023

Download Jan and Feb 2022 Yellow Trip Dataset
```bash
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-02.parquet
```


```bash
conda create --name mlops_zoom python=3.9 pandas numpy jupyterlab -y
conda activate mlops_zoom
jupyter lab
```