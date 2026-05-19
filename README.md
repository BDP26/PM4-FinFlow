# PM4-FinFlow

FinFlow is a compact project that combines Global Fishing Watch (GFW) AIS data with OBIS biodiversity observations to produce maps, aggregated datasets and models for marine activity analysis.

Table of contents

- [Quickstart](#quickstart)
- [Data layout](#data-layout)
- [Repository structure](#repository-structure)
- [Key notebook notes](#key-notebook-notes)
- [Authors & course](#authors--course)
- [License](#license)

Quickstart

--------

- Python 3.9+ recommended.
- Notebooks reference a shared data path used during development: `/mnt/shared_data/finflow/`. Adjust paths in each notebook to your local layout before running.
- Common packages used across notebooks: `duckdb`, `pandas`, `pyarrow`, `datashader`, `colorcet`, `plotly`, `pydeck`, `ray`, `gfwapiclient`.

Install (minimal):

```bash
pip install duckdb pandas pyarrow datashader colorcet plotly pydeck ray gfwapiclient
```

Data layout

-----------

- `/mnt/shared_data/finflow/gfw_raw/` — GFW monthly Parquet output (organized by year/month).
- `/mnt/shared_data/finflow/obis_raw/` — OBIS raw Parquet files organized by taxa (input to OBIS pipelines).
- `/mnt/shared_data/finflow/images/` — generated PNGs, basemap, tiles and web assets.

Repository structure

--------------------

`data_analysis/`

- `data_analysis.ipynb` — DuckDB inspection queries for GFW Parquet files (counts, schemas, aggregates).
- `obis_playground.ipynb` — interactive OBIS exploration and quick Plotly maps.
- `obis_structuring.ipynb` — restructuring pipeline: union_by_name, normalize columns, spatial-sort and write partitioned Parquet (`layer/year/month`).

`downloader/` (data acquisition)

- `GFW.ipynb` — GFW downloader using `gfwapiclient` (tiled requests, monthly aggregation, retries, Ray).
- `GFW_status.ipynb` — monitor for the Ray `GFW_Worker` actor (progress + errors).
- `OBIS.ipynb` — OBIS orchestration: fetch live inventory, run resumable chunked downloads per species (Ray workers), store Parquet per species/category.
- `MVBK.ipynb` — Movebank downloader orchestrator: query studies and download event data to Parquet (Ray parallelism).
- `ATN.ipynb` — ATN ERDDAP inventory helper: query ERDDAP and save CSV inventory.
- `Untitled.ipynb` — small GFW test notebook used for API experiments.
- example CSVs: `finflow_live_inventory.csv`, `finflow_live_inventory1.csv`.

`images/` (image generation + viewers)

- `create_basemap.ipynb` — create full-resolution base PNG from a provided TIF.
- `create_image.ipynb` — composite GFW aggregates and OBIS points on the base map using Datashader + PIL.
- `create_images_months.ipynb`, `create_images_years.ipynb` — generate monthly/yearly composite PNGs.
- `serve_images.ipynb` — helper/placeholder for serving or packaging generated images.
- `web_app/`, `web_app_monthly/` — static viewers (`index.html`, `script.js`, `style.css`) that display generated PNG overlays with Leaflet (CRS.Simple).

`map/` (visualization)

- `dynamic_vis.ipynb` — distributed rendering using Ray+Dask and Datashader (interactive bounding boxes / species filters).
- `static_vis.ipynb` — static compositing of GFW + OBIS with Datashader and Plotly.
- `tiles.ipynb` — generate PNG tiles at a zoom level from aggregated data.
- `hexa_vis.ipynb` — H3 hexagon aggregation + pydeck export.
- `vector_fields.ipynb` — compute movement vector fields and render trend arrows between two periods.
- `vis_holoviews.ipynb` — HoloViews + Datashader interactive viewer (Bokeh backend).

`model/` (training & experiments)

- `data_prep.ipynb`, `sequence_files.ipynb`, `model_training.ipynb`, `anomaly_detection.ipynb` — notebooks for sequence prep, LSTM training and anomaly experiments.
- `train_lstm.py`, `train_lstm_multi.py` — training scripts for sequence models.

Key notebook notes

------------------

- Most notebooks include a short header cell describing purpose and key variables; open the first cell to see usage notes before running.
- Several notebooks perform heavy computations and assume a cluster or sufficient local resources (Ray, Dask). Check each notebook's top cells for runtime requirements.

Authors & course

----------------

- Sebastian Brütsch
- Mika Segmüller
- Drilon Krasniqi

This project was completed as part of the Big Data Project course at ZHAW. Heavy processing and distributed runs were executed on a Ray cluster provided for the course.

License

-------

This repository is a course/research project. Reuse and adaptation are permitted for educational purposes; please contact the authors for other uses.
