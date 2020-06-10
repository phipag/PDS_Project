# Programming Data Science – Semester Project
 Within the same folder as ```setup.py``` run ```pip3 install .``` to install the package. Use flag ```-e``` to install in development mode. Import via ```import nextbike```. 

# Project Report
[report.pdf]()

# Quick Start
#### Table of Contents
* [Jupyter Notebooks](#jupyter-notebooks)
* [Preprocessing Package](#preprocessing-package)
* [Command Line Interface (CLI)](#command-line-interface-cli)

## Jupyter Notebooks
Install the `requirements.txt` via `pip` to use the Jupyter Notebooks of the `notebooks` directory.
```shell script
pip3 install -r requirements.txt
```

## Preprocessing Package
The Preprocessing package exports to classes. Import them as follows:
```python
from nextbike.preprocessing import Preprocessor, Transformer
```
The Preprocessor can load and clean a raw NextBike data set.
```python
preprocessor = Preprocessor()
preprocessor.load_gdf() # Load the data set as GeoDataFrame
preprocessor.clean_gdf() # Clean the data set for Mannheim
```
At any point of time the current state of the data can be accessed through the `gdf` property. A `UserWarning` is raised
if the GeoDataFrame is not initialized.
```python
preprocessor.gdf
```
The Transformation class transforms the preprocessed data set to the target data format. It needs a Preprocesssor
instance.

The Transformer only takes a Preprocessor instance containing a valid preprocessed data set.
```python
transformer = Transformer(p)
```
Transform and save the data set as follows:
```python
transformer.transform()
# filename parameter is optional
transformer.save(filename='mannheim_transformed.csv')
```
# Prediction Process
## Duration Prediction
### Loading Data
From Transformer Objects
```python
model = DurationModel()
model.load_from_transformer(t)
```
From a raw .csv file
```python
model = DurationModel()
model.load_from_csv('data/input/mannheim.csv')
```
### Training
```python
model.train()
```
Calculate Training Score
```python
model.predict()
model.training_score()
```
### Predict unseen Data
```python
predictor = DurationModel()
predictor.predict('data/input/mannheim.csv')
```
## Destination Prediction
### Train
```python
destinations = DestinationModel()
destinations.load_from_transformer(t)

destinations.train()
```
Calculate Training Score
```python
destinations.predict()
destinations.training_score()
```
### Predict unseen Data
```python
destination_predictor = DestinationModel()
destination_predictor.predict('data/input/mannheim.csv')
```

### Combine Predictions and Save
```python
combine_predictions()
```

# Command Line Interface (CLI)
The following CLI commands are available. Each command provides a helper text if you have problems using them.
## Transform the Raw Data
```bash
nextbike transform [--output <output-path>] <data-path>
```
## Train the Duration and Destination Model
```bash
nextbike train <data-path>
```
## Predict new Data
```bash
nextbike predict <data-path>
```
