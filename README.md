# Programming Data Science – Semester Project
 Within the same folder as ```setup.py``` run ```pip3 install .``` to install the package. Use flag ```-e``` to install in development mode. Import via ```import nextbike```. 

# Project Report
Please access the project report via the root directory of this repository or via this link: [Report.pdf]()

# Quick Start

## Table of Contents
* [Preprocessing API](#preprocessing-api)
* [Prediction API](#prediction-api)
* [Command Line Interface (CLI)](#command-line-interface-cli)

## Preprocessing API
The Preprocessing package exports two classes, `Preprocessor` and `Transformer`. Import them as follows:
```python
from nextbike.preprocessing import Preprocessor, Transformer
```
The Preprocessor can load and clean a raw NextBike data set. Column format validation of the input data is done automatically.
```python
preprocessor = Preprocessor()
preprocessor.load_gdf() # Load the data set as geopandas GeoDataFrame
preprocessor.clean_gdf() # Clean the data set for Mannheim
```
At any point of time the current state of the data can be accessed through the `gdf` property. A `UserWarning` is raised
if the GeoDataFrame is not initialized.
```python
preprocessor.gdf
```

The Transformation class transforms the preprocessed data set to the target data format. It needs a Preprocesssor
instance. It checks automatically on instantiation if the `Preprocessor` has run successfully.
```python
transformer = Transformer(preprocessor)
```

Transform and save the data set as follows:
```python
transformer.transform()
# filename parameter is optional
transformer.save(filename='mannheim_transformed.csv')
```

## Prediction API

### Duration Prediction

#### Loading Data
Data can be loaded from a valid `Transformer` instance or from a file path. The recommended way is to use a valid `Transformer` instance, because, under the hood, the prediction sub-package also uses it to load data from a file path.

Data loading with the `Transformer`:
```python
model = DurationModel()
model.load_from_transformer(t)
```

Data loading from a file path to the raw input data:
```python
model = DurationModel()
model.load_from_csv('data/input/mannheim.csv')
```

#### Training
Training can be conducted on an instantiated `Model` instance, in this case a `DurationModel`. Please note that the methods called are standardized for all implemented models through the abstract base class `nextbike.models.Model`.

Training the model:
```python
model.train()
```

Printing the training score after prediction:
```python
model.predict() # Conduct predictions on the training data
model.training_score() # Print the training score to the console
```

#### Predict unseen Data
Prediction for unseen data can be conducted on a `Model` instance by simply calling the `predict` method with a path to the data which should be predicted. It automatically loads the previously trained model or throws an error if it does not exist.

Prediction can be conducted as follows:
```python
predictor = DurationModel() # Create a DurationModel instance
predictor.predict('data/input/mannheim_test.csv') # Predict unseen data
```

## Destination Prediction
Destination prediction works exactly the same way as duration prediction. Use the `nextbike.models.DestinationModel` instance instead of the `nextbike.models.DurationModel`. All methods are the same as for the `DurationModel`.

For example:
```python
from nextbike.models import DestinationModel
destination_model = DestinationModel()
destination_model.load_from_transformer(transformer)
...
```

## Combine both predictions into one data set
Currently, the destination and duration prediction models save to separate data sets to disk. To combine them automatically into one data set, you can use `combine_predictions()` as follows:
```python
from nextbike.io import combine_predictions()
combine_predictions()
```

## Command Line Interface (CLI)
The following CLI commands are available. Each command provides a helper text if you have problems using them.

### Transform the Raw Data
```bash
nextbike transform [--output <output-path>] <data-path>
```
### Train the Duration and Destination Model
```bash
nextbike train <data-path>
```
### Predict new Data
```bash
nextbike predict <data-path>
```
