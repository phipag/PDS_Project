# Programming Data Science – Semester Project
 
Within the same folder as ```setup.py``` run ```pip install .``` to install the package. Use flag ```-e``` to install in development mode. In subdirectory ```notebooks``` run ```pip install ..``` to install the package. Import via ```import nextbike```. 

# Documentation
#### Table of Contents
* [Jupyter Notebooks](#jupyter-notebooks)
* [Preprocessing Package](#preprocessing-package)  
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
