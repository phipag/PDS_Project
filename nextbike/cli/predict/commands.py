import os

import click
from yaspin import yaspin

from nextbike.io import get_data_path
from nextbike.models import DurationModel, DestinationModel
from nextbike.preprocessing import Preprocessor, Transformer


@click.command()
@click.argument('filename', type=click.Path('rb'))
def predict(filename):
    """
    Predicts the duration of the trips specified in the given data frame and saves them to disk at
    {project_dir}/data/output
    :param filename: Path to the data frame which should be used for prediction
    :return: None
    """
    with yaspin(color='blue') as spinner:
        spinner.text = 'Conducting Pre-Processing and Transformation steps ...\t'
        preprocessor = Preprocessor()
        preprocessor.load_gdf(filename)
        preprocessor.clean_gdf()
        transformer = Transformer(preprocessor)
        transformer.transform()
        spinner.text = 'Performing duration prediction ...\t'
        duration_predictor = DurationModel()
        duration_predictor.load_from_transformer(transformer, training=False)
        duration_predictor.predict(save=True)
        spinner.text = 'Performing destination prediction ...\t'
        destination_predictor = DestinationModel()
        destination_predictor.load_from_transformer(transformer, training=False)
        destination_predictor.predict(save=True)
        spinner.text = 'Predictions performed and saved to disk at {}.'.format(os.path.join(get_data_path(), 'output'))
        spinner.ok('✅ ')
