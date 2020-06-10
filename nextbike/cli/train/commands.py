import os

import click
from yaspin import yaspin

from nextbike.io import get_data_path
from nextbike.models import DurationModel, DirectionModel
from nextbike.preprocessing import Preprocessor, Transformer


@click.command()
@click.argument('filename', type=click.Path('rb'))
def train(filename):
    """
    Trains a model based on a given data frame and saves it to disk at {project_dir}/data/output
    :param filename: Path to the data frame which should be used for training
    :return: None
    """
    with yaspin(color='blue') as spinner:
        spinner.text = 'Conducting Pre-Processing and Transformation steps ...\t'
        preprocessor = Preprocessor()
        preprocessor.load_gdf(filename)
        preprocessor.clean_gdf()
        transformer = Transformer(preprocessor)
        transformer.transform()
        spinner.text = 'Training duration model ...\t'
        duration_model = DurationModel()
        duration_model.load_from_transformer(transformer, training=True)
        duration_model.train()
        duration_model.predict()
        duration_model.training_score()
        spinner.text = 'Training destination model ...\t'
        destination_model = DirectionModel()
        destination_model.load_from_transformer(transformer, training=True)
        destination_model.train()
        destination_model.predict()
        destination_model.training_score()
        spinner.text = 'Models trained and saved to disk at {}.'.format(os.path.join(get_data_path(), 'output'))
        spinner.ok('✅ ')
