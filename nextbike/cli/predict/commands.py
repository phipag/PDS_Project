import os

import click
from yaspin import yaspin

from nextbike.io import get_data_path
from nextbike.models import DurationModel, DestinationModel


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
        spinner.text = 'Performing duration prediction ...\t'
        duration_predictor = DurationModel()
        duration_predictor.predict(filename, save=True)
        spinner.text = 'Performing destination prediction ...\t'
        destination_predictor = DestinationModel()
        destination_predictor.predict(filename, save=True)
        spinner.text = 'Predictions performed and saved to disk at {}.'.format(os.path.join(get_data_path(), 'output'))
        spinner.ok('✅ ')
