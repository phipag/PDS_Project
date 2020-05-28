import click
from yaspin import yaspin

from nextbike.models import DurationModel


@click.command()
@click.argument('filename', type=click.Path('rb'))
def train(filename):
    """
    Trains a model based on a given data frame and saves it to disk at {project_dir}/data/output
    :param filename: Path to the data frame which should be used for training
    :return: None
    """
    with yaspin(color='blue') as spinner:
        spinner.text = 'Training model ...\t'
        model = DurationModel()
        model.load_from_csv(filename)
        model.train()
        spinner.text = 'Model trained and saved to disk.'
        spinner.ok('✅ ')
