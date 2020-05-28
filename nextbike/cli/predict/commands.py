import click
from yaspin import yaspin

from nextbike.models import DurationModel


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
        spinner.text = 'Performing prediction ...\t'
        predictor = DurationModel()
        predictor.predict(filename, save=True)
        spinner.text = 'Prediction performed and saved to disk.'
        spinner.ok('✅ ')
