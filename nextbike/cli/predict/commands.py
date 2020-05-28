import click
from yaspin import yaspin

from nextbike.models import DurationModel


@click.command()
@click.argument('filename', type=click.Path('rb'))
def predict(filename):
    with yaspin(color='blue') as spinner:
        spinner.text = 'Performing prediction ...\t'
        predictor = DurationModel()
        predictor.predict(filename, save=True)
        spinner.text = 'Prediction performed and saved to disk.'
        spinner.ok('✅ ')
