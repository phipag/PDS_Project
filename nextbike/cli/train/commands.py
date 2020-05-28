import click
from yaspin import yaspin

from nextbike.models import DurationModel


@click.command()
@click.argument('filename', type=click.Path('rb'))
def train(filename):
    with yaspin(color='blue') as spinner:
        spinner.text = 'Training model ...\t'
        model = DurationModel()
        model.load_from_csv(filename)
        model.train()
        spinner.text = 'Model trained and saved to disk.'
        spinner.ok('✅ ')
