import os

import click
from yaspin import yaspin

from nextbike.io import get_data_path
from nextbike.preprocessing import Preprocessor, Transformer


@click.command()
@click.argument('filename', type=click.Path('rb'))
@click.option('--output', default='mannheim_transformed.csv', help='Filename of transformed data frame file.')
def transform(filename, output):
    with yaspin(color='blue') as spinner:
        spinner.text = 'Loading data frame ...'
        preprocessor = Preprocessor()
        preprocessor.load_gdf(filename)
        spinner.write('Data frame loaded.')
        spinner.text = 'Cleaning data frame ...'
        preprocessor.clean_gdf()
        spinner.write('Data frame cleaned.')
        spinner.text = 'Translating to target format ...'
        transformer = Transformer(preprocessor)
        transformer.transform()
        spinner.write('Data frame transformed.')
        spinner.text = 'Check if valid ...'
        transformer.validate()
        spinner.write('Data frame valid.')
        spinner.text = 'Saving to {}'.format(os.path.join(get_data_path(), 'output', output))
        transformer.save(output)
        spinner.text = 'Saved to {}'.format(os.path.join(get_data_path(), 'output', output))
        spinner.ok('✅ ')
