import click

from nextbike.cli.predict import predict
from nextbike.cli.train import train
from nextbike.cli.transform import transform


@click.group()
def cli():
    """
    This function serves as the entry point for the cli and all its sub-commands.
    :return: None
    """
    pass


cli.add_command(transform)
cli.add_command(train)
cli.add_command(predict)
