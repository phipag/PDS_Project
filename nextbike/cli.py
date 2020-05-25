import click

from nextbike import models


@click.command()
@click.option('--train/--no-train', default=False, help='Train the models.')
def main(train):
    if train:
        models.train()
    else:
        print('You don\'t do anything.')


if __name__ == '__main__':
    main()
