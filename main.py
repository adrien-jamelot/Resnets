import click  #
from resnets.training.train import train


@click.command()
@click.option(
    "--mode",
    default="inference",
    type=click.Choice(["inference", "training"]),
)
def main(mode):
    if mode == "training":
        train()


if __name__ == "__main__":
    main()
