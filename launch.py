# /usr/bin/env python
from .cli import parse_args, launch


def main():
    """Main entrypoint
    You should call this at your own launch.py after ``add_model``
    """
    args = parse_args()
    launch(args)


if __name__ == "__main__":
    main()
