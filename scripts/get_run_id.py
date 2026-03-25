import argparse

from src.utils.utils import get_next_run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="xgb")
    args = p.parse_args()
    print(get_next_run_id(args.model))


if __name__ == "__main__":
    main()