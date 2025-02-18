from src import main_data_preprocessing
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing raw data to generate main data."
    )

    parser.add_argument(
        "year_month_str", type=str, help="Year and month in 'YYYYMM' format."
    )
    parser.add_argument(
        "--workspace", type=str, default="./", help="Path to the workspace folder."
    )

    args = parser.parse_args()

    print("starting")
    main_data_preprocessing(
        year_month_str=args.year_month_str, workspace_folder_path=args.workspace
    )

    print("ended")


if __name__ == "__main__":
    main()
