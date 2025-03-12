from src import main_generate_all_dealers_results
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Generate all dealers results (excels and htmls) . "
    )

    # Required arguments
    parser.add_argument(
        "dealer_region_name", type=str, help="Name of the dealer region."
    )
    parser.add_argument("product_group_id", type=str, help="ID of the product group.")
    parser.add_argument(
        "year_month_str", type=str, help="Year and month in 'YYYYMM' format."
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--workspace", type=str, default="./", help="Path to the workspace folder."
    )
    parser.add_argument(
        "--no_save_results",
        action="store_false",
        dest="save_results",
        help="Flag to NOT save results (default is True).",
    )

    args = parser.parse_args()

    print("starting")
    main_generate_all_dealers_results(
        dealer_region_name=args.dealer_region_name,
        product_group_id=args.product_group_id,
        year_month_str=args.year_month_str,
        workspace_folder_path=args.workspace,
        save_results=args.save_results,
    )
    print("ended")


if __name__ == "__main__":
    main()
