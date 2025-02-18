import pandas as pd


def generate_clean_region_data_main(
    dealer_region_name: str, df_total_path: pd.DataFrame, product_group_id: str
) -> pd.DataFrame:
    """
    args:
    """

    df_total = pd.read_parquet(df_total_path)
    print(f"总数据为 {len(df_total)} 条")
    df_total = df_total.loc[df_total["ORG_REGION_NAME"] == dealer_region_name, :]
    print(f"所选区域所有品项共有 {len(df_total)} 条")

    df_total = df_total.loc[df_total["PRODUCT_GROUP_CODE"] == product_group_id, :]
    print(f"所选区域该品项共有 {len(df_total)} 条")

    selected_cols = [
        "BARCODE_BOTTLE",
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "BARCODE_CORNER",
        "PRODUCT_GROUP_CODE",
        "PRODUCT_GROUP_NAME",
        "CUST_SCAN_DATE",
        "OPEN_ADDRESS",
        "OPEN_PROVINCE",
        "OPEN_CITY",
        "OPEN_DISTRICT",
        "OPEN_TOWN",
        "LONGITUDE",
        "LATITUDE",
        "OUT_DEALER_NO",
        "SALE_OUT_TIME",
        "OUT_DEALER_DATE",
    ]

    df_cleaned = df_total.loc[
        ~(
            df_total["OUT_DEALER_NO"].notna()
            & (df_total["OUT_DEALER_NO"] != df_total["BELONG_DEALER_NO"])
        ),
        selected_cols,
    ]

    if len(df_cleaned) == 0:
        print(f"df_cleaned 无数据。请核查所选的大区名称{dealer_region_name}和品项代号{product_group_id}")

    print(
        f"共计去除 {len(df_total) - len(df_cleaned)} 条数据 ({round(((len(df_total) - len(df_cleaned)) / len(df_total) * 100), 2)} %)，剩余 {len(df_cleaned)} 条数据。"
    )
    print(
        f'实际出库经销商缺失的数据共计{df_cleaned['OUT_DEALER_NO'].isna().sum()}, 占"df_cleaned" {round(df_cleaned['OUT_DEALER_NO'].isna().sum() / len(df_cleaned) * 100, 2)}%'
    )

    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    # print(f'清洗后数据形状为: {df_cleaned.shape}')
    # print("=" * 150)

    return df_cleaned
