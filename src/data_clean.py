import pandas as pd


def generate_clean_region_data_main(
    dealer_region_name: str,
    df_total_path: pd.DataFrame,
    product_group_id: str,
    df_report_path: pd.DataFrame,
) -> pd.DataFrame:
    """

    需要修改!!!!! 报备问题

    基于全国扫码数据, 筛选出指定大区{dealer_region_name}的指定品项{product_group_id}的数据。
    同时移除所筛选的数据中出货经销商(OUT_DEALER_NO)记录存在, 但与所属经销商(BELONG_DEALER_NO)不同的数据，
    避免这种复杂情况影响窜货预警。统一根据所属经销商(BELONG_DEALER_NO)判定每条扫码记录的归属。

    Parameters
    ----------
    dealer_region_name : str
        所选经销商大区名称。

    df_total_path : str
        数据预处理后包含所有扫码记录的df_total.parquet 的存储路径。

    product_group_id : str
        所选品项的代码。如普仙为'01'。

    df_report_path : str
        异地报备数据。

    Returns
    -------
    pd.DataFrame
        经过清理后的所选大区和品项的扫码记录。

    Examples
    --------
    >>> df_cleaned = generate_clean_region_data_main("河南大区", "/data/202412/main_data/df_total.parquet", "01")
    """

    df_total = pd.read_parquet(df_total_path)
    print(f"当月总数据为 {len(df_total)} 条")

    df_report = pd.read_parquet(df_report_path)
    set_report_ids = set(df_report["拐角码"])
    df_total = df_total.loc[~df_total["BARCODE_CORNER"].isin(set_report_ids), :]
    print(f"移除报备后共有{len(df_total)} 条")

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
        "OPEN_PROVINCE_ID",
        "OPEN_CITY_ID",
        "OPEN_DISTRICT_ID",
        "OPEN_TOWN_ID",
        "LONGITUDE",
        "LATITUDE",
        "OUT_DEALER_NO",
        "SALE_OUT_TIME",
        "OUT_DEALER_DATE",
    ]

    # df_cleaned = df_total.loc[
    #     ~(
    #         df_total["OUT_DEALER_NO"].notna()
    #         & (df_total["OUT_DEALER_NO"] != df_total["BELONG_DEALER_NO"])
    #     ),
    #     selected_cols,
    # ]
    df_cleaned = df_total.loc[:, selected_cols]

    if len(df_cleaned) == 0:
        print(
            f"df_cleaned 无数据。请核查所选的大区名称{dealer_region_name}和品项代号{product_group_id}"
        )

    print(
        f"共计去除 {len(df_total) - len(df_cleaned)} 条数据 ({round(((len(df_total) - len(df_cleaned)) / len(df_total) * 100), 2)} %)，剩余 {len(df_cleaned)} 条数据。"
    )
    print(
        f'实际出库经销商缺失的数据共计{df_cleaned['OUT_DEALER_NO'].isna().sum()}, 占"df_cleaned" {round(df_cleaned['OUT_DEALER_NO'].isna().sum() / len(df_cleaned) * 100, 2)}%'
    )

    df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)

    return df_cleaned
