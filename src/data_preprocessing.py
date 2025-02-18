import os
import pandas as pd
from datetime import timedelta
import pandas as pd
import pickle



def generate_df_total_from_raw_data(year_month_str, workspace_folder_path="./"):
    """
    load raw files: bottlecode, salesarea_map, salesarea_info
    """
    raw_data_folder = os.path.join(
        workspace_folder_path, f"data/{year_month_str}/raw_data/"
    )
    main_data_folder = os.path.join(
        workspace_folder_path, f"data/{year_month_str}/main_data/"
    )

    bottlecode_file = f"bottlecode_{year_month_str}.csv"
    salesarea_map_file = f"salesarea_map_{year_month_str}.csv"
    salesarea_info_file = f"salesarea_info_{year_month_str}.csv"

    bottlecode_file_path = os.path.join(raw_data_folder, bottlecode_file)
    salesarea_map_file_path = os.path.join(raw_data_folder, salesarea_map_file)
    salesarea_info_file_path = os.path.join(raw_data_folder, salesarea_info_file)

    df_bottlecode = pd.read_csv(
        bottlecode_file_path,
        dtype={
            "BELONG_DEALER_NO": str,
            "BELONG_DEALER_NAME": str,
            "PRODUCT_GROUP_CODE": str,
        },
        parse_dates=[
            "CUST_SCAN_DATE",
            "OPEN_BOX_TIME",
            "CUST_SCAN_TIME",
            "SALE_OUT_TIME",
            "OUT_DEALER_DATE",
            "TO_STORE_DATE",
        ],
    )
    bottlecode_total_cols = [
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "BARCODE_BOTTLE",
        "BARCODE_CORNER",
        "PRODUCT_CODE",
        "PRODUCT_NAME",
        "PRODUCT_GROUP_CODE",
        "PRODUCT_GROUP_NAME",
        "SALE_OUT_TIME",
        "OUT_DEALER_NO",
        "OUT_DEALER_NAME",
        "OUT_DEALER_DATE",
        "OUT_DEALER_TO_CODE",
        "OUT_DEALER_TO_NAME",
        "CUST_SCAN_DATE",
        "OPEN_BOX_TIME",
        "CUST_SCAN_TIME",
        "LATITUDE",
        "LONGITUDE",
        "OPEN_ADDRESS",
        "OPEN_PROVINCE",
        "OPEN_CITY",
        "OPEN_DISTRICT",
        "OPEN_TOWN",
        "IF_DIFF_PLACE",
        "IS_STORE_IN",
        "TO_STORE_DATE",
        "STORE_CODE",
        "STORE_NAME",
    ]
    df_bottlecode = df_bottlecode.loc[:, bottlecode_total_cols]
    df_bottlecode["CUST_SCAN_DATE"] = df_bottlecode["CUST_SCAN_DATE"].dt.date
    df_bottlecode["OUT_DEALER_DATE"] = df_bottlecode["OUT_DEALER_DATE"].dt.date
    df_bottlecode["TO_STORE_DATE"] = df_bottlecode["TO_STORE_DATE"].dt.date
    # print(df_bottlecode.shape)

    df_salesarea_map = pd.read_csv(
        salesarea_map_file_path,
        dtype={"SALESAREA_ID": str, "DEALER_CODE": str, "PRODUCT_GROUP_CODE": str},
    )
    salesarea_map_total_cols = [
        "YEARMONTH",
        "ID",
        "SALESAREA_ID",
        "DEALER_CODE",
        "PRODUCT_GROUP_CODE",
    ]
    df_salesarea_map = df_salesarea_map.loc[:, salesarea_map_total_cols]
    df_salesarea_map = df_salesarea_map.rename(
        columns={"DEALER_CODE": "BELONG_DEALER_NO"}
    )


    df_salesarea_info = pd.read_csv(
        salesarea_info_file_path, dtype={"SALESAREA_ID": str, "ORG_REGION_NAME": str}
    )
    salesarea_info_total_cols = [
        "SALESAREA_ID",
        "EFFECTIVE_DATE",
        "INACTIVE_DATE",
        "ORG_CNTER_NAME",
        "ORG_REGION_NAME",
        "ORG_AGENCY_NAME",
        "ORG_URBAN_UNIT_NAME",
    ]
    df_salesarea_info = df_salesarea_info.loc[:, salesarea_info_total_cols]
    # print(df_salesarea_info.shape)
    df_salesarea_info = df_salesarea_info.loc[
        df_salesarea_info["INACTIVE_DATE"] == 99991231, :
    ]
    # print(df_salesarea_info .shape)



    # Merge df_total
    df_total = pd.merge(
        df_bottlecode,
        df_salesarea_map,
        on=["BELONG_DEALER_NO", "PRODUCT_GROUP_CODE"],
        how="left",
    )
    df_total = pd.merge(df_total, df_salesarea_info, on="SALESAREA_ID", how="left")

    df_total["EFFECTIVE_DATE"] = df_total[
        "EFFECTIVE_DATE"
    ].replace(99991231, 20990101)
    df_total["INACTIVE_DATE"] = df_total[
        "INACTIVE_DATE"
    ].replace(99991231, 20990101)
    df_total["EFFECTIVE_DATE"] = pd.to_datetime(
        df_total["EFFECTIVE_DATE"], format="%Y%m%d"
    )
    df_total["INACTIVE_DATE"] = pd.to_datetime(
        df_total["INACTIVE_DATE"], format="%Y%m%d"
    )
    print(df_total.shape)

    df_total_file = "df_total.parquet"
    os.makedirs(main_data_folder, exist_ok=True)
    df_total_file_path = os.path.join(main_data_folder, df_total_file)
    df_total.to_parquet(df_total_file_path, index=False)


def generate_dealer_scope_dict_from_raw_data(
    year_month_str, workspace_folder_path="./"
):

    raw_data_folder = os.path.join(
        workspace_folder_path, f"data/{year_month_str}/raw_data/"
    )
    main_data_folder = os.path.join(
        workspace_folder_path, f"data/{year_month_str}/main_data/"
    )

    business_scope_file = f"business_scope_{year_month_str}.csv"
    business_scope_file_path = os.path.join(raw_data_folder, business_scope_file)

    business_scope_dtype_dict = {
        "AREA_CODE": str,
        "PRODUCT_GROUP_CODE": str,
        "PROVINCE_CODE": str,
        "CITY_CODE": str,
        "DISTRICT_CODE": str,
        "STREET_CODE": str,
        "IS_BELONG": str,
    }
    df_dealer_busiuness_scope = pd.read_csv(
        business_scope_file_path, dtype=business_scope_dtype_dict
    )
    df_dealer_busiuness_scope = df_dealer_busiuness_scope.dropna(
        subset=["PRODUCT_GROUP_CODE"]
    )
    cols_fill = ["PROVINCE", "CITY", "DISTRICT", "STREET"]
    df_dealer_busiuness_scope[cols_fill] = df_dealer_busiuness_scope[cols_fill].fillna(
        "-1"
    )
    # print(df_dealer_busiuness_scope.dtypes)

    # 替换 99991231 为 20990101， 因为99991231为Datetime不支持的时间。
    df_dealer_busiuness_scope["EFFECTIVE_DATE"] = df_dealer_busiuness_scope[
        "EFFECTIVE_DATE"
    ].replace(99991231, 20990101)
    df_dealer_busiuness_scope["INACTIVE_DATE"] = df_dealer_busiuness_scope[
        "INACTIVE_DATE"
    ].replace(99991231, 20990101)
    df_dealer_busiuness_scope["EFFECTIVE_DATE"] = pd.to_datetime(
        df_dealer_busiuness_scope["EFFECTIVE_DATE"], format="%Y%m%d"
    )
    df_dealer_busiuness_scope["INACTIVE_DATE"] = pd.to_datetime(
        df_dealer_busiuness_scope["INACTIVE_DATE"], format="%Y%m%d"
    )

    df_dealer_busiuness_scope = df_dealer_busiuness_scope[
        [
            "DEALER_CODE",
            "PRODUCT_GROUP_CODE",
            "EFFECTIVE_DATE",
            "INACTIVE_DATE",
            "AREA_CODE",
            "AREA_NAME",
            "PROVINCE",
            "CITY",
            "DISTRICT",
            "STREET",
        ]
    ]
    # 这里用来替换已知的 经营范围area_code 与 高德api最新的adcode 不一致的内容
    df_dealer_busiuness_scope["AREA_CODE"] = df_dealer_busiuness_scope[
        "AREA_CODE"
    ].replace(
        {
        # '340203': '340209', # 芜湖市弋江区
        '320571': '320506', # 苏州市苏州工业园区 -> 现高德api没有此项的编码，用吴中区代替
        '320602': '320613', # 江苏省南通市崇川区 
        '410381': '410307', # 河南省洛阳市偃师市 -> 偃师区
        '410322': '410308', # 洛阳市孟津县 -> 洛阳市孟津区
        "330103": "330105", # 杭州市下城区 -> 杭州市拱墅区
        "330104": "330102", # 杭州市江干区 -> 杭州市上城区 2021年规划调整 江干区变为新的上城区 和 钱塘区
        '350402': '350404', # 三明市梅列区 -> 三明市三元区 2021年
        '340208': '340209', # 芜湖市三山区 -> 芜湖市弋江区 2020年
        '220181': '220184', # 公主岭市 -> 公主岭市 
        '370903': '370911', # 泰安市岱岳区
        }
    )

    df_dealer_busiuness_scope["AREA_NAME"] = df_dealer_busiuness_scope[
        "AREA_NAME"
    ].replace(
        { 
        '偃师市': '偃师区', # 河南省洛阳市偃师市 -> 偃师区
        '孟津县': '孟津区', # 洛阳市孟津县 -> 孟津区
        '下城区': '拱墅区', # 杭州市下城区 -> 杭州市拱墅区
        '江干区': '上城区', # 杭州市江干区 -> 杭州市上城区 2021年规划调整 江干区变为新的上城区 和 钱塘区
        '梅列区': '三元区', # 三明市梅列区 -> 三明市三元区
        '三山区': '弋江区', # 芜湖市三山区 -> 芜湖市弋江区
    
        }
    )

    grouped = df_dealer_busiuness_scope.groupby(["DEALER_CODE", "PRODUCT_GROUP_CODE"])
    # dealer_scope_dict = {key: group.reset_index(drop=True) for key, group in grouped}

    dealer_scope_dict = {}

    for key, df in grouped:
        target_grouped = df.groupby("AREA_NAME")
        # 合并同一范围的多条连续记录
        dfs = []
        for area_name, df_group in target_grouped:
            df_group = df_group.sort_values(by="EFFECTIVE_DATE").reset_index(drop=True)
            durations = []
            eff = df_group.EFFECTIVE_DATE
            inact = df_group.INACTIVE_DATE
            start = eff.iloc[0]
            end = inact.iloc[0]
            for i in range(len(df_group) - 1):
                j = i + 1
                if inact.iloc[i] + timedelta(days=1) == eff.iloc[j]:
                    end = inact.iloc[j]
                else:
                    durations.append((start, end))
                    start = eff.iloc[j]
                    end = inact.iloc[j]
            durations.append((start, end))

            df_new_range = pd.DataFrame(
                durations, columns=["EFFECTIVE_DATE", "INACTIVE_DATE"]
            )

            for col in [
                "DEALER_CODE",
                "PRODUCT_GROUP_CODE",
                "AREA_CODE",
                "AREA_NAME",
                "PROVINCE",
                "CITY",
                "DISTRICT",
                "STREET",
            ]:
                df_new_range[col] = df_group[col].iloc[-1]

            if not df_new_range.empty:
                dfs.append(df_new_range)

        df_key_new_range = pd.concat(dfs).reset_index(drop=True)
        dealer_scope_dict[key] = df_key_new_range

    os.makedirs(main_data_folder, exist_ok=True)
    dealer_scope_dict_file = "dealer_scope_dict.pkl"
    dealer_scope_dict_path = os.path.join(main_data_folder, dealer_scope_dict_file)
    with open(dealer_scope_dict_path, "wb") as f:
        pickle.dump(dealer_scope_dict, f)


def data_preprocessing_main(year_month_str, workspace_folder_path="./"):

    generate_df_total_from_raw_data(
        year_month_str, workspace_folder_path=workspace_folder_path
    )
    generate_dealer_scope_dict_from_raw_data(
        year_month_str, workspace_folder_path=workspace_folder_path
    )
