import os
import pandas as pd
from datetime import timedelta
import pandas as pd
import pickle


def generate_df_total_from_raw_data(
    year_month_str: str, workspace_folder_path: str = "./"
) -> None:
    """
    加载数仓获取的原始数据, 合并清洗生成所需的扫码记录主数据(df_total.parquet)写入工作区指定月份的main_data文件夹。

    Parameters
    ----------
    year_month_str : str
        年月份字符串, 以"YYYYMM"的格式。

    workspace_folder_path : str
        工作区目录的路径。默认值为当前目录。

    Returns
    -------
    None
        此函数储存生成的扫码记录主数据, 不返回任何值。

    Description
    -----------
    此函数执行完成以下功能:

    1. 加载工作目录下指定月份的原始数据文件夹 ("/data/{year_month_str}/raw_data/")中从数仓中获取的原始数据:
       - "bottlecode_{year_month_str}.csv": 包含所有瓶扫码数据。
       - "salesarea_map_{year_month_str}.csv": 包含经销商品项的销售区域映射信息。
       - "salesarea_info_{year_month_str}.csv": 包含销售区域的具体信息。

    2. 通过"DEALER_CODE", "PRODUCT_GROUP_CODE"在销售区域映射表中找到对应的销售区域("SALESAREA_ID"),
        再基于销售区域信息表找到该区域的最新有效信息("INACTIVE_DATE"为99991231), 如大区名称("ORG_REGION_NAME")等。


    3. Formate date type ["CUST_SCAN_DATE", "OUT_DEALER_DATE", "TO_STORE_DATE"], 替换99991231为20991231以便转化为datetime格式。

    4. 储存合并后的扫码记录主数据("df_total.parquet"), 写入"/data/{year_month_str}/main_data/"

    另外将发现的同一编号但不同名字的经销商，硬编码同一改成最新的名字。

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
            "OPEN_PROVINCE_ID": str,
            "OPEN_CITY_ID": str,
            "OPEN_DISTRICT_ID": str,
            "OPEN_TOWN_ID": str,
        },
        parse_dates=[
            "CUST_SCAN_DATE",
            "OPEN_BOX_TIME",
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
        "LATITUDE",
        "LONGITUDE",
        "OPEN_ADDRESS",
        "OPEN_PROVINCE",
        "OPEN_CITY",
        "OPEN_DISTRICT",
        "OPEN_TOWN",
        "OPEN_PROVINCE_ID",
        "OPEN_CITY_ID",
        "OPEN_DISTRICT_ID",
        "OPEN_TOWN_ID",
        "TO_STORE_DATE",
        "STORE_CODE",
    ]
    df_bottlecode = df_bottlecode.loc[:, bottlecode_total_cols]
    df_bottlecode["CUST_SCAN_DATE"] = df_bottlecode["CUST_SCAN_DATE"].dt.date
    df_bottlecode["OUT_DEALER_DATE"] = df_bottlecode["OUT_DEALER_DATE"].dt.date
    df_bottlecode["TO_STORE_DATE"] = df_bottlecode["TO_STORE_DATE"].dt.date

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
    df_salesarea_info = df_salesarea_info.loc[
        df_salesarea_info["INACTIVE_DATE"] == 99991231, :
    ]

    # Merge df_total
    df_total = pd.merge(
        df_bottlecode,
        df_salesarea_map,
        on=["BELONG_DEALER_NO", "PRODUCT_GROUP_CODE"],
        how="left",
    )
    df_total = pd.merge(df_total, df_salesarea_info, on="SALESAREA_ID", how="left")

    df_total["EFFECTIVE_DATE"] = df_total["EFFECTIVE_DATE"].replace(99991231, 20990101)
    df_total["INACTIVE_DATE"] = df_total["INACTIVE_DATE"].replace(99991231, 20990101)
    df_total["EFFECTIVE_DATE"] = pd.to_datetime(
        df_total["EFFECTIVE_DATE"], format="%Y%m%d"
    )
    df_total["INACTIVE_DATE"] = pd.to_datetime(
        df_total["INACTIVE_DATE"], format="%Y%m%d"
    )
    # print(df_total.shape)
    df_total.loc[df_total["BELONG_DEALER_NO"] == "D0011813", "BELONG_DEALER_NAME"] = (
        "冀州区屹鑫烟酒商行"
    )
    df_total.loc[df_total["BELONG_DEALER_NO"] == "D0014029", "BELONG_DEALER_NAME"] = (
        "馆陶县云川烟酒商行（个人独资）"
    )
    df_total.loc[df_total["BELONG_DEALER_NO"] == "D0014607", "BELONG_DEALER_NAME"] = (
        "长沙晨希酒类贸易有限公司（浓）"
    )

    df_total_file = "df_total.parquet"
    os.makedirs(main_data_folder, exist_ok=True)
    df_total_file_path = os.path.join(main_data_folder, df_total_file)
    df_total.to_parquet(df_total_file_path, index=False)


def generate_dealer_scope_dict_from_raw_data(
    year_month_str: str, workspace_folder_path: str = "./"
) -> None:
    """
    加载数仓获取的原始数据, 合并清洗生成所需的经销商品项经营范围主数据(dealer_scope_dict.pkl)并写入工作区指定月份的main_data文件夹。

    Parameters
    ----------
    year_month_str : str
        年月份字符串, 以"YYYYMM"的格式。

    workspace_folder_path : str
        工作区目录的路径。默认值为当前目录。

    Returns
    -------
    None
        此函数储存生成的经销商品项经营范围主数据, 不返回任何值。

    Description
    -----------
    此函数执行完成以下功能:

    1. 加载工作目录下指定月份的原始数据文件夹 ("/data/{year_month_str}/raw_data/")中从数仓中获取的原始数据:
       - "business_scope_{year_month_str}.csv": 包含经销商品项经营范围信息。

    2. 替换 99991231 为 20990101， 因为99991231为datetime不支持的时间。替换经营范围原始数据中"AREA_CODE", "AREA_NAME"在高德api中已变更的信息。

    3. 移除"PRODUCT_GROUP_CODE"为空的记录。将经营范围为空的栏标记为"-1"(str), 以此提示经营范围具体到哪一行政级别。

    4. 将原始数据中经销商经营范围为同一区域且时间连续的多条记录合并为一条， 生成dealer_scope_dict {("DEALER_CODE", "PRODUCT_GROUP_CODE") : df_scope}储存。
        目前的合并过程以"AREA_NAME"为判断为同一区域的主键，因为相对于"AREA_CODE", 事实为同一区域的名称改变较少。合并过程同一"AREA_NAME"但不同"AREA_CODE"的
        取最新（"EFFECTIVE_DATE"）一条数据中的"AREA_CODE", 最大避免失效"AREA_CODE"出现在主数据中。其它经过模型运行后发现的失效"AREA_NAME"和"AREA_CODE"，目前根据行政区划变动的现实情况，被硬编码替换清洗。

    # 另外剔除了一些已知的在督导那"不算数"的经营范围（Hhhh。。。）
    """

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
            "320571": "320506",  # 苏州市苏州工业园区 -> 现高德api没有此项的编码，用吴中区代替
            "320602": "320613",  # 江苏省南通市崇川区
            "410381": "410307",  # 河南省洛阳市偃师市 -> 偃师区
            "410322": "410308",  # 洛阳市孟津县 -> 洛阳市孟津区
            "330103": "330105",  # 杭州市下城区 -> 杭州市拱墅区
            "330104": "330102",  # 杭州市江干区 -> 杭州市上城区 2021年规划调整 江干区变为新的上城区 和 钱塘区
            "350402": "350404",  # 三明市梅列区 -> 三明市三元区 2021年
            "340208": "340209",  # 芜湖市三山区 -> 芜湖市弋江区 2020年
            "220181": "220184",  # 公主岭市 -> 公主岭市
            "370903": "370911",  # 泰安市岱岳区
            "340221": "340210",  # 芜湖市湾沚区
            "350625": "350605",  # 漳州市长泰县 -> 漳州市长泰区
        }
    )

    df_dealer_busiuness_scope["AREA_NAME"] = df_dealer_busiuness_scope[
        "AREA_NAME"
    ].replace(
        {
            "偃师市": "偃师区",  # 河南省洛阳市偃师市 -> 偃师区
            "孟津县": "孟津区",  # 洛阳市孟津县 -> 孟津区
            "下城区": "拱墅区",  # 杭州市下城区 -> 杭州市拱墅区
            "江干区": "上城区",  # 杭州市江干区 -> 杭州市上城区 2021年规划调整 江干区变为新的上城区 和 钱塘区
            "梅列区": "三元区",  # 三明市梅列区 -> 三明市三元区
            "三山区": "弋江区",  # 芜湖市三山区 -> 芜湖市弋江区
            "长泰县": "长泰区",  # 21年撤县设区 漳州市
        }
    )

    # 删除长沙毅鑫商贸有限公司 - 普仙曾出现过的湖南省的经营范围记录
    delete_mask = (
        (df_dealer_busiuness_scope["DEALER_CODE"] == "D0011685")
        & (df_dealer_busiuness_scope["PRODUCT_GROUP_CODE"] == "01")
        & (df_dealer_busiuness_scope["AREA_NAME"] == "湖南省")
    )
    df_dealer_busiuness_scope = df_dealer_busiuness_scope.loc[~delete_mask, :]

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


def generate_df_report_from_raw_data(
    year_month_str: str, workspace_folder_path: str = "./"
) -> None:
    """
    加载异地报备数据，生成异地报备主数据。

    """

    raw_data_folder = os.path.join(
        workspace_folder_path, f"data/{year_month_str}/raw_data/"
    )
    main_data_folder = os.path.join(
        workspace_folder_path, f"data/{year_month_str}/main_data/"
    )
    df_report_file = f"df_report_{year_month_str}.csv"

    df_report_file_path = os.path.join(raw_data_folder, df_report_file)
    df_report = pd.read_csv(
        df_report_file_path,
        dtype={
            "报备编码": str,
            "报表类型编码": str,
            "报备类型名称": str,
            "经销商编码": str,
            "跨区市场编码": str,
            "跨区市场名称": str,
            "产品编码": str,
            "品项编码": str,
        },
    )
    df_report_main_file = "df_report.parquet"
    os.makedirs(main_data_folder, exist_ok=True)
    df_report_main_file_path = os.path.join(main_data_folder, df_report_main_file)
    df_report.to_parquet(df_report_main_file_path, index=False)


def data_preprocessing_main(year_month_str: str, workspace_folder_path: str = "./"):
    """
    数据预处理模块的主函数, 将获取的原始数据清洗生成为建模所需的主数据(df_total.parquet, dealer_scope_dict.pkl)。

    Parameters
    ----------
    year_month_str : str
        年月份字符串, 以"YYYYMM"的格式。

    workspace_folder_path : str
        工作区目录的路径。默认值为当前目录。

    Returns
    -------
    None
        数据预处理部分生成两个文件在main_data文件夹下, 但此函数不返回任何值。
    """

    generate_df_total_from_raw_data(
        year_month_str, workspace_folder_path=workspace_folder_path
    )
    generate_dealer_scope_dict_from_raw_data(
        year_month_str, workspace_folder_path=workspace_folder_path
    )
    generate_df_report_from_raw_data(year_month_str, workspace_folder_path)
