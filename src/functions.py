from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import json
import os
import pandas as pd
import requests
import time
import yaml


def get_address_from_lat_lon(location, config_file_path):

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get("gaode_api_key")

    url = "https://restapi.amap.com/v3/geocode/regeo?parameters"
    params = {
        "key": gaode_api_key,
        "location": location,  # 经度在前，纬度在后 （lon, lat)
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def get_adcode(config_file_path, area_name, sleep=0):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get("gaode_api_key")

    url = "https://restapi.amap.com/v3/geocode/geo?parameters"

    params = {
        "key": gaode_api_key,
        "address": area_name,
        "city": area_name,
    }

    try:
        if sleep:
            time.sleep(sleep)
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def get_region_polyline(config_file_path, adcode, sleep=0):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get("gaode_api_key")

    url = "https://restapi.amap.com/v3/config/district?parameters"
    params = {
        "key": gaode_api_key,
        "keywords": adcode,
        "subdistrict": 0,
        "filter": adcode,
        "extensions": "all",
    }

    try:
        if sleep:
            time.sleep(sleep)
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None


def get_polyline_points(config_file_path, adcode, sleep=0):
    if len(adcode) > 6:
        adcode = adcode[:6]  # 替换低于区一级 的行政代码。高德api 最低只有区一级的行政边界
    data = get_region_polyline(config_file_path, adcode)

    if not data["districts"]:
        print(f"adcode 变化： {adcode}")
        return []
    
    polyline = data["districts"][0]["polyline"]
    # 这里可能会出现 data["districts"] 为空，原因是由于adcode与api的最新不符合 ######################
    polyline_points_list = []
    # if '|' in polyline:
    polylines = polyline.split("|")

    for polyline in polylines:
        polyline = polyline.split(";")
        polyline_points = []
        for i, coordinate in enumerate(polyline):
            coordinate = coordinate.split(",")
            coordinate[0], coordinate[1] = float(coordinate[1]), float(
                coordinate[0]
            )  # folium (lat, lon)

            # polyline_points.append((coordinate[0], coordinate[1]))
            polyline_points.append(coordinate)
        polyline_points_list.append(polyline_points)
    return polyline_points_list


def find_valid_regions(dealer_id, product_group_id, query_date, dealer_scope_dict):

    # 转换为 datetime 对象
    query_date = datetime.strptime(query_date, "%Y-%m-%d")

    is_archive = 0  # if (dealer_id, product_group_id) 在当前经销商合同范围表
    if (dealer_id, product_group_id) in dealer_scope_dict:
        is_archive = 1
        df_dealer_scope = dealer_scope_dict[(dealer_id, product_group_id)]
        df_valid_region = df_dealer_scope[
            (query_date >= df_dealer_scope["EFFECTIVE_DATE"])
            & (query_date <= df_dealer_scope["INACTIVE_DATE"])
        ]

        if not df_valid_region.empty:
            df_scope = df_valid_region[
                ["AREA_CODE", "AREA_NAME", "PROVINCE", "CITY", "DISTRICT", "STREET"]
            ]
            return df_scope.reset_index(drop=True), is_archive

        return (
            pd.DataFrame(
                columns=[
                    "AREA_CODE",
                    "AREA_NAME",
                    "PROVINCE",
                    "CITY",
                    "DISTRICT",
                    "STREET",
                ]
            ),
            is_archive,
        )
    return (
        pd.DataFrame(
            columns=["AREA_CODE", "AREA_NAME", "PROVINCE", "CITY", "DISTRICT", "STREET"]
        ),
        is_archive,
    )


def find_equivalent_regions(dealer_id, product_group_id, query_date_str, dealer_scope_dict, margin_months=6):
    query_date = datetime.strptime(query_date_str, "%Y-%m-%d")

    # Subtract months using relativedelta
    target_date = query_date - relativedelta(months=margin_months)

    is_archive = 0  # if (dealer_id, product_group_id) 在当前经销商合同范围表
    if (dealer_id, product_group_id) in dealer_scope_dict:
        is_archive = 1
        df_dealer_scope = dealer_scope_dict[(dealer_id, product_group_id)]
        df_equivalent_region = df_dealer_scope.loc[
            df_dealer_scope["INACTIVE_DATE"] >= target_date, :
        ]
        df_equivalent_region = df_equivalent_region.sort_values(by='INACTIVE_DATE', ascending=False).drop_duplicates(subset='AREA_NAME', keep='first')

        if not df_equivalent_region.empty:
            df_scope = df_equivalent_region[
                ["AREA_CODE", "AREA_NAME", "PROVINCE", "CITY", "DISTRICT", "STREET"]
            ]
            return df_scope.reset_index(drop=True), is_archive

        return (
            pd.DataFrame(
                columns=[
                    "AREA_CODE",
                    "AREA_NAME",
                    "PROVINCE",
                    "CITY",
                    "DISTRICT",
                    "STREET",
                ]
            ),
            is_archive,
        )
    return (
        pd.DataFrame(
            columns=["AREA_CODE", "AREA_NAME", "PROVINCE", "CITY", "DISTRICT", "STREET"]
        ),
        is_archive,
    )




def get_month_start_end(year_month_str):
    # 解析年份和月份
    year = int(year_month_str[:4])
    month = int(year_month_str[4:])

    # 获取月份的第一天
    start_date = datetime(year, month, 1)

    # 计算下个月的第一天，然后减去一天得到本月最后一天
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def save_model_outputs(
    df_dealer_results,
    df_total_scanning_locations,
    df_total_centroids,
    df_suspicious_hotspots_parameters,
    year_month_str,
    dealer_region_name,
    product_group_id,
    output_path,
    dense_model=False,
):

    output_month_path = os.path.join(
        output_path, f"{dealer_region_name}/{product_group_id}/{year_month_str}/"
    )
    os.makedirs(output_month_path, exist_ok=True)

    file_names = [
        "df_dealer_results.pkl",
        "df_total_centroids.pkl",
        "df_suspicious_hotspots_parameters.parquet",
        "df_total_scanning_locations.parquet",
    ]

    if dense_model:
        file_names = [
            "df_dealer_results_dense.pkl",
            "df_total_centroids_dense.pkl",
            "df_suspicious_hotspots_parameters_dense.parquet",
            "df_total_scanning_locations_dense.parquet",
        ]

    # df_dealer_results = pd.concat([df_result_within_archive, df_result_without_archive])
    df_dealer_results_path = os.path.join(output_month_path, file_names[0])
    df_dealer_results.to_pickle(df_dealer_results_path)

    # 转空地址空list 变成空字符串
    for col in ["formatted_address", "province", "district", "city", "street"]:
        df_total_centroids[col] = df_total_centroids[col].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )

    df_total_centroids_path = os.path.join(output_month_path, file_names[1])
    df_total_centroids.to_pickle(df_total_centroids_path)

    df_suspicious_hotspots_parameters_path = os.path.join(
        output_month_path, file_names[2]
    )
    df_suspicious_hotspots_parameters.to_parquet(df_suspicious_hotspots_parameters_path)

    df_total_scanning_locations_path = os.path.join(output_month_path, file_names[3])
    df_total_scanning_locations.to_parquet(df_total_scanning_locations_path)


def load_from_raw_outputs(
    df_dealer_results_path,
    df_total_centroids_path,
    df_total_scanning_locations_path,
    df_suspicious_hotspots_parameters_path,
):

    df_dealer_results = pd.read_pickle(df_dealer_results_path)
    df_total_centroids = pd.read_pickle(df_total_centroids_path)
    df_total_scanning_locations = pd.read_parquet(df_total_scanning_locations_path)
    df_suspicious_hotspots_parameters = pd.read_parquet(
        df_suspicious_hotspots_parameters_path
    )

    return (
        df_dealer_results,
        df_total_centroids,
        df_total_scanning_locations,
        df_suspicious_hotspots_parameters,
    )


def read_outputs(output_files_path, dense_model=False):
    file_names = []

    if dense_model:
        file_names = [
            "df_dealer_results_dense.pkl",
            "df_total_centroids_dense.pkl",
            "df_total_scanning_locations_dense.parquet",
            "df_suspicious_hotspots_parameters_dense.parquet",
        ]

    else:
        file_names = [
            "df_dealer_results.pkl",
            "df_total_centroids.pkl",
            "df_total_scanning_locations.parquet",
            "df_suspicious_hotspots_parameters.parquet",
        ]

    df_dealer_results_path = os.path.join(output_files_path, file_names[0])
    df_total_centroids_path = os.path.join(output_files_path, file_names[1])
    df_total_scanning_locations_path = os.path.join(output_files_path, file_names[2])
    df_suspicious_hotspots_parameters_path = os.path.join(
        output_files_path, file_names[3]
    )

    (
        df_dealer_results,
        df_total_centroids,
        df_total_scanning_locations,
        df_suspicious_hotspots_parameters,
    ) = load_from_raw_outputs(
        df_dealer_results_path,
        df_total_centroids_path,
        df_total_scanning_locations_path,
        df_suspicious_hotspots_parameters_path,
    )

    return (
        df_dealer_results,
        df_total_centroids,
        df_total_scanning_locations,
        df_suspicious_hotspots_parameters,
    )


def extract_paths(workspace_folder_path, year_month_str):
    data_folder = os.path.join(workspace_folder_path, "data/")
    output_folder = os.path.join(workspace_folder_path, "outputs/")
    config_folder = os.path.join(workspace_folder_path, "config/")

    # year_str = year_month_str[:4]
    # month_str = year_month_str[4:]

    main_data_folder = os.path.join(data_folder, f"{year_month_str}/main_data/")

    df_total_path = os.path.join(main_data_folder, "df_total.parquet")
    dealer_scope_dict_path = os.path.join(main_data_folder, "dealer_scope_dict.pkl")
    config_file_path = os.path.join(config_folder, "config.yaml")
    parameter_file_path = os.path.join(config_folder, "parameters.json")

    return (
        df_total_path,
        dealer_scope_dict_path,
        config_file_path,
        parameter_file_path,
        output_folder,
    )


def find_valid_regions_monthly_application(
    dealer_id, product_group_id, start_date_str, end_date_str, dealer_scope_dict
):

    df_valid_scope, is_archive = find_valid_regions(
        dealer_id, product_group_id, start_date_str, dealer_scope_dict
    )

    # 以为每个月月初的范围为基准，如果为空，再查看月末是否为空。
    if df_valid_scope.empty:
        df_valid_scope_end, is_archive = find_valid_regions(
            dealer_id, product_group_id, end_date_str, dealer_scope_dict
        )
        if not df_valid_scope_end.empty:
            df_valid_scope = df_valid_scope_end

    df_valid_scope = df_valid_scope.reset_index(drop=True)

    return df_valid_scope, is_archive


def get_polylines_adcodes_for_valid_regions(df_valid_scope, config_file_path, sleep=0):

    adcodes = list(df_valid_scope['AREA_CODE']) # 有一部分公司的area_Code 与高德api的adcode 不一致！！！！
    # print(df_valid_scope[['AREA_CODE', "PROVINCE", "CITY", "DISTRICT", "STREET"]])

    # 添加当前（月初或月末）有效经营范围的区域划分线
    # adcodes = []
    # for i in range(len(df_valid_scope)):
    #     address = df_valid_scope.loc[
    #         i, ["PROVINCE", "CITY", "DISTRICT", "STREET"]
    #     ].tolist()
    #     area_name = ""
    #     for item in address:
    #         if item != "-1":
    #             area_name += item
    #         else:
    #             break
    #     adcode = get_adcode(config_file_path, area_name, sleep=sleep)["geocodes"][0][
    #         "adcode"
    #     ]
    #     adcodes.append(adcode)

    polyline_points_list_total = []
    if adcodes:
        for adcode in adcodes:
            # print(adcode)
            polyline_points_list = get_polyline_points(
                config_file_path, adcode, sleep=sleep
            )
            for x in polyline_points_list:
                polyline_points_list_total.append(x)

    return polyline_points_list_total, adcodes


def show_region_product_code_overview(
    dealer_region_name, product_group_id, year_month_str, workspace_folder_path="./"
):

    df_total_path, _, _, _, _ = extract_paths(workspace_folder_path, year_month_str)

    df_total = pd.read_parquet(df_total_path)

    df_total = df_total.loc[df_total["ORG_REGION_NAME"] == dealer_region_name, :]
    print(f"{dealer_region_name} - {year_month_str}所有品项 共有 {len(df_total)} 条")

    df_total = df_total.loc[df_total["PRODUCT_GROUP_CODE"] == product_group_id, :]
    print(
        f"{dealer_region_name} - {year_month_str} - {product_group_id}品项 共有 {len(df_total)} 条"
    )

    df_count = (
        df_total.groupby(by="BELONG_DEALER_NO")["BARCODE_BOTTLE"]
        .count()
        .reset_index(name="scanning_count")
    )

    print(f"扫码经销商数量 {len(df_count)}")
    print(f"经销商扫码-均值 {df_count['scanning_count'].mean().round(2)}")
    print(f"经销商扫码-中位数 {df_count['scanning_count'].median()}")

    return df_count


def cal_noise_ratio(
    dealer_region_name,
    product_group_id,
    year_month_str,
    workspace_folder_path="./",
):

    _, _, _, _, output_folder = extract_paths(workspace_folder_path, year_month_str)

    folder_path = (
        f"{output_folder}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )

    (
        df_dealer_results,
        df_total_centroids,
        df_total_scanning_locations,
        df_suspicious_hotspots_parameters,
    ) = read_outputs(folder_path)

    noise_count = len(
        df_total_scanning_locations.loc[df_total_scanning_locations.cluster_label == -1]
    )
    total_count = len(df_total_scanning_locations)
    noise_ratio = round((noise_count / total_count), 2)
    print(
        f"噪声点比: {dealer_region_name} - {product_group_id} - {year_month_str}: {noise_ratio}"
    )

    return noise_ratio


def read_output_files_tem(
    dealer_region_name,
    product_group_id,
    year_month_str,
    dense_model = False,
    workspace_folder_path="./",
):
    """
    临时的function, 读取sparse 的数据
    """
    _, _, _, _, output_folder = extract_paths(workspace_folder_path, year_month_str)

    folder_path = (
        f"{output_folder}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )

    (
        df_dealer_results,
        df_total_centroids,
        df_total_scanning_locations,
        df_suspicious_hotspots_parameters,
    ) = read_outputs(folder_path, dense_model=dense_model)

    return (
        df_dealer_results,
        df_total_centroids,
        df_total_scanning_locations,
        df_suspicious_hotspots_parameters,
    )


# import json

# def load_model_parameters_config(region_name, product_group_id, paramters_config_file_path):
#     # 读取配置文件
#     with open(paramters_config_path, "r") as file:
#         config_data = json.load(file)

#     # 查找对应的参数
#     try:
#         region_config = config_data.get(region_name, {})
#         product_group_config = region_config.get(product_group_id, {})

#         # 如果没有找到，返回默认值
#         return product_group_config
#     except KeyError:
#         raise ValueError(f"Configuration for {region_name} and {product_group_id} not found")


# 加载配置文件
def load_model_parameters_config(
    dealer_region_name, product_group_id, paramters_config_file_path
):
    with open(paramters_config_file_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # 获取默认配置
    default_config = config_data["parameters_config"]["default_config"]
    merged_config = default_config.copy()
    default_thresholds = default_config["thresholds"]
    merged_thresholds = default_thresholds.copy()
    default_dense_thresholds = default_config["dense_thresholds"].copy()
    merged_dense_thresholds = default_dense_thresholds.copy()

    # 获取区域和产品组的配置，若不存在则使用默认配置
    region_config = config_data["parameters_config"].get(dealer_region_name, {})
    # print(f"region_config:{region_config}")
    product_group_config = region_config.get(product_group_id, {})
    # print(f"product_group_config:{product_group_config}")

    # merged_config.update(product_group_config)  # 覆盖默认配置
    if "thresholds" in product_group_config:
        merged_thresholds.update(product_group_config.get("thresholds"))
        product_group_config['thresholds'] = merged_thresholds
    if "dense_thresholds" in product_group_config:
        merged_dense_thresholds.update(product_group_config.get("dense_thresholds"))
        product_group_config['dense_thresholds'] = merged_dense_thresholds
    merged_config.update(product_group_config)  # 覆盖默认配置

    # print(merged_config)
    special_dealer_region_names = config_data["special_dealer_region_names"]

    return merged_config, special_dealer_region_names
