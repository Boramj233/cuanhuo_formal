from datetime import datetime, timedelta

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


def get_acode(config_file_path, area_name, sleep=0):
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


def get_region_polyline(config_file_path, acode, sleep=0):
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get("gaode_api_key")

    url = "https://restapi.amap.com/v3/config/district?parameters"
    params = {
        "key": gaode_api_key,
        "keywords": acode,
        "subdistrict": 0,
        "filter": acode,
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


def get_polyline_points(config_file_path, acode, sleep=0):
    if len(acode) > 6:
        acode = acode[:6]
    data = get_region_polyline(config_file_path, acode)
    polyline = data["districts"][0]["polyline"]
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


def save_model_results(
    df_dealer_results,
    df_total_centroids,
    df_suspicious_hotspots_parameters,
    df_total_scanning_locations,
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


# 这里可以多返回一个 df_dealer_result


def extract_paths(workspace_folder_path, year_month_str):
    data_folder = os.path.join(workspace_folder_path, "data/")
    output_folder = os.path.join(workspace_folder_path, "outputs/")
    config_folder = os.path.join(workspace_folder_path, "config/")

    year_str = year_month_str[:4]
    month_str = year_month_str[4:]

    main_data_folder = os.path.join(data_folder, f"{year_str}-{month_str}/main_data/")

    df_total_path = os.path.join(main_data_folder, "df_total.parquet")
    dealer_scope_dict_path = os.path.join(main_data_folder, "dealer_scope_dict.pkl")
    config_file_path = os.path.join(config_folder, "config.yaml")

    return df_total_path, dealer_scope_dict_path, config_file_path, output_folder


def find_valid_regions_monthly_application(
    dealer_id, product_group_id, start_date_str, end_date_str, dealer_scope_dict
):

    df_valid_scope, is_archive = find_valid_regions(
        dealer_id, product_group_id, start_date_str, dealer_scope_dict
    )

    if df_valid_scope.empty:
        df_valid_scope_end, is_archive = find_valid_regions(
            dealer_id, product_group_id, end_date_str, dealer_scope_dict
        )
        if not df_valid_scope_end.empty:
            df_valid_scope = df_valid_scope_end

    df_valid_scope = df_valid_scope.reset_index(drop=True)
    return df_valid_scope, is_archive


def get_polylines_acodes_for_valid_regions(df_valid_scope, config_file_path, sleep=0):

    # df_valid_scope = find_valid_regions_monthly_application(dealer_id, product_group_id, start_date_str, end_date_str, dealer_scope_dict)

    # acodes = list(df_valid_scope['AREA_CODE']) # 有一部分公司的area_Code 与高德api的acode 不一致！！！！
    # 添加当前（月初或月末）有效经营范围的区域划分线
    acodes = []
    for i in range(len(df_valid_scope)):
        address = df_valid_scope.loc[
            i, ["PROVINCE", "CITY", "DISTRICT", "STREET"]
        ].tolist()
        area_name = ""
        for item in address:
            if item != "-1":
                area_name += item
            else:
                break
        acode = get_acode(config_file_path, area_name, sleep=sleep)["geocodes"][0][
            "adcode"
        ]
        acodes.append(acode)

    polyline_points_list_total = []
    if acodes:
        for acode in acodes:
            polyline_points_list = get_polyline_points(
                config_file_path, acode, sleep=sleep
            )
            for x in polyline_points_list:
                polyline_points_list_total.append(x)

    return polyline_points_list_total, acodes
