from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from geopy.distance import geodesic

import json
import os
import pandas as pd
import requests
import time
import yaml


def extract_paths(
    workspace_folder_path: str, year_month_str: str
) -> tuple[str, str, str, str, str, str]:
    """
    需要修改！！！！

    基于给定的工作区文件路径和年月份，解析出扫码记录主数据(df_total.parquet)， 经销商品项经营范围字典主数据(dealer_scope_dict.pkl),
    API key等重要敏感信息的配置文件(config.yaml), 窜货模型应用的参数配置文件(parameters.json)和模型输出的文件夹的路径。

    Parameters
    ----------
    worspace_folder_path: str
        该项目工作区文件夹路径。

    year_month_str: str
        模型对应的年月份字符串，格式为"%Y%m", 例如"202412"。

    Returns
    -------
    tuple[str, str, str, str, str]
        1. 扫码记录主数据(df_total.parquet)的路径。
        2. 经销商品项经营范围字典主数据(dealer_scope_dict.pkl)的路径。
        3. API key等重要敏感信息的配置文件(config.yaml)的路径。
        4. 窜货模型应用的参数配置文件(parameters.json)的路径。
        5. 模型输出的文件夹(outputs/)的路径。
    """
    data_folder = os.path.join(workspace_folder_path, "data/")
    output_folder = os.path.join(workspace_folder_path, "outputs/")
    config_folder = os.path.join(workspace_folder_path, "config/")
    main_data_folder = os.path.join(data_folder, f"{year_month_str}/main_data/")

    df_total_path = os.path.join(main_data_folder, "df_total.parquet")
    dealer_scope_dict_path = os.path.join(main_data_folder, "dealer_scope_dict.pkl")
    df_report_path = os.path.join(main_data_folder, "df_report.parquet")
    config_file_path = os.path.join(config_folder, "config.yaml")
    parameter_file_path = os.path.join(config_folder, "parameters.json")

    return (
        df_total_path,
        dealer_scope_dict_path,
        df_report_path,
        config_file_path,
        parameter_file_path,
        output_folder,
    )


def find_valid_regions(
    dealer_id: str, product_group_id: str, query_date: str, dealer_scope_dict: dict
) -> tuple[pd.DataFrame, int]:
    """
    基于给定日期，查询给定经销商品项，是否在经销商经营范围字典中存在记录以及在此日期时生效的经营范围(当前未在主模型使用)。

    Parameters
    ----------
    dealer_id: str
        经销商编码。

    product_group_id: str
        产品品项编码。

    query_date: str
        给定的查询日期字符串。格式为"%Y-%m-%d"。例如："2024-12-01"。

    dealer_scope_dict: dict
        清洗后的主数据：经销商品项经营范围字典

    Returns
    -------
    tuple[pd.DataFrame, int]
        pd.DataFrame: 在给定查询日期时，给定的经销商品项的生效经营范围。如果该经销商品项不存在记录，则返回empty df.
        int: 给定的经销商品项是否在经营范围字典中存在记录。1: 存在; 0: 不存。


    """
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


def find_equivalent_regions(
    dealer_id: str,
    product_group_id: str,
    query_date_str: str,
    dealer_scope_dict: dict,
    margin_months: int = 0,
) -> tuple[pd.DataFrame, int]:
    """
    基于给定的查询日期和宽容的月份数，查询给定的经销商品项的经营范围是否收录以及此时间政策下的“等效经营范围”。

    Parameters
    ----------
    dealer_id: str
        经销商编码。

    product_group_id: str
        产品品项编码。

    query_date: str
        给定的查询日期字符串。格式为"%Y-%m-%d"。例如："2024-12-01"。

    dealer_scope_dict: dict
        清洗后的主数据：经销商品项经营范围字典

    margin_months:
        “等效经营范围”允许宽容的月份数。默认值为6。

    Returns
    -------
    tuple[pd.DataFrame, int]
        pd.DataFrame: 在给定查询日期时，给定的经销商品项的"等效经营范围"。如果该经销商品项不存在记录，则返回empty df.
        int: 给定的经销商品项是否在经营范围字典中存在记录。1: 存在; 0: 不存在。

    Description
    -----------
    这里的“等效经营范围”为查询日期{query_date_str}向前宽容月份数{margin_months}，在此日期后有过生效的经营范围。
    例如：query_date_str="2024-12-01", margin_months=6 即查询2024年6月1日后，该经销商品项生效过的经营范围（无论查询日期时是否仍生效）。
    “等效经营范围”的设定是因为经销商出货给下游与消费者实际开瓶扫码之间存在时间差。存在虽然经营范围已失效，但已铺货到下游销售的正常场景。

    """
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
        df_equivalent_region = df_equivalent_region.sort_values(
            by="INACTIVE_DATE", ascending=False
        ).drop_duplicates(subset="AREA_NAME", keep="first")

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


def find_valid_regions_monthly_application(
    dealer_id: str,
    product_group_id: str,
    start_date_str: str,
    end_date_str: str,
    dealer_scope_dict: dict,
) -> tuple[pd.DataFrame, str]:
    """
    基于给定的查询月份首日和最后一日，查询给定的经销商品项的是否收录以及此时间政策下的"当月有效经营范围"。(当前未在模型使用)。

    Parameters
    ----------
    dealer_id: str
        经销商编码。

    product_group_id: str
        产品品项编码。

    start_date_str: str
        给定的查询月份的第一天字符串。格式为"%Y-%m-%d"。例如："2024-12-01"。

    end_date_str: str
        给定的查询月份的最后一天字符串。格式为"%Y-%m-%d"。例如："2024-12-31"。

    dealer_scope_dict: dict
        清洗后的主数据：经销商品项经营范围字典。

    Returns
    -------
    tuple[pd.DataFrame, int]
        pd.DataFrame: 在给定查询日期时，给定的经销商品项的"当月有效经营范围"。如果该经销商品项不存在记录，则返回empty df.
        int: 给定的经销商品项是否在经营范围字典中存在记录。1: 存在; 0: 不存在。

    Description
    -----------
    "当月有效经营范围"定义规则为先查询该月第一天的生效经营范围，如果此范围不为空，则取此范围。
    如果此范围为空，则查询该月最后一天的生效经营范围作为最终的"当月有效经营范围"。
    """
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


def get_address_raw_data_from_lat_lon(
    location: str, config_file_path: str
) -> dict | None:
    """
    通过高德API, 获取给定经纬度的地址原始信息。

    该函数使用高德地图的逆地理编码 API，将经纬度信息转换为地址。函数会读取配置文件，获取高德 API 的密钥，
    然后向高德 API 发送请求并返回响应数据。如果请求失败，则返回 None。

    Parameters
    ----------
    location: str
        经纬度字符串，格式为"经度, 纬度"（例如："116.481488, 39.990464"）

    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    Returns
    -------
    dict
        高德地图 API 返回的 JSON 格式地址数据。如果请求失败，返回 None。
    """

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


def get_region_info(
    config_file_path: str, adcode: str, sleep: float = 0
) -> dict | None:
    """
    通过高德行政区域API，根据区域代码adcode，获取行政区域信息。

    Parameters
    ----------
    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    adcdoe: str
        区域代码。

    sleep: float
        多次访问api时的间歇时间(s)，默认值为0。

    Returns
    -------
    dict: 高德地图 API 返回的 JSON 格式行政区域信息数据。如果请求失败，返回 None。
    """
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


def get_polyline_points(
    config_file_path: str, adcode: str, sleep: int = 0
) -> list[list[list[float, float]]]:
    """
    根据给定区域代码adcode，通过高德行政区域API获取该区域行政区域边界点集合。

    高德API返回的原始polyline坐标是[经度lon, 纬度lat]。此函数整理最后返回的坐标点顺序为[纬度lat, 经度lon]。高德API最细可获取区一级行政边界。

    Parameters
    ----------
    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    adcdoe: str
        区域代码。

    sleep: int
        query高德API时的间歇时间。默认值为0.

    Returns
    -------
    list[list[list[float, float]]]: 该区域行政边界坐标点集合。第一层list内部为不同封闭区域的坐标点集合。第二层单一封闭区域内的坐标点[纬度lat, 经度lon].

    Description
    -----------
    高德行政区域API中的最细颗粒度为区一级。公司数仓中经销商经营范围中存在更低级的经营范围（镇街）及相关area_code(不知道从哪里来的？)。
    这里根据观察规律，直接截取这类更低级区域的area_code的前六位，获得其对应的区县一级的行政区域code。如果此规律不对，后续需更改。
    因此此函数最细可返回区县一级行政边界点的集合。

    """
    if len(adcode) > 6:
        adcode = adcode[
            :6
        ]  # 替换低于区一级 的行政代码。高德api 最低只有区一级的行政边界
    data = get_region_info(config_file_path, adcode, sleep=sleep)

    if not data["districts"]:
        print(f"adcode 变化： {adcode}")
        return []

    polyline = data["districts"][0]["polyline"]
    # 这里可能会出现 data["districts"] 为空，原因是由于adcode与api的最新不符合 ######################
    polyline_points_list = []
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


def get_polylines_adcodes_for_valid_regions(
    df_valid_scope: pd.DataFrame, config_file_path: str, sleep: int = 0
) -> tuple[list[list[list[float, float]]], list[str]]:
    """
    基于给定的经营范围数据帧, 通过query高德行政区域API, 给出对应范围的行政边界坐标点集合和经营范围的行政区域代码(adcode)集合。

    Parameters
    ----------
    df_valid_scope: pd.DataFrame
        给定的经营范围数据帧。至少包含"AREA_CODE"这一列，实际应用中应包含['AREA_CODE', "PROVINCE", "CITY", "DISTRICT", "STREET"]。

    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    sleep: int
        query高德API时的间歇时间。默认值为0.

    Returns
    -------
    tuple[list[list[list[float, float]]], list[str]]
        list[list[list[float, float]]]: 所有给定区域的行政边界坐标点集合。第一层list内部为不同封闭区域的坐标点集合。第二层单一封闭区域内的坐标点[纬度lat, 经度lon].
        list[str]: 所有给定区域的行政区域代码的集合。

    Description
    -----------
        此函数目前访问API获取行政边界坐标，依靠区域编码adcode。在这个函数中，直接利用数仓中经营范围表中的"AREA_CODE"这一列的值作为adcode.
        经过实际测试，"AREA_CODE"与高德API中的"adcode"几乎完全相同。但是因为数仓数据的历史性以及更新规则，会存在已失效code，引发获取失败。
        这一部分问题目前依靠在data_preprocessing.py硬编码替换处理。注释掉的部分采用先根据区域名称访问API获取最新adcode,再query获得行政边界，
        这个方法因较为浪费API token， 运行较慢且adcode不一致的情况属极特殊情况暂未采用。
    """

    adcodes = list(
        df_valid_scope["AREA_CODE"]
    )  # 有一部分公司的area_Code 与高德api的adcode 不一致！！！！
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


def get_month_start_end(year_month_str: str) -> tuple[str, str]:
    """
    给定年月份，返回该月份的第一天和最后一天。

    Parameters
    ----------
    year_month_str: str
        年月份字符串，格式为"%Y%m"，例如"202412"。

    Returns
    -------
    tuple[str, str]
        str: 给定月份的第一天，格式为"%Y-%m-%d"。例如："2024-12-01"。
        str: 给定月份的最后一天，格式为"%Y-%m-%d"。例如："2024-12-31"。

    """
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


def is_belong_to(
    address_list: list[str, str, str, str], scope_list: list[str, str, str, str]
) -> bool:
    """
    判断某一地址是否符合经营范围。

    Parameters
    ----------
    address_list: list[str, str, str, str]
        被判断是否在经营范围的地址列表。格式为["PROVINCE", "CITY", "DISTRICT", "STREET"]。
        在此窜货模型中, 这一地址主要来自于: 1.簇质心通过高德API获取的地址; 2. 开瓶扫码表中的开瓶地址。

    scope_list: list[str, str, str, str]
        用于判断的经营列表, 来自于经过清洗后的经销商品项经营范围字典。格式为["PROVINCE", "CITY", "DISTRICT", "STREET"]。
        用"-1"替代空, 用来提示范围细化到哪一级别。例如['广东省', '广州市', '-1', '-1'] 意味着这个经营范围包含整个广州。

    Returns
    -------
    bool
        被判断地址是否在经营范围内。

    Description
    -----------
    此模型涉及的三种来源的地址列表：
        1. 簇心经纬度获取的簇心地址 ["province", "city", "district", "city"]
        2. 开瓶扫码表中的开瓶地址。["OPEN_PROVINCE", "OPEN_CITY", "OPEN_DISTRICT", "OPEN_TOWN"]
        3. 清洗后的经营范围地址。["PROVINCE", "CITY", "DISTRICT", "STREET"]
    """

    if "-1" in scope_list:
        level = scope_list.index("-1")
    else:
        level = len(scope_list)

    if level == 0:
        return False

    for i in range(level):
        if address_list[i] != scope_list[i]:
            return False
    return True


def load_from_raw_outputs(
    df_dealer_results_path: str,
    df_total_centroids_path: str,
    df_total_scanning_locations_path: str,
    df_suspicious_hotspots_parameters_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载模型运行输出的文件。

    Parameters
    ----------
    df_dealer_results_path: str
        模型输出的经销商维度结果的存储路径。

    df_total_centroids_path: str
        模型输出的簇维度结果的存储路径。

    df_total_scanning_locations_path: str
        模型输出的每条扫码记录维度结果的存储路径。

    df_suspicious_hotspots_parameters_path: str
        模型采用分簇和判别可以热点的参数。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        1. pd.DataFrame: 模型输出的经销商维度结果。
        2. pd.DataFrame: 模型输出的簇维度结果。
        3. pd.DataFrame: 模型输出的每条扫码记录维度结果。
        4. pd.DataFrame: 模型采用分簇和判别可以热点的参数。
    """

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


def read_outputs(
    output_files_path: str, dense_model: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    给定模型输出文件的存储路径，读取全部输出文件。

    Parameters
    ----------
    output_files_path: str
        模型输出文件的存储路径。

    dense_model: bool
        文件是否由二次分簇的过程输出。默认值为False.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        1. pd.DataFrame: 模型输出的经销商维度结果。
        2. pd.DataFrame: 模型输出的簇维度结果。
        3. pd.DataFrame: 模型输出的每条扫码记录维度结果。
        4. pd.DataFrame: 模型采用分簇和判别可以热点的参数。
    """
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


def load_model_parameters_config(
    dealer_region_name: str, product_group_id: str, parameters_config_file_path: str
) -> tuple[dict, list[list[str, str]]]:
    """
    加载模型参数文件，获得给定大区品项的模型设置参数及需要二级分簇的大区品项的集合。

    Parameters
    ----------
    dealer_region_name: str
        大区名称。

    product_group_id: str
        品项编码。

    parameters_config_file_path: str
        包含模型参数的配置文件(json)的路径。

    Returns
    -------
    tuple[dict, list[list[str, str]]]
        dict: 给定大区品项对应的模型参数字典。
        list[list[str, str]]: 所有需要二级分簇的["dealer_region_name", "product_group_id"]集合。

    Description
    -----------
    模型参数配置文件为json格式。里面包含默认参数组"default_config"，为所有模型的基础参数。
    采用不同参数的大区品项对应的特殊参数会被显示表明，此函数将特殊参数覆盖在默认参数上返回。
    """
    with open(parameters_config_file_path, "r", encoding="utf-8") as f:
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
        product_group_config["thresholds"] = merged_thresholds
    if "dense_thresholds" in product_group_config:
        merged_dense_thresholds.update(product_group_config.get("dense_thresholds"))
        product_group_config["dense_thresholds"] = merged_dense_thresholds
    merged_config.update(product_group_config)  # 覆盖默认配置

    # print(merged_config)
    special_dealer_region_names = config_data["special_dealer_region_names"]

    return merged_config, special_dealer_region_names


def save_model_outputs(
    df_dealer_results: pd.DataFrame,
    df_total_scanning_locations: pd.DataFrame,
    df_total_centroids: pd.DataFrame,
    df_suspicious_hotspots_parameters: pd.DataFrame,
    year_month_str: str,
    dealer_region_name: str,
    product_group_id: str,
    output_path: str,
    dense_model: bool = False,
) -> None:
    """
    将模型运行后输出的四个文件写入磁盘。
    写入位置为 "{output_path}/{dealer_region_name}/{product_group_id}/{year_month_str}/"。

    Parameters
    ----------
    df_dealer_results: pd.DataFrame
        模型运行的输出，主要包含 经销商维度的汇总信息。

    df_total_scanning_locations: pd.DataFrame
        模型运行的输出，包含 每条扫码记录的相关信息。

    df_total_centroids: pd.DataFrame
        模型运行的输出，主要包含 每个簇维度的相关信息。

    df_suspicious_hotspots_parameters: pd.DataFrame
        模型运行的输出，包含模型聚簇及判定可以热点的参数信息。

    year_month_str: str
        模型运行的年月份。

    dealer_region_name: str
        模型运行的经销商大区名称。

    product_group_id: str
        模型运行的产品品项ID。

    output_path: str
        给定的模型输出文件的写入路径。

    dense_model: bool
        输出的文件是否来源二级分簇时的运行结果。默认值为False.

    Returns
    -------
    None
        该函数用于写入文件，不返回任何内容。

    Description
    -----------
    该函数将模型运行生成的四个文件(不同维度的信息）：
        1. df_dealer_results
        2. df_total_scanning_locations
        3. df_total_centroids
        4. df_suspicious_hotspots_parameters
    储存成pickle或者parquet格式写入硬盘。

    如果文件来源于正常分簇的输出，则文件命名分别为：
        1. "df_dealer_results.pkl"
        2. "df_total_scanning_locations.parquet"
        3. "df_total_centroids.pkl"
        4. "df_suspicious_hotspots_parameters.parquet"

    如果文件来源于一些特殊大区品项(例如天津大区）正常分簇后，针对特大簇的二级分簇，则文件命名分别为：
        1. "df_dealer_results_dense.pkl"
        2. "df_total_scanning_locations_dense.parquet"
        3. "df_total_centroids_dense.pkl"
        4. "df_suspicious_hotspots_parameters_dense.parquet"

    针对df_total_centroids中，由于高德API返回的簇心地址中的空为空list([])的情况，为避免后续处理的麻烦，全部转化为空str("")。

    """

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


def find_closest_point_geodesic(
    fixed_point: tuple[float, float], coordinates: list[list[float, float]]
) -> tuple[tuple[float, float], float]:
    """
    给定一个固定点坐标和一个坐标集合，找到坐标集合中距离固定点的最近的坐标和对应的最小距离。

    Parameters
    ----------
    fixed_point: tuple[float, float]:
        固定点的坐标(lat,lon)

    coordinates: list[list[float, float]]
        坐标集合。[[lat, lon], [lat, lon], ...]

    Returns
    -------
    tuple[tuple[float, float], float]
        tuple[float, float]: 坐标集合中距离固定点最近的坐标(lat, lon)。
        float: 对应的最小距离(lat, lon)。

    """
    min_distance = float("inf")
    # min_distance = float("99999")
    closest_point = None

    for coord in coordinates:
        coordinate = (coord[0], coord[1])
        distance = geodesic(
            fixed_point, coordinate
        ).kilometers  # Distance in kilometers
        if distance < min_distance:
            min_distance = distance
            closest_point = coord

    return closest_point, min_distance


# def find_all_suspicious_dealers_in_specific_area(product_group_id, year_month_str, province=None, city=None, district=None):
#     all_regions = ['上海大区', '云贵川渝大区', '北京大区',  '晋陕宁大区', '江苏大区', '浙江大区', '湖南大区', '甘青新大区',
#                    '福建大区', '赣皖大区', '黑吉辽大区', '河北大区', '河南大区',  '山东大区', '天津大区', '广东大区']
#     list_suspicious_dealers = []
#     for region in all_regions:
#         folder_path = f"outputs/{region}/{product_group_id}/{year_month_str}/"
#         df_centroids_path = os.path.join(folder_path, "df_total_centroids.pkl")
#         df_centroids = pd.read_pickle(df_centroids_path)
#         mask = (df_centroids["is_suspicious"] == 1) & (df_centroids["is_dealer_within_archive"] == 1)

#         # if province is not None and city is None and district is None:
#         #     mask = mask & (df_centroids["province"] == province)
#         # elif city is not None and province is None and district is None:
#         #     mask = mask & (df_centroids["city"] == city)
#         # elif district is not None and province is None and city is None:
#         #     mask = mask & (df_centroids["district"] == district)
#         # else:
#         #     print("请正确输入想要查找的特定区域, 如海珠区，只需要 district='海珠区', 其它区域参数不填。")
#         #     return None
#         if province is not None and city is None and district is None:
#             mask = mask & (df_centroids["province"] == province)
#         elif province is not None and city is not None and district is None:
#             mask = mask & (df_centroids["city"] == city)
#         elif province is not None and city is not None and district is not None:
#             mask = mask & (df_centroids["district"] == district)
#         else:
#             print("请正确输入想要查找的特定区域, 如海珠区, province='广东省', city='广州市', district='海珠区'")
#             return None

#         df_suspicious = df_centroids.loc[mask, ["dealer_id"]]

#         if not df_suspicious.empty:
#             df_suspicious["dealer_region_name"] = region
#             print(df_suspicious)
#             list_suspicious_dealers.append(df_suspicious)
#             print("-" * 50)
#         else:
#             if region == '天津大区':
#                 folder_path = f"outputs/{region}/{product_group_id}/{year_month_str}/"
#                 df_centroids_path = os.path.join(folder_path, "df_total_centroids_dense.pkl")
#                 df_centroids = pd.read_pickle(df_centroids_path)
#                 mask = (df_centroids["is_suspicious"] == 1) & (df_centroids["is_dealer_within_archive"] == 1)
#                 if province is not None and city is None and district is None:
#                     mask = mask & (df_centroids["province"] == province)
#                 elif province is not None and city is not None and district is None:
#                     mask = mask & (df_centroids["city"] == city)
#                 elif province is not None and city is not None and district is not None:
#                     mask = mask & (df_centroids["district"] == district)
#                 else:
#                     print("请正确输入想要查找的特定区域, 如海珠区, province='广东省', city='广州市', district='海珠区'")
#                     return None
#                 df_suspicious = df_centroids.loc[mask, ["dealer_id"]]
#                 if not df_suspicious.empty:
#                     df_suspicious["dealer_region_name"] = region
#                     print(df_suspicious)
#                     list_suspicious_dealers.append(df_suspicious)
#                     print("-" * 50)

#             if region == '广东大区' and product_group_id == '01':
#                 folder_path = f"outputs/{region}/{product_group_id}/{year_month_str}/"
#                 df_centroids_path = os.path.join(folder_path, "df_total_centroids_dense.pkl")
#                 df_centroids = pd.read_pickle(df_centroids_path)
#                 mask = (df_centroids["is_suspicious"] == 1) & (df_centroids["is_dealer_within_archive"] == 1)
#                 if province is not None and city is None and district is None:
#                     mask = mask & (df_centroids["province"] == province)
#                 elif province is not None and city is not None and district is None:
#                     mask = mask & (df_centroids["city"] == city)
#                 elif province is not None and city is not None and district is not None:
#                     mask = mask & (df_centroids["district"] == district)
#                 else:
#                     print("请正确输入想要查找的特定区域, 如海珠区, province='广东省', city='广州市', district='海珠区'")
#                     return None
#                 df_suspicious = df_centroids.loc[mask, ["dealer_id"]]
#                 if not df_suspicious.empty:
#                     df_suspicious["dealer_region_name"] = region
#                     print(df_suspicious)
#                     list_suspicious_dealers.append(df_suspicious)
#                     print("-" * 50)
#     if list_suspicious_dealers:
#         df_results = pd.concat(list_suspicious_dealers)
#         df_results = df_results.drop_duplicates(ignore_index=True)
#         df_results["product_group_id"] = product_group_id
#         df_results["month"] = year_month_str

#         return df_results

#     else:
#         return None


# 怎么生成excel
def find_all_suspicious_dealers_in_specific_area(
    product_group_id, year_month_str, province=None, city=None, district=None
):
    all_regions = [
        "上海大区",
        "云贵川渝大区",
        "北京大区",
        "晋陕宁大区",
        "江苏大区",
        "浙江大区",
        "湖南大区",
        "甘青新大区",
        "福建大区",
        "赣皖大区",
        "黑吉辽大区",
        "河北大区",
        "河南大区",
        "山东大区",
        "天津大区",
        "广东大区",
    ]
    list_suspicious_dealers = []
    for region in all_regions:
        folder_path = f"outputs/{region}/{product_group_id}/{year_month_str}/"
        df_centroids_path = os.path.join(folder_path, "df_total_centroids.pkl")
        df_centroids = pd.read_pickle(df_centroids_path)
        mask = (df_centroids["is_suspicious"] == 1) & (
            df_centroids["is_dealer_within_archive"] == 1
        )

        if province is not None and city is None and district is None:
            mask = mask & (df_centroids["province"] == province)
        elif province is not None and city is not None and district is None:
            mask = mask & (df_centroids["city"] == city)
        elif province is not None and city is not None and district is not None:
            mask = mask & (df_centroids["district"] == district)
        else:
            print(
                "请正确输入想要查找的特定区域, 如海珠区, province='广东省', city='广州市', district='海珠区'"
            )
            return None

        df_suspicious = df_centroids.loc[mask, ["dealer_id"]]

        if not df_suspicious.empty:
            df_suspicious["dealer_region_name"] = region
            print(df_suspicious)
            list_suspicious_dealers.append(df_suspicious)

            print("-" * 50)
        else:
            if region == "天津大区":
                folder_path = f"outputs/{region}/{product_group_id}/{year_month_str}/"
                df_centroids_path = os.path.join(
                    folder_path, "df_total_centroids_dense.pkl"
                )
                df_centroids = pd.read_pickle(df_centroids_path)
                mask = (df_centroids["is_suspicious"] == 1) & (
                    df_centroids["is_dealer_within_archive"] == 1
                )
                if province is not None and city is None and district is None:
                    mask = mask & (df_centroids["province"] == province)
                elif province is not None and city is not None and district is None:
                    mask = mask & (df_centroids["city"] == city)
                elif province is not None and city is not None and district is not None:
                    mask = mask & (df_centroids["district"] == district)
                else:
                    print(
                        "请正确输入想要查找的特定区域, 如海珠区, province='广东省', city='广州市', district='海珠区'"
                    )
                    return None
                df_suspicious = df_centroids.loc[mask, ["dealer_id"]]
                if not df_suspicious.empty:
                    df_suspicious["dealer_region_name"] = region
                    print(df_suspicious)
                    list_suspicious_dealers.append(df_suspicious)
                    print("-" * 50)

            if region == "广东大区" and product_group_id == "01":
                folder_path = f"outputs/{region}/{product_group_id}/{year_month_str}/"
                df_centroids_path = os.path.join(
                    folder_path, "df_total_centroids_dense.pkl"
                )
                df_centroids = pd.read_pickle(df_centroids_path)
                mask = (df_centroids["is_suspicious"] == 1) & (
                    df_centroids["is_dealer_within_archive"] == 1
                )
                if province is not None and city is None and district is None:
                    mask = mask & (df_centroids["province"] == province)
                elif province is not None and city is not None and district is None:
                    mask = mask & (df_centroids["city"] == city)
                elif province is not None and city is not None and district is not None:
                    mask = mask & (df_centroids["district"] == district)
                else:
                    print(
                        "请正确输入想要查找的特定区域, 如海珠区, province='广东省', city='广州市', district='海珠区'"
                    )
                    return None
                df_suspicious = df_centroids.loc[mask, ["dealer_id"]]
                if not df_suspicious.empty:
                    df_suspicious["dealer_region_name"] = region
                    print(df_suspicious)
                    list_suspicious_dealers.append(df_suspicious)
                    print("-" * 50)
    if list_suspicious_dealers:
        df_results = pd.concat(list_suspicious_dealers)
        df_results = df_results.drop_duplicates(ignore_index=True)
        df_results["product_group_id"] = product_group_id
        df_results["month"] = year_month_str

        return df_results

    else:
        return None


# 以下函数目前仅为测试时的便利，暂未应用的到主模型中。


def add_border_scanning_ratio_to_df_suspicious_dealers(
    df_suspicious_dealers, df_total_scanning_locations, threshold=5
):

    df_suspicious_dealers_new = df_suspicious_dealers.reset_index(drop=True).copy()
    df_suspicious_dealers_new["border_scanning_ratio"] = -1.0
    for i in range(len(df_suspicious_dealers_new)):
        dealer_id = df_suspicious_dealers_new.loc[i, "BELONG_DEALER_NO"]
        # df_centroids = df_total_centroids.loc[df_total_centroids["dealer_id"] == dealer_id, :].reset_index(drop=True)
        df_scanning_locations = df_total_scanning_locations.loc[
            df_total_scanning_locations["BELONG_DEALER_NO"] == dealer_id, :
        ].reset_index(drop=True)
        polyline_points_list_total = df_suspicious_dealers_new.loc[
            i, "dealer_polyline_points_list_total"
        ]

        dealer_total_scanning_count = df_suspicious_dealers_new.loc[
            i, "dealer_total_scanning_count"
        ]
        count = 0
        for j in range(len(df_scanning_locations)):

            fixed_point = (
                df_scanning_locations.loc[j, "LATITUDE"],
                df_scanning_locations.loc[j, "LONGITUDE"],
            )
            flat_coordinates = [
                point for sublist in polyline_points_list_total for point in sublist
            ]
            # closest_point, min_distance = find_closest_point_geodesic(
            #     fixed_point, flat_coordinates
            # )
            # if min_distance <= threshold:
            #     count += 1
            for coord in flat_coordinates:
                coordinate = (coord[0], coord[1])
                distance = geodesic(
                    fixed_point, coordinate
                ).kilometers  # Distance in kilometers
                if distance <= threshold:
                    count += 1
                    break

        border_scanning_count = round(count / dealer_total_scanning_count, 3)

        df_suspicious_dealers_new.loc[i, "border_scanning_ratio"] = (
            border_scanning_count
        )
    return df_suspicious_dealers_new


def cal_border_scanning_ratio_for_dealer(
    dealer_region_name,
    dealer_id,
    product_group_id,
    year_month_str,
    threshold=5,
    workspace_folder_path="./",
):
    _, _, _, _, output_folder = extract_paths(workspace_folder_path, year_month_str)
    folder_path = (
        f"{output_folder}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )
    if not os.path.exists(folder_path):
        print(f"没有找到数据路径: {folder_path}")
    df_dealer_results, _, df_total_scanning_locations, _ = read_outputs(folder_path)

    df_scanning_locations = df_total_scanning_locations.loc[
        df_total_scanning_locations["BELONG_DEALER_NO"] == dealer_id,
    ].reset_index(drop=True)
    if df_scanning_locations.empty:
        print(
            f"未查询到{dealer_id}在{dealer_region_name}-{product_group_id}-{year_month_str}的扫码记录。"
        )

        return

    df_results = df_dealer_results.loc[
        df_dealer_results["BELONG_DEALER_NO"] == dealer_id, :
    ].reset_index(drop=True)
    polyline_points_list_total = df_results.loc[0, "dealer_polyline_points_list_total"]
    dealer_total_scanning_count = df_results.loc[0, "dealer_total_scanning_count"]

    count = 0
    for j in range(len(df_scanning_locations)):

        fixed_point = (
            df_scanning_locations.loc[j, "LATITUDE"],
            df_scanning_locations.loc[j, "LONGITUDE"],
        )
        flat_coordinates = [
            point for sublist in polyline_points_list_total for point in sublist
        ]
        # closest_point, min_distance = find_closest_point_geodesic(
        #     fixed_point, flat_coordinates
        # )
        # if min_distance <= threshold:
        #     count += 1
        for coord in flat_coordinates:
            coordinate = (coord[0], coord[1])
            distance = geodesic(
                fixed_point, coordinate
            ).kilometers  # Distance in kilometers
            if distance <= threshold:
                count += 1
                break

    border_scanning_count = round(count / dealer_total_scanning_count, 3)

    print(border_scanning_count)
    return border_scanning_count


def show_region_product_code_overview(
    dealer_region_name, product_group_id, year_month_str, workspace_folder_path="./"
):

    df_total_path, _, _, _, _, _ = extract_paths(workspace_folder_path, year_month_str)

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
    dense_model=False,
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
