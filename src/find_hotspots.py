from datetime import datetime
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

from .functions import (
    find_closest_point_geodesic,
    find_equivalent_regions,
    get_address_raw_data_from_lat_lon,
    get_polylines_adcodes_for_valid_regions,
    is_belong_to,
)

import numpy as np
import pandas as pd
import pickle


### Find hotspots
def find_clusters_for_dealer(
    df_cleaned: pd.DataFrame,
    dealer_id: str,
    product_group_id: str,
    start_date_str: str,
    end_date_str: str,
    radius: float,
    min_samples: int,
) -> pd.DataFrame:
    """
    给定清洗过的扫码记录数据和DBSCAN算法的模型参数，返回给定经销商品项在一定时间段内, 基于经纬度经过DBSCAN算法，带有聚类label结果的扫码记录数据帧。

    Parameters
    ----------
    df_cleaned: pd.DataFrame
        经过清洗后的包含扫码记录数据的数据帧。

    dealer_id: str
        给定的经销商编码。

    product_group_id: str
        给定的产品品项编码。

    start_date_str: str
        所选时间段的第一天。格式为"%Y-%m-%d"，例如"2024-12-01".

    end_date_str: str
        所选时间段的最后一天。格式为"%Y-%m-%d"，例如"2024-12-31".

    radius: float
        模型所用基于经纬度的DBSCAN聚类算法的 半径大小(邻域距离阈值)。在模型中可以粗略的理解为聚簇时的半径大小(地球球面距离, 如4km)。

    min_samples: int
        模型所用基于经纬度的DBSCAN聚类算法的 成簇最少样本数(邻域中数据点个数的最小个数)。在模型中可以粗略地理解为聚簇时邻域内的最少开瓶扫码数(如6瓶)。

    Returns
    -------
    pd.DataFrame
        经过DBSCAN聚类后, 给定时间段内该经销商品项的所有扫码记录依照地理位置（经纬度）聚簇的结果（"cluster_label"）。

    Description
    -----------
    此模型中利用此函数对于单个给定的经销商品项，根据经纬度聚簇。结果可以将开瓶密度达到阈值的区域(簇)寻找出来。
    所有DBSCAN算法依据经纬度二维数据, 距离采用"haversine"距离。因此聚簇的两个参数"radius"和"min_samples"可以粗略地认为具有现实意义，
    即最小距离半径和半径内的最小开瓶扫码数。例如(4, 6)可以粗略理解为聚簇的密度阈值条件是半径为4公里的范围内最小开瓶扫码数为6瓶。

    "cluster_label": -1为未被聚成簇的离散点。
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    df_scanning_locations = df_cleaned.loc[
        (
            (df_cleaned["BELONG_DEALER_NO"] == dealer_id)
            & (df_cleaned["CUST_SCAN_DATE"] >= start_date)
            & (df_cleaned["CUST_SCAN_DATE"] <= end_date)
            & (df_cleaned["PRODUCT_GROUP_CODE"] == product_group_id)
        ),
        :,
    ].copy()

    if df_scanning_locations.empty:
        print(f"No scanning data for dealer ID {dealer_id} in the specified period.")

    else:
        # 先维度lat, 再经度lon
        coords = df_scanning_locations.loc[:, ["LATITUDE", "LONGITUDE"]].values
        # 将经纬度转换为弧度
        data_rad = np.radians(coords)
        # 使用 DBSCAN 并指定 Haversine 距离
        dbscan = DBSCAN(
            eps=radius / 6371, min_samples=min_samples, metric="haversine"
        )  # 地球半径6371公里
        df_scanning_locations["cluster_label"] = dbscan.fit_predict(data_rad)

    df_scanning_locations = df_scanning_locations.reset_index(drop=True)
    return df_scanning_locations


def get_centroids(
    df_scanning_locations: pd.DataFrame, config_file_path: str
) -> pd.DataFrame:
    """
    基于经过聚簇后的开瓶扫码df, 生成包含每个簇相关信息的df, 计算质心并通过高德API获取质心地址信息。

    Parameters
    ----------
    df_scanning_locations: pd.DataFrame
        经过聚簇后的开瓶扫码信息df。

    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    Returns
    -------
    pd.DataFrame
        包含聚簇结果，簇维度的相关信息df。

    """

    if not df_scanning_locations.empty:
        centroids = df_scanning_locations.groupby("cluster_label")[
            ["LATITUDE", "LONGITUDE"]
        ].mean()
        df_centroids = centroids.reset_index()
        df_centroids["formatted_address"] = "-1"
        df_centroids["province"] = "-1"
        df_centroids["city"] = "-1"
        df_centroids["district"] = "-1"
        df_centroids["street"] = "-1"

        for i in range(len(df_centroids)):
            location = f"{round(df_centroids.loc[i, 'LONGITUDE'], 3)}, {round(df_centroids.loc[i, 'LATITUDE'], 3)}"  # Gaode api 经度在前，纬度在后
            address = get_address_raw_data_from_lat_lon(location, config_file_path)
            df_centroids.at[i, "formatted_address"] = address["regeocode"][
                "formatted_address"
            ]
            df_centroids.at[i, "province"] = address["regeocode"]["addressComponent"][
                "province"
            ]
            df_centroids.at[i, "city"] = address["regeocode"]["addressComponent"][
                "city"
            ]
            df_centroids.at[i, "district"] = address["regeocode"]["addressComponent"][
                "district"
            ]
            df_centroids.at[i, "street"] = address["regeocode"]["addressComponent"][
                "township"
            ]
        df_centroids = df_centroids.reset_index(drop=True)
        return df_centroids
    return pd.DataFrame()


# def get_scanning_locations_and_centroids(
#     df_cleaned,
#     dealer_id,
#     product_group_id,
#     start_date_str,
#     end_date_str,
#     radius,
#     min_samples,
#     config_file_path,
# ):
#     df_scanning_loactions = find_clusters_for_dealer(
#         df_cleaned,
#         dealer_id,
#         product_group_id,
#         start_date_str,
#         end_date_str,
#         radius,
#         min_samples,
#     )
#     if df_scanning_loactions.empty:
#         return pd.DataFrame(), pd.DataFrame()

#     df_centroids = get_centroids(df_scanning_loactions, config_file_path)
#     df_scanning_loactions = df_scanning_loactions.reset_index(drop=True)

#     return df_scanning_loactions, df_centroids


def verify_centroids_within_scope(
    df_centroids: pd.DataFrame, df_scope: pd.DataFrame
) -> pd.DataFrame:
    """
    判断经销商的每个开瓶簇心是否在其经营范围内, 并加以标记。

    Parameters
    ----------
    df_centroids: pd.DataFrame
        包含经销商簇维度信息的df。

    df_scope: pd.DataFrame
        用于判断是否为异地的经销商品项经营范围df。(应该只包含认定"有效的"经营范围记录)

    Returns
    -------
    pd.DataFrame
        包含每个簇心是否异地"is_remote"标签的簇维度df.

    Description
    -----------
    簇维度df, df_centroids中的"is_remote":
        0: 本地；
        1: 异地；
        -1: 离散点的集合, 并非是真实的簇；
        -100: 该簇未进行判断是否异地, 特殊提示符。
    """
    df_centroids_with_remote = df_centroids.copy()
    df_centroids_with_remote["is_remote"] = -100

    # 0：本地；1：异地; -1: noise
    df_centroids_with_remote = df_centroids_with_remote.reset_index(drop=True)
    for i in range(len(df_centroids_with_remote)):
        cluster_label = df_centroids_with_remote.loc[i, "cluster_label"]
        if cluster_label != -1:
            address_list = (
                df_centroids_with_remote.loc[
                    i, ["province", "city", "district", "street"]
                ]
                .values.flatten()
                .tolist()
            )
            # 将高德api 返回的centroid位置格式 转化成 经营范围表的格式
            # ['天津市', []] -> ['天津', '天津市']
            # 高德api 通过经纬度查询地址名称特殊情况下 返回的为 [] （空list）
            if (
                address_list[0] in ["北京市", "上海市", "天津市", "重庆市"]
                and address_list[1] == []
            ):
                address_list[1] = address_list[0]
                address_list[0] = address_list[0][:2]
            # 这个可能是因为 ['河南省', [], '济源市']
            # 将高德api 返回的centroid 位置格式转化成 经营范围表的格式
            if address_list[1] == []:
                address_list[1] = address_list[2]
            # 高德返回格式 -> 经营范围表格式
            # ['广东省', '东莞市', '[]', '东坑镇'] -> ['广东省', '东莞市', '东坑镇', '-1']
            # 红包扫码表 df_total 中 的特殊符号是 '[]' 字符串！！！！
            if address_list[1] == "东莞市" and address_list[2] == []:
                address_list[2] = address_list[3]
                address_list[3] = "-1"

            # 新增
            df_copy_to_use = df_scope.copy()
            df_copy_to_use = df_copy_to_use.loc[
                :, ["PROVINCE", "CITY", "DISTRICT", "STREET"]
            ]
            flag = True
            for j in range(
                len(df_copy_to_use)
            ):  # if df_copy_to_use 是空，不会进此循环， flag 永远是True.

                scope_list = df_copy_to_use.loc[j, :].values.flatten().tolist()

                if is_belong_to(address_list, scope_list):
                    flag = False  # 只要在一个有效经营范围内，就不是异地
                    break

            if flag:
                df_centroids_with_remote.loc[i, "is_remote"] = 1

            else:
                df_centroids_with_remote.loc[i, "is_remote"] = 0
        else:
            df_centroids_with_remote.loc[i, "is_remote"] = (
                cluster_label  # 噪声点的簇心(label: -1)的"is_remote"标签也标记为-1.
            )
    return df_centroids_with_remote


def verify_points_within_scope(
    df_scanning_locations: pd.DataFrame, df_scope: pd.DataFrame
) -> pd.DataFrame:
    """
    给定开瓶扫码df和经营范围df, 判断每一条开瓶记录是否在给定的经营范围内。

    Parameters
    ----------
    df_scanning_locations: pd.DataFrame
        开瓶扫码记录df。

    df_scope: pd.DataFrame
        经营范围信息df。

    Returns
    -------
    pd.DataFrame: 增加扫码是否超范围标签("is_remote_point_new")的开瓶扫码df.
    """

    df_scanning_locations_with_new_remote_label = (
        df_scanning_locations.copy().reset_index(drop=True)
    )
    df_scanning_locations_with_new_remote_label["is_remote_point_new"] = -100
    # 0：本地；1：异地；

    for i in range(len(df_scanning_locations_with_new_remote_label)):
        address_list = (
            df_scanning_locations_with_new_remote_label.loc[
                i, ["OPEN_PROVINCE", "OPEN_CITY", "OPEN_DISTRICT", "OPEN_TOWN"]
            ]
            .values.flatten()
            .tolist()
        )
        # print(address_list)

        # 红包扫码表格式 -> 经营范围表格式
        # ['天津市', '天津市'] -> ['天津', '天津市']
        if address_list[0] in ["天津市", "北京市", "上海市", "重庆市"]:
            address_list[0] = address_list[0][:2]

        # 红包扫码表格式 -> 经营范围表格式
        # ['广东省', '东莞市', '[]', '东坑镇'] -> ['广东省', '东莞市', '东坑镇', '-1']
        # 红包扫码表 df_total 中 的特殊符号是 '[]' 字符串！！！！
        if address_list[1] == "东莞市" and address_list[2] == "[]":
            address_list[2] = address_list[3]
            address_list[3] = "-1"

        flag = True
        df_copy_to_use = df_scope.copy()
        for j in range(len(df_copy_to_use)):
            # 新增
            df_copy_to_use = df_copy_to_use.loc[
                :, ["PROVINCE", "CITY", "DISTRICT", "STREET"]
            ]
            scope_list = df_copy_to_use.loc[j, :].values.flatten().tolist()
            if is_belong_to(address_list, scope_list):
                flag = False
                break
        # print(flag)
        if flag:
            df_scanning_locations_with_new_remote_label.loc[
                i, "is_remote_point_new"
            ] = 1
        else:
            df_scanning_locations_with_new_remote_label.loc[
                i, "is_remote_point_new"
            ] = 0

    return df_scanning_locations_with_new_remote_label


def clustering_and_verify_remote_for_dealer(
    df_cleaned: pd.DataFrame,
    dealer_id: str,
    product_group_id: str,
    start_date_str: str,
    end_date_str: str,
    radius: float,
    min_samples: int,
    config_file_path: str,
    dealer_scope_dict_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    为单个经销商的扫码聚簇, 生成簇维度df, 分别判断扫码点和簇心是否异地。同时增加许多额外指标, 返回集成后的开瓶扫码维度df和簇维度df, 以及该经销商品项范围是否收录flag。

    Parameters
    ----------
    df_cleaned: pd.DataFrame
        经过清洗后的包含扫码记录数据的数据帧。

    dealer_id: str
        给定的经销商编码。

    product_group_id: str
        给定的产品品项编码。

    start_date_str: str
        所选时间段的第一天。格式为"%Y-%m-%d"，例如"2024-12-01".

    end_date_str: str
        所选时间段的最后一天。格式为"%Y-%m-%d"，例如"2024-12-31".

    radius: float
        模型所用基于经纬度的DBSCAN聚类算法的 半径大小(邻域距离阈值)。在模型中可以粗略的理解为聚簇时的半径大小(地球球面距离, 如4km)。

    min_samples: int
        模型所用基于经纬度的DBSCAN聚类算法的 成簇最少样本数(邻域中数据点个数的最小个数)。在模型中可以粗略地理解为聚簇时邻域内的最少开瓶扫码数(如6瓶)。

    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    dealer_scope_dict_path: str
        经销商经营范围字典主数据的路径。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, int]
        pd.DataFrame: df_centroids_with_remote_label 包含簇维度信息的df。
        pd.DataFrame: df_scanning_locations_with_remote_labels 包含扫码开瓶信息的df.
        int: 经销商品项经营范围是否收录。

    Descripition
    ------------

    """
    with open(dealer_scope_dict_path, "rb") as f:
        dealer_scope_dict = pickle.load(f)

    # df_scanning_locations_with_labels, df_centroids = (
    #     get_scanning_locations_and_centroids(
    #         df_cleaned,
    #         dealer_id,
    #         product_group_id,
    #         start_date_str,
    #         end_date_str,
    #         radius,
    #         min_samples,
    #         config_file_path,
    #     )
    # )

    df_scanning_locations_with_labels = find_clusters_for_dealer(
        df_cleaned,
        dealer_id,
        product_group_id,
        start_date_str,
        end_date_str,
        radius,
        min_samples,
    )
    # if df_scanning_loactions.empty:
    #     return pd.DataFrame(), pd.DataFrame()

    df_centroids = get_centroids(df_scanning_locations_with_labels, config_file_path)
    # df_scanning_loactions = df_scanning_loactions.reset_index(drop=True)

    # df_valid_scope, is_within_archive = find_valid_regions_monthly_application(
    #     dealer_id, product_group_id, start_date_str, end_date_str, dealer_scope_dict
    # )

    df_valid_scope, is_within_archive = find_equivalent_regions(
        dealer_id, product_group_id, start_date_str, dealer_scope_dict
    )

    # df_valid_scope_short = df_valid_scope[["PROVINCE", "CITY", "DISTRICT", "STREET"]]
    df_centroids_with_remote_label = verify_centroids_within_scope(
        df_centroids, df_valid_scope
    )
    df_scanning_locations_with_remote_labels = verify_points_within_scope(
        df_scanning_locations_with_labels, df_valid_scope
    )

    ##################### 这里增加一个 有效经营范围是否 为空， 用于最后单独取
    df_centroids_with_remote_label["is_dealer_no_valid_scope"] = 0
    if df_valid_scope.empty:
        df_centroids_with_remote_label["is_dealer_no_valid_scope"] = 1

    df_centroids_with_remote_label["is_dealer_within_archive"] = is_within_archive
    # df_centroids_with_remote_label['product_group_id'] = product_group_id
    df_centroids_with_remote_label["dealer_total_scanning_count"] = len(
        df_scanning_locations_with_remote_labels
    )
    df_centroids_with_remote_label["dealer_total_box_count"] = (
        df_scanning_locations_with_remote_labels["BARCODE_CORNER"].nunique()
    )

    # df_centroids_with_remote_label['dealer_valid_scope'] = df_valid_scope # 无法这样赋值
    # df_centroids_with_remote_label['dealer_valid_scope'] = df_centroids_with_remote_label['dealer_valid_scope'].apply(lambda x: df_valid_scope)
    df_centroids_with_remote_label["dealer_valid_scope"] = (
        df_centroids_with_remote_label.apply(lambda x: df_valid_scope, axis=1)
    )

    df_scanning_locations_with_remote_labels["is_dealer_within_archive"] = (
        is_within_archive
    )

    for label in df_centroids_with_remote_label.cluster_label:

        df_locations_label = df_scanning_locations_with_remote_labels[
            df_scanning_locations_with_remote_labels.cluster_label == label
        ].copy()
        cluster_mask = df_centroids_with_remote_label["cluster_label"] == label

        # 每个label内的扫码点数
        df_centroids_with_remote_label.loc[
            cluster_mask, "scanning_count_within_cluster"
        ] = len(df_locations_label)

        df_centroids_with_remote_label.loc[cluster_mask, "box_count_within_cluster"] = (
            df_locations_label["BARCODE_CORNER"].nunique()
        )

        if label != -1:
            points = list(
                zip(df_locations_label.LATITUDE, df_locations_label.LONGITUDE)
            )
            centroid_for_label = (
                df_centroids_with_remote_label.loc[
                    df_centroids_with_remote_label["cluster_label"] == label, "LATITUDE"
                ].iloc[0],
                df_centroids_with_remote_label.loc[
                    df_centroids_with_remote_label["cluster_label"] == label,
                    "LONGITUDE",
                ].iloc[0],
            )

            # # 到所有点的质心 的距离
            # df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == label, 'dis_to_overall_centroid'] = np.round(geodesic(centroid_for_label, centroid).kilometers, 2)

            distances = [
                geodesic(point, centroid_for_label).kilometers for point in points
            ]
            df_centroids_with_remote_label.loc[
                df_centroids_with_remote_label["cluster_label"] == label,
                "avg_distance_within_cluster",
            ] = round(np.mean(distances), 2)
            df_centroids_with_remote_label.loc[
                df_centroids_with_remote_label["cluster_label"] == label,
                "std_distance_within_cluster",
            ] = round(np.std(distances), 2)

    df_centroids_with_remote_label["scanning_ratio_for_cluster"] = (
        df_centroids_with_remote_label["scanning_count_within_cluster"]
        / df_centroids_with_remote_label["dealer_total_scanning_count"]
    )
    df_centroids_with_remote_label["box_count_ratio_for_cluster"] = (
        df_centroids_with_remote_label["box_count_within_cluster"]
        / df_centroids_with_remote_label["dealer_total_box_count"]
    )
    df_centroids_with_remote_label = df_centroids_with_remote_label.reset_index(
        drop=True
    )
    df_scanning_locations_with_remote_labels = (
        df_scanning_locations_with_remote_labels.reset_index(drop=True)
    )

    return (
        df_centroids_with_remote_label,
        df_scanning_locations_with_remote_labels,
        is_within_archive,
    )


def find_hotspots_for_region(
    df_cleaned: pd.DataFrame,
    product_group_id: str,
    start_date_str: str,
    end_date_str: str,
    radius: float,
    min_samples: int,
    config_file_path: str,
    dealer_scope_dict_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple]]:
    """
    生成整个大区的开瓶扫码聚簇信息df, 簇维度信息df, 以及大区内该品项未收录的经销商名单。

    Parameters
    ----------
    df_cleaned: pd.DataFrame
        经过清洗后的包含扫码记录数据的数据帧。

    dealer_id: str
        给定的经销商编码。

    product_group_id: str
        给定的产品品项编码。

    start_date_str: str
        所选时间段的第一天。格式为"%Y-%m-%d"，例如"2024-12-01".

    end_date_str: str
        所选时间段的最后一天。格式为"%Y-%m-%d"，例如"2024-12-31".

    radius: float
        模型所用基于经纬度的DBSCAN聚类算法的 半径大小(邻域距离阈值)。在模型中可以粗略的理解为聚簇时的半径大小(地球球面距离, 如4km)。

    min_samples: int
        模型所用基于经纬度的DBSCAN聚类算法的 成簇最少样本数(邻域中数据点个数的最小个数)。在模型中可以粗略地理解为聚簇时邻域内的最少开瓶扫码数(如6瓶)。

    config_file_path: str
        配置文件的路径，包含高德 API 密钥。

    dealer_scope_dict_path: str
        经销商经营范围字典主数据的路径。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[tuple]]
        pd.DataFrame: df_centroids_with_remote_label 包含簇维度信息的df。
        pd.DataFrame: df_scanning_locations_with_remote_labels 包含扫码开瓶信息的df.
        list[tuple[str, str]]: 该大区经销商品项未收录的集合。("dealer_id", "product_group_id")
    """

    # 增加'dealer_polyline_points_list_total', 'dealer_adcodes' -> df_total_centroids
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    ids_within_region = df_cleaned[
        (df_cleaned["CUST_SCAN_DATE"] >= start_date)
        & (df_cleaned["CUST_SCAN_DATE"] <= end_date)
        & (df_cleaned["PRODUCT_GROUP_CODE"] == product_group_id)
    ]["BELONG_DEALER_NO"].unique()

    centroids_list = []
    scanning_locations_list = []
    # dealers_change_remote_scope = {}
    dealers_not_within_archive = []

    for dealer_id in ids_within_region:

        (
            df_centroids_with_remote_label,
            df_scanning_locations_with_remote_labels,
            is_within_archive,
        ) = clustering_and_verify_remote_for_dealer(
            df_cleaned,
            dealer_id,
            product_group_id,
            start_date_str,
            end_date_str,
            radius,
            min_samples,
            config_file_path,
            dealer_scope_dict_path,
        )

        if not df_centroids_with_remote_label.empty:

            df_centroids_with_remote_label["dealer_id"] = dealer_id
            df_valid_scope = df_centroids_with_remote_label.loc[0, "dealer_valid_scope"]
            # print(dealer_id)
            polyline_points_list_total, adcodes = (
                get_polylines_adcodes_for_valid_regions(
                    df_valid_scope, config_file_path
                )
            )

            df_centroids_with_remote_label["dealer_polyline_points_list_total"] = (
                df_centroids_with_remote_label.apply(
                    lambda x: polyline_points_list_total, axis=1
                )
            )
            df_centroids_with_remote_label["dealer_adcodes"] = (
                df_centroids_with_remote_label.apply(lambda x: adcodes, axis=1)
            )
            centroids_list.append(df_centroids_with_remote_label)

        if not df_scanning_locations_with_remote_labels.empty:
            scanning_locations_list.append(df_scanning_locations_with_remote_labels)

        if not is_within_archive:
            dealers_not_within_archive.append((dealer_id, product_group_id))

    try:
        df_total_centroids = pd.concat(centroids_list).reset_index(drop=True)

    except Exception as e:
        df_total_centroids = pd.DataFrame()  # 创建一个空的 DataFrame
        print(f"发生错误: {e}")

    try:
        df_total_scanning_locations = pd.concat(scanning_locations_list).reset_index(
            drop=True
        )
    except Exception as e:
        df_total_scanning_locations = pd.DataFrame()  # 创建一个空的 DataFrame
        print(f"发生错误: {e}")

    return df_total_centroids, df_total_scanning_locations, dealers_not_within_archive


def calculate_distances_to_local_centroids_for_centroids(
    df_total_scanning_locations, df_total_centroids, dealers_not_within_archive
):
    # ids_within_archive = set(df_total_centroids.dealer_id.unique()) - set([item[0] for item in dealers_not_within_archive])
    ids_within_archive = list(
        df_total_centroids.loc[
            df_total_centroids["is_dealer_within_archive"] == 1, "dealer_id"
        ].drop_duplicates()
    )
    centroids_all_local_points_dict = {}
    centroids_all_local_hotspots_dict = {}

    for id in ids_within_archive:

        df_locations = df_total_scanning_locations.loc[
            df_total_scanning_locations["BELONG_DEALER_NO"] == id, :
        ]
        df_centroids = df_total_centroids.loc[df_total_centroids["dealer_id"] == id, :]

        # 所有为本地的 扫码点 的质心
        df_all_local_points = df_locations.loc[
            df_total_scanning_locations["is_remote_point_new"] == 0,
            ["LATITUDE", "LONGITUDE"],
        ]
        if len(df_all_local_points) != 0:
            centroid_all_local_points = df_all_local_points.mean()
            centroids_all_local_points_dict[id] = (
                centroid_all_local_points["LATITUDE"],
                centroid_all_local_points["LONGITUDE"],
            )
        else:
            centroids_all_local_points_dict[id] = np.nan

        # 所有在本地热点内的 扫码点 的质心
        local_hotspot_labels = df_centroids.loc[
            (~df_centroids["cluster_label"].isin([-2, -1]))
            & (df_centroids["is_remote"] == 0),
            "cluster_label",
        ].unique()
        df_all_local_hotspots = df_locations.loc[
            df_locations["cluster_label"].isin(local_hotspot_labels),
            ["LATITUDE", "LONGITUDE"],
        ]
        if len(df_all_local_hotspots) != 0:
            centroid_all_local_hotspots = df_all_local_hotspots.mean()
            centroids_all_local_hotspots_dict[id] = (
                centroid_all_local_hotspots["LATITUDE"],
                centroid_all_local_hotspots["LONGITUDE"],
            )
        else:
            centroids_all_local_hotspots_dict[id] = np.nan
    # 将质心映射到 df_total_centroids 中
    df_total_centroids["centroid_all_local_points_coordinate"] = df_total_centroids[
        "dealer_id"
    ].map(centroids_all_local_points_dict)
    df_total_centroids["centroid_all_local_hotspots_coordinate"] = df_total_centroids[
        "dealer_id"
    ].map(centroids_all_local_hotspots_dict)

    def calculate_distance_hotspots(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if (
            pd.isna(row["LATITUDE"])
            or pd.isna(row["LONGITUDE"])
            or pd.isna(row["centroid_all_local_hotspots_coordinate"])
        ):
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_hotspots_coordinate"],
        ).kilometers

    def calculate_distance_points(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if (
            pd.isna(row["LATITUDE"])
            or pd.isna(row["LONGITUDE"])
            or pd.isna(row["centroid_all_local_points_coordinate"])
        ):
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_points_coordinate"],
        ).kilometers

    # 使用 apply 方法计算距离，并保留缺失值的行
    df_total_centroids["dis_to_all_local_hotspots_centroid"] = df_total_centroids.apply(
        calculate_distance_hotspots, axis=1
    )
    df_total_centroids["dis_to_all_local_hotspots_centroid"] = np.round(
        df_total_centroids["dis_to_all_local_hotspots_centroid"], 2
    )

    df_total_centroids["dis_to_all_local_points_centroid"] = df_total_centroids.apply(
        calculate_distance_points, axis=1
    )
    df_total_centroids["dis_to_all_local_points_centroid"] = np.round(
        df_total_centroids["dis_to_all_local_points_centroid"], 2
    )

    return df_total_centroids


# def find_closest_point_geodesic(fixed_point, coordinates):
#     min_distance = float('inf')
#     # min_distance = float("99999")
#     closest_point = None

#     for coord in coordinates:
#         coordinate = (coord[0], coord[1])
#         distance = geodesic(
#             fixed_point, coordinate
#         ).kilometers  # Distance in kilometers
#         if distance < min_distance:
#             min_distance = distance
#             closest_point = coord

#     return closest_point, min_distance


def calculate_min_distance_to_border(df_total_centroids):

    df_total_hotspots = df_total_centroids.loc[
        (~(df_total_centroids["cluster_label"].isin([-1, -2])))
        & (df_total_centroids["is_dealer_within_archive"] == 1),
        :,
    ].reset_index(drop=True)
    # df_total_hotspots['dis_border'] = float('inf')

    for i in range(len(df_total_hotspots)):

        polyline_points_list_total = df_total_hotspots.loc[
            i, "dealer_polyline_points_list_total"
        ]
        fixed_point = (
            df_total_hotspots.loc[i, "LATITUDE"],
            df_total_hotspots.loc[i, "LONGITUDE"],
        )
        flat_coordinates = [
            point for sublist in polyline_points_list_total for point in sublist
        ]

        closest_point, min_distance = find_closest_point_geodesic(
            fixed_point, flat_coordinates
        )

        df_total_hotspots.loc[i, "dis_border"] = round(min_distance, 2)

    df_total_hotspots_to_merge = df_total_hotspots.loc[
        :, ["dealer_id", "cluster_label", "dis_border"]
    ]
    df_total_centroids_new = pd.merge(
        df_total_centroids,
        df_total_hotspots_to_merge,
        on=["dealer_id", "cluster_label"],
        how="left",
    )

    df_total_centroids_new.loc[
        (df_total_centroids_new["is_remote"] == 0)
        & (~(df_total_centroids_new["cluster_label"].isin([-1, -2]))),
        "dis_border",
    ] = -df_total_centroids_new["dis_border"]

    return df_total_centroids_new


def find_hotspots_main(
    df_cleaned,
    product_group_id,
    start_date_str,
    end_date_str,
    radius,
    min_samples,
    config_file_path,
    dealer_scope_dict_path,
):

    df_total_centroids, df_total_scanning_locations, dealers_not_within_archive = (
        find_hotspots_for_region(
            df_cleaned,
            product_group_id,
            start_date_str,
            end_date_str,
            radius,
            min_samples,
            config_file_path,
            dealer_scope_dict_path,
        )
    )

    df_total_centroids = calculate_distances_to_local_centroids_for_centroids(
        df_total_scanning_locations, df_total_centroids, dealers_not_within_archive
    )

    df_total_centroids = calculate_min_distance_to_border(df_total_centroids)

    return df_total_centroids, df_total_scanning_locations


def find_hotspots_continue_for_dense_main(
    df_total_centroids_sparse,
    df_total_scanning_locations_sparse,
    large_hotspots_threshold,
    df_cleaned,
    product_group_id,
    start_date_str,
    end_date_str,
    radius_dense,
    min_samples_dense,
    config_file_path,
    dealer_scope_dict_path,
):
    """
    增加 df_total_centroids_sparse
    """

    large_hotspots_mask = (
        ~(df_total_centroids_sparse["cluster_label"].isin([-1, -2]))
    ) & (
        df_total_centroids_sparse["scanning_count_within_cluster"]
        >= large_hotspots_threshold
    )

    df_total_centroids_sparse.loc[large_hotspots_mask, "is_large_hotspot"] = 1
    df_total_centroids_sparse.loc[~large_hotspots_mask, "is_large_hotspot"] = 0

    df_large_hotspots = df_total_centroids_sparse.loc[large_hotspots_mask, :]

    df_large_hotspots_label = df_large_hotspots.loc[:, ["dealer_id", "cluster_label"]]
    df_large_hotspots_label = df_large_hotspots_label.reset_index(drop=True).rename(
        columns={"dealer_id": "BELONG_DEALER_NO"}
    )

    df_large_hotspots_scanning_locations = pd.merge(
        df_large_hotspots_label,
        df_total_scanning_locations_sparse,
        on=["BELONG_DEALER_NO", "cluster_label"],
        how="left",
    )
    # print(len(df_large_hotspots_scanning_locations))
    # print(len(df_total_scanning_locations_sparse))

    print(
        f"大型热点内的扫码点数量占比: {len(df_large_hotspots_scanning_locations) / len(df_total_scanning_locations_sparse)}"
    )
    set_diff = set(df_large_hotspots_scanning_locations.columns) - set(
        df_cleaned.columns
    )
    df_large_hotspots_scanning_locations_cleaned = (
        df_large_hotspots_scanning_locations.drop(columns=set_diff)
    )

    # dense part
    # radius_dense, min_samples_dense = dbscan_parameters_tuple_dense

    (
        df_total_centroids_dense,
        df_total_scanning_locations_dense,
        dealers_not_within_archive_dense,
    ) = find_hotspots_for_region(
        df_large_hotspots_scanning_locations_cleaned,
        product_group_id,
        start_date_str,
        end_date_str,
        radius_dense,
        min_samples_dense,
        config_file_path,
        dealer_scope_dict_path,
    )
    df_total_centroids_dense = df_total_centroids_dense.drop(
        columns=[
            "dealer_total_scanning_count",
            "scanning_ratio_for_cluster",
            "dealer_total_box_count",
            "box_count_ratio_for_cluster",
        ]
    )

    # 在dense 分簇里计算距本地扫码点坐标也要根据sparse里的所有点
    # dis_to_all_local_points_centroid, centroid_all_local_points_coordinate
    df_all_local_points_location = df_total_centroids_sparse.loc[
        :, ["dealer_id", "centroid_all_local_points_coordinate"]
    ].drop_duplicates()
    df_total_centroids_dense = pd.merge(
        df_total_centroids_dense,
        df_all_local_points_location,
        on="dealer_id",
        how="left",
    )

    def calculate_distance_points(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if (
            pd.isna(row["LATITUDE"])
            or pd.isna(row["LONGITUDE"])
            or pd.isna(row["centroid_all_local_points_coordinate"])
        ):
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_points_coordinate"],
        ).kilometers

    df_total_centroids_dense["dis_to_all_local_points_centroid"] = (
        df_total_centroids_dense.apply(calculate_distance_points, axis=1)
    )
    df_total_centroids_dense["dis_to_all_local_points_centroid"] = np.round(
        df_total_centroids_dense["dis_to_all_local_points_centroid"], 2
    )

    # dis_to_all_local_hotspots_centroid, centroid_all_local_hotspots_coordinate
    df_all_local_hotspots_location = df_total_centroids_sparse.loc[
        :, ["dealer_id", "centroid_all_local_hotspots_coordinate"]
    ].drop_duplicates()
    df_total_centroids_dense = pd.merge(
        df_total_centroids_dense,
        df_all_local_hotspots_location,
        on="dealer_id",
        how="left",
    )

    # print(df_total_centroids_dense.shape)
    def calculate_distance_hotspots(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if (
            pd.isna(row["LATITUDE"])
            or pd.isna(row["LONGITUDE"])
            or pd.isna(row["centroid_all_local_hotspots_coordinate"])
        ):
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_hotspots_coordinate"],
        ).kilometers

    df_total_centroids_dense["dis_to_all_local_hotspots_centroid"] = (
        df_total_centroids_dense.apply(calculate_distance_hotspots, axis=1)
    )
    df_total_centroids_dense["dis_to_all_local_hotspots_centroid"] = np.round(
        df_total_centroids_dense["dis_to_all_local_hotspots_centroid"], 2
    )

    # "scanning_ratio_for_cluster" ,"dealer_total_scanning_count",
    # "box_count_ratio_for_cluster", "dealer_total_box_count"
    df_total_centroids_dense = pd.merge(
        df_total_centroids_dense,
        df_total_centroids_sparse[
            ["dealer_id", "dealer_total_scanning_count", "dealer_total_box_count"]
        ].drop_duplicates(),
        on="dealer_id",
        how="left",
    )
    df_total_centroids_dense["scanning_ratio_for_cluster"] = (
        df_total_centroids_dense["scanning_count_within_cluster"]
        / df_total_centroids_dense["dealer_total_scanning_count"]
    )
    df_total_centroids_dense["box_count_ratio_for_cluster"] = (
        df_total_centroids_dense["box_count_within_cluster"]
        / df_total_centroids_dense["dealer_total_box_count"]
    )

    # print(df_total_centroids_dense.columns)
    df_total_centroids_dense = calculate_min_distance_to_border(
        df_total_centroids_dense
    )

    return (
        df_total_centroids_dense,
        df_total_scanning_locations_dense,
        df_total_centroids_sparse,
    )
