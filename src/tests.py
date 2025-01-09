from .data_clean import main_generate_clean_region_data
from .find_hotspots import (
    main_find_hotspots,
    find_valid_regions,
    plot_clusters_with_folium,
)
from IPython.display import display
from tabulate import tabulate


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


def test_find_hotspots_run(
    df_total_path,
    dealer_scope_dict_path,
    config_file_path,
    dealer_region_name,
    product_group_id,
    date_str_tuple,
    dbscan_parameters_tuple,
):

    start_date_str, end_date_str = date_str_tuple
    radius, min_samples = dbscan_parameters_tuple

    df_cleaned = main_generate_clean_region_data(dealer_region_name, df_total_path)

    df_total_centroids, df_total_scanning_locations = main_find_hotspots(
        df_cleaned,
        product_group_id,
        start_date_str,
        end_date_str,
        radius,
        min_samples,
        config_file_path,
        dealer_scope_dict_path,
    )

    return df_total_centroids, df_total_scanning_locations


def test_show_all_results(
    df_total_scanning_locations,
    df_total_centroids,
    df_result_within_archive,
    df_suspicious_hotspots_parameters,
    dealer_scope_dict_path,
    start_date_str,
    end_date_str,
    dealer_region_name,
    radius,
    min_samples,
):

    suspicious_dealers_overall_cols = [
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "PRODUCT_GROUP_NAME",
    ]

    dealer_info_cols = [
        "remote_ratio",
        "total_scanning_count",
        "remote_scanning_count",
        "remote_hotspot_ratio",
        "dealer_hotspot_count",
        "dealer_remote_hotspot_count",
        "dealer_suspicious_hotspot_count",
        "suspicious_hotspot_ratio",
        "dealer_total_box_count",
    ]

    dealer_cluster_cols = [
        "cluster_label",
        "province",
        "city",
        "district",
        "street",
        "scanning_count_within_cluster",
        "is_remote",
        "is_suspicious",
        "ratio_scanning_count",
        "dis_to_all_local_hotspots_centroid",
        "std_distance_within_cluster",
        "box_count_within_cluster",
        "box_count_ratio_for_cluster",
    ]

    rename_dict = {
        "remote_ratio": "开瓶异地率",
        "total_scanning_count": "扫码总量",
        "remote_scanning_count": "异地扫码量",
        "remote_hotspot_ratio": "热点异地率",
        "dealer_hotspot_count": "热点总量",
        "dealer_remote_hotspot_count": "异地热点数量",
        "suspicious_hotspot_ratio": "热点可疑率",
        "dealer_suspicious_hotspot_count": "可疑热点数量",
        "cluster_label": "簇标签",
        "province": "省",
        "city": "市",
        "district": "区",
        "street": "镇街",
        "scanning_count_within_cluster": "簇内点数量",
        "is_remote": "是否异地",
        "is_suspicious": "是否可疑",
        "ratio_scanning_count": "扫码量占比",
        "dis_to_overall_centroid": "热点质心到总质心的距离（质心距离）",
        "dis_to_all_local_hotspots_centroid": "到本地热点总质心的距离",
        "avg_distance_within_cluster": "热点内各点到该热点质心距离的平均值（热点内距离均值）",
        "std_distance_within_cluster": "标准差-簇内离散度",
        "ratio_dis_to_overall_centroid": '质心距离/本地热点的"质心距离”均值',
        "ratio_avg_distance_within_cluster": "热点内距离均值/所有在档本地热点的均值",
        "ratio_std_distance_within_cluster": "热点内距离标准差/所有在档本地热点的均值",
        "OPEN_PROVINCE": "开瓶省",
        "OPEN_CITY": "开瓶市",
        "count": "数量",
        "is_remote_city": "是否为异地二级城市",
        "dealer_total_box_count": "总开箱数",
        "box_count_within_cluster": "箱数",
        "box_count_ratio_for_cluster": "箱数占比",
    }

    df_model_parameters = pd.DataFrame(
        {
            "簇半径": radius,
            "簇内最少样本数": min_samples,
        },
        index=[0],
    )

    df_model_parameters = pd.concat(
        [df_model_parameters, df_suspicious_hotspots_parameters], axis=1
    )
    product_group_name = df_result_within_archive.loc[0, "PRODUCT_GROUP_NAME"]
    print("全部经销商测试-----------------------------------------------")
    print(
        f"-------------------基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下-------------------"
    )
    print()
    print("---模型参数---")
    print(
        tabulate(
            df_model_parameters, headers="keys", tablefmt="pretty", showindex=False
        )
    )
    print()
    print(f"扫码经销商总数： {df_total_scanning_locations.BELONG_DEALER_NO.nunique()}")
    print(f"当前规则下在档经销商数量:{df_result_within_archive.shape[0]}")
    print("---在档经销商汇总表---")
    print()
    df_suspicious_dealers_to_show = df_result_within_archive.loc[
        :, suspicious_dealers_overall_cols
    ]
    print(
        tabulate(
            df_suspicious_dealers_to_show,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print("-" * 200)
    # print(f'出现扫码但不在合同档案里的经销商数量:{df_dealers_without_archive.shape[0]}')

    print()
    print()
    print()
    print("经销商详细信息如下:")
    print("*" * 150)

    for i in range(len(df_result_within_archive)):
        dealer_id = df_result_within_archive.loc[i, "BELONG_DEALER_NO"]
        dealer_name = df_result_within_archive.loc[i, "BELONG_DEALER_NAME"]
        product_group_id = df_result_within_archive.loc[i, "PRODUCT_GROUP_CODE"]
        product_group_name = df_result_within_archive.loc[i, "PRODUCT_GROUP_NAME"]

        print(f"经销商: {dealer_id} - {dealer_name} - {product_group_name}")
        print("-" * 100)

        # for col in dealer_info_cols:
        #     print(f'{rename_dict[col]}: {df_suspicious_dealers.at[i, col]}')
        with open(dealer_scope_dict_path, "rb") as f:
            dealer_scope_dict = pickle.load(f)
        df_busiuness_scope = dealer_scope_dict[(dealer_id, product_group_id)]

        print(f"---经营范围---")
        print(
            tabulate(
                df_busiuness_scope, headers="keys", tablefmt="pretty", showindex=False
            )
        )
        df_dealer_info = df_result_within_archive.loc[i, dealer_info_cols].to_frame().T
        df_dealer_info = df_dealer_info.rename(columns=rename_dict)
        print()
        print(f"---范围内经销商异地信息---")
        print(
            tabulate(df_dealer_info, headers="keys", tablefmt="pretty", showindex=False)
        )

        # print("=" * 40)

        # for cols in [dealer_cluster_cols, dealer_cluster_cols_2]:
        for cols in [dealer_cluster_cols]:

            df_dealer_cluster = df_total_centroids.loc[
                df_total_centroids.dealer_id == dealer_id, cols
            ]
            df_dealer_cluster = df_dealer_cluster.loc[
                ~(df_dealer_cluster["cluster_label"].isin([-2, -1])), :
            ]
            print()
            print(f"---经销商热点信息---")
            print("可疑热点标签")
            suspicious_labels = list(
                map(
                    int,
                    list(
                        df_dealer_cluster.loc[
                            df_dealer_cluster["is_suspicious"] == 1, "cluster_label"
                        ].values
                    ),
                )
            )
            print(suspicious_labels)
            df_dealer_cluster = df_dealer_cluster.rename(columns=rename_dict)
            print(
                tabulate(
                    df_dealer_cluster,
                    headers="keys",
                    tablefmt="pretty",
                    showindex=False,
                )
            )

        # df_city_for_dealer = df_total_scanning_locations.loc[df_total_scanning_locations['BELONG_DEALER_NO'] == dealer_id , ['OPEN_PROVINCE', 'OPEN_CITY']]
        # df_city_count = df_city_for_dealer.value_counts().reset_index()

        df_scanning_locations = df_total_scanning_locations.loc[
            df_total_scanning_locations.BELONG_DEALER_NO == dealer_id
        ]
        df_city_count = (
            df_scanning_locations[["OPEN_PROVINCE", "OPEN_CITY"]]
            .value_counts()
            .reset_index()
        )
        # 扫码表格式 -> 经营范围表格式
        # ['天津市', '天津市'] -> ['天津', '天津市']
        df_city_count["OPEN_PROVINCE"] = df_city_count["OPEN_PROVINCE"].apply(
            lambda x: x[:2] if x in ["天津市", "北京市", "上海市", "重庆市"] else x
        )

        # print(df_city_count)
        df_valid_scope, _ = find_valid_regions(
            dealer_id, product_group_id, start_date_str, dealer_scope_dict
        )

        df_count_merged = pd.DataFrame()

        if df_valid_scope.empty:

            df_valid_scope_end, _ = find_valid_regions(
                dealer_id, product_group_id, end_date_str, dealer_scope_dict
            )

            if not df_valid_scope_end.empty:
                print(
                    f"该经销商{start_date_str}时有效经营范围为空， 按照{end_date_str}时的有效范围计算。"
                )
                df_valid_scope = df_valid_scope_end
                df_count_merged = df_city_count.merge(
                    df_valid_scope,
                    left_on=["OPEN_PROVINCE", "OPEN_CITY"],
                    right_on=["PROVINCE", "CITY"],
                    how="left",
                    indicator=True,
                )
                df_count_merged["is_remote_city"] = (
                    df_count_merged["_merge"] == "left_only"
                )
                df_count_merged = (
                    df_count_merged.loc[
                        :, ["OPEN_PROVINCE", "OPEN_CITY", "count", "is_remote_city"]
                    ]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                df_count_merged["is_remote_city"] = df_count_merged[
                    "is_remote_city"
                ].map({True: "是", False: "否"})

            else:
                print(f"该经销商{start_date_str}和{end_date_str}时有效经营范围都为空")
                df_count_merged = (
                    df_city_count.loc[:, ["OPEN_PROVINCE", "OPEN_CITY", "count"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                df_count_merged["is_remote_city"] = "是"

        else:
            df_count_merged = df_city_count.merge(
                df_valid_scope,
                left_on=["OPEN_PROVINCE", "OPEN_CITY"],
                right_on=["PROVINCE", "CITY"],
                how="left",
                indicator=True,
            )
            df_count_merged["is_remote_city"] = df_count_merged["_merge"] == "left_only"
            df_count_merged = (
                df_count_merged.loc[
                    :, ["OPEN_PROVINCE", "OPEN_CITY", "count", "is_remote_city"]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            df_count_merged["is_remote_city"] = df_count_merged["is_remote_city"].map(
                {True: "是", False: "否"}
            )

        df_count_merged = df_count_merged.rename(columns=rename_dict)
        print()

        print(f"开瓶二级城市数： {len(df_city_count)}")
        print(f"---可疑经销商开瓶城市统计表(大于两瓶的城市）---")
        print(
            tabulate(
                df_count_merged.loc[df_count_merged["数量"] > 2, :],
                headers="keys",
                tablefmt="pretty",
                showindex=False,
            )
        )

        m = plot_clusters_with_folium(
            df_scanning_locations, points_size=3, noise_size=3, if_save=False
        )
        display(m)


# 测试不同阈值对可以热点数量的变化
def suspicious_test(df_total_centroids, **kwargs):

    hotspot_mask = ~(df_total_centroids["cluster_label"].isin([-2, -1]))

    local_hotspots_std = df_total_centroids.loc[
        hotspot_mask & (df_total_centroids["is_remote"] == 0),
        "std_distance_within_cluster",
    ]

    default_thresholds = {
        "dis_hotspots_c_t": 35,
        "dis_points_c_t": 0,
        "ratio_scanning_t": 0.1,
        "scanning_count_t": 12,
        "std_quantile_t": 0.1,
        "box_count_t": 4,
    }

    thresholds = {**default_thresholds, **kwargs}

    dis_hotspots_c_t = thresholds["dis_hotspots_c_t"]
    dis_points_c_t = thresholds["dis_points_c_t"]
    ratio_scanning_t = thresholds["ratio_scanning_t"]
    scanning_count_t = thresholds["scanning_count_t"]

    std_distance_within_cluster_threshold = np.quantile(
        local_hotspots_std, thresholds["std_quantile_t"]
    )
    box_count_t = thresholds["box_count_t"]

    ##################################################################################################################################################
    # 设置可以热点条件
    # verison 1.0

    suspicious_mask = (hotspot_mask) & (df_total_centroids["is_remote"] == 1) & (
        df_total_centroids["dis_to_all_local_hotspots_centroid"] >= dis_hotspots_c_t
    ) & (df_total_centroids["dis_points_c_t"]) >= dis_points_c_t & (
        (df_total_centroids["ratio_scanning_count"] >= ratio_scanning_t)
        | (df_total_centroids["scanning_count_within_cluster"] >= scanning_count_t)
    ) & (
        (
            df_total_centroids["std_distance_within_cluster"]
            >= std_distance_within_cluster_threshold
        )
        | (
            (
                df_total_centroids["std_distance_within_cluster"]
                < std_distance_within_cluster_threshold
            )
            & (df_total_centroids["box_count_within_cluster"] >= box_count_t)
        )
    )

    df_total_centroids_with_suspicious_label = df_total_centroids.copy()
    df_suspicious_hotspots = df_total_centroids_with_suspicious_label.loc[
        suspicious_mask, :
    ].reset_index(drop=True)

    return (
        len(df_suspicious_hotspots),
        std_distance_within_cluster_threshold,
        df_suspicious_hotspots,
    )


def show_test_result(df_total_centroids, threshold_name):

    threshold_range = range(0, 105, 5)

    if threshold_name == "dis_hotspots_c_t":
        threshold_range = range(0, 105, 5)

    elif threshold_name == "dis_points_c_t":
        threshold_range = range(0, 50, 2)

    elif threshold_name == "ratio_scanning_t":
        threshold_range = np.arange(0, 0.55, 0.05)

    elif threshold_name == "scanning_count_t":
        threshold_range = range(6, 37, 1)

    elif threshold_name == "std_quantile_t":
        threshold_range = np.arange(0, 1.05, 0.05)

    elif threshold_name == "box_count_t":
        threshold_range = range(0, 7, 1)

    # 测试结果
    results = []

    for d in threshold_range:
        num, std, _ = suspicious_test(df_total_centroids, **{threshold_name: d})
        results.append((d, num))
    # print(std)

    # print(results)
    # 提取 x 和 y 值
    x, y = zip(*results)

    # 创建折线图
    plt.plot(
        x, y, marker="o", linestyle="-", color="b"
    )  # 'o' 是圆圈标记，'-' 是折线，'b' 是蓝色
    plt.title(f"{threshold_name}", fontproperties="SimSun")
    plt.xlabel(f"阈值: {threshold_name}", fontproperties="SimSun")
    plt.ylabel("可疑热点总数", fontproperties="SimSun")

    # 显示图形
    plt.grid(True)  # 添加网格
    plt.show()


def show_thresholds_suspicious_relations(df_total_centroids):
    thresholds = [
        "dis_hotspots_c_t",
        "dis_points_c_t",
        "ratio_scanning_t",
        "scanning_count_t",
        "std_quantile_t",
        "box_count_t",
    ]
    for threshold_name in thresholds:
        show_test_result(df_total_centroids, threshold_name)
