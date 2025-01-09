from IPython.display import display
from tabulate import tabulate

from .libs import (
    get_polyline_points,
    get_acode,
    find_valid_regions,
    find_valid_regions_monthly_application,
)
from IPython.display import display, HTML, Markdown

import folium
import geopandas as gpd
import pandas as pd
import pickle


def plot_clusters_with_folium(
    df_scanning_locations, points_size=5, noise_size=5, polyline_points_list=[]
):

    gdf = gpd.GeoDataFrame(
        df_scanning_locations,
        geometry=gpd.points_from_xy(
            df_scanning_locations.LONGITUDE, df_scanning_locations.LATITUDE
        ),
    )  # x -> longitude ; y -> latitude

    m = folium.Map(
        location=[gdf["LATITUDE"].mean(), gdf["LONGITUDE"].mean()],
        tiles="https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7",
        attr="高德-常规图",
        zoom_start=7,
    )

    unique_clusters = gdf["cluster_label"].sort_values().unique()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#aab7ff",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#f5b041",
        "#d62728",
    ]

    for i, cluster in enumerate(unique_clusters):
        if cluster != -1:  # 排除噪声点
            cluster_points = gdf[gdf["cluster_label"] == cluster]

            color_index = cluster % 20
            color = colors[color_index]
            for _, row in cluster_points.iterrows():

                folium.CircleMarker(
                    location=(row["LATITUDE"], row["LONGITUDE"]),
                    radius=points_size,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f"Cluster: {cluster}",
                ).add_to(m)

    # 绘制噪声点（可选）
    noise_points = gdf[gdf["cluster_label"] == -1]
    for _, row in noise_points.iterrows():
        folium.CircleMarker(
            location=(row["LATITUDE"], row["LONGITUDE"]),
            radius=noise_size,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.6,
            popup="Noise",
        ).add_to(m)

    legend_html = """
    <div style="
        position: fixed; 
        top: 0px; left: 0px; 
        width: 120px; height: auto; 
        border: 1px solid grey; 
        z-index: 9999; font-size: 12px; 
        background-color: rgba(255, 255, 255, 0.6); 
        padding: 2px; 
        border-radius: 8px; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.3); 
        overflow-y: auto; max-height: 200px;">
        <b>Cluster Legend</b><br>
    """
    for i, cluster in enumerate(unique_clusters):
        if cluster != -1:
            color_index = cluster % 20
            color = colors[color_index]
            legend_html += f'<i style="background:{color}; width: 15px; height: 15px; display:inline-block; margin-right: 5px;"></i> Cluster {cluster}<br>'
    legend_html += "</div>"

    # 将图例添加到地图
    folium.Marker(
        location=[gdf["LATITUDE"].mean(), gdf["LONGITUDE"].mean()],
        icon=folium.DivIcon(html=legend_html),
    ).add_to(m)

    if polyline_points_list:
        for polyline_points in polyline_points_list:
            folium.PolyLine(
                locations=polyline_points, color="BLUE", weight=2.5, opacity=1
            ).add_to(m)

    return m


def display_forth_title(title):
    display(Markdown(f"#### {title}"))


def display_fifth_title(title):
    display(Markdown(f"##### {title}"))


def main_show_results(
    df_dealer_results,
    df_total_scanning_locations,
    df_total_centroids,
    df_suspicious_hotspots_parameters,
    dealer_scope_dict_path,
    start_date_str,
    end_date_str,
    dealer_region_name,
):

    suspicious_dealers_overall_cols = [
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "PRODUCT_GROUP_NAME",
    ]

    # "簇半径": radius,
    # "簇内最少样本数": min_samples,
    # "距本地热点总质心的距离阈值": dis_hotspots_c_t,
    # "距本地扫码点总质心的距离阈值": dis_points_c_t,
    # "距边界最小距离阈值": dis_border_t,
    # "热点扫码量占比阈值": ratio_scanning_t,
    # "热点扫码量阈值": scanning_count_t,
    # "簇内离散度阈值": round(std_distance_within_cluster_threshold, 2),
    # "紧密热点的箱数阈值": box_count_t,

    # "radius": radius,
    # "min_samples": min_samples,
    # "dis_hotspots_c_t": dis_hotspots_c_t,
    # "dis_points_c_t": dis_points_c_t,
    # "dis_border_t": dis_border_t,
    # "ratio_scanning_t": ratio_scanning_t,
    # "scanning_count_t": scanning_count_t,
    # "std_distance_t": round(std_distance_within_cluster_threshold, 2),
    # "box_count_t": box_count_t,

    rename_dict = {
        "radius": "簇半径",
        "min_samples": "簇内最少样本数",
        "dis_hotspots_c_t": "距本地热点总质心的距离阈值",
        "dis_points_c_t": "距本地扫码点总质心的距离阈值",
        "dis_border_t": "距边界最小距离阈值",
        "ratio_scanning_t": "热点扫码量占比阈值",
        "scanning_count_t": "热点扫码量阈值",
        "std_distance_t": "热点离散度阈值",
        "box_count_t": "紧密热点的箱数阈值",
        "BELONG_DEALER_NO": "经销商编码",
        "BELONG_DEALER_NAME": "经销商名称",
        "PRODUCT_GROUP_NAME": "品项名称",
        # "remote_ratio": "开瓶异地率",
        # "total_scanning_count": "扫码总量",
        # "remote_scanning_count": "异地扫码量",
        # "remote_hotspot_ratio": "热点异地率",
        # "dealer_hotspot_count": "热点总量",
        # "dealer_remote_hotspot_count": "异地热点数量",
        # "dealer_suspicious_hotspot_count": "可疑热点数量",
        # "suspicious_hotspot_ratio": "热点可疑率",
        # "dealer_total_box_count": "总开箱数",
        # "cluster_label": "簇标签",
        # "province": "省",
        # "city": "市",
        # "district": "区",
        # "street": "镇街",
        # "scanning_count_within_cluster": "簇内点数量",
        # "is_remote": "是否异地",
        # "is_suspicious": "是否可疑",
        # "ratio_scanning_count": "扫码量占比",
        # "dis_to_overall_centroid": "热点质心到总质心的距离（质心距离）",
        # "dis_to_all_local_hotspots_centroid": "距本地热点总质心",
        # "dis_to_all_local_points_centroid": "距本地点总质心",
        # "std_distance_within_cluster": "簇内离散度",
        # "dis_border": "距范围边界",
        # "avg_distance_within_cluster": "热点内各点到该热点质心距离的平均值（热点内距离均值）",
        # "OPEN_PROVINCE": "开瓶省",
        # "OPEN_CITY": "开瓶市",
        # "count": "数量",
        # "is_remote_city": "是否为异地二级城市",
        # "box_count_within_cluster": "箱数",
        # "box_count_ratio_for_cluster": "箱数占比",
        # "BELONG_DEALER_NO": "经销商编码",
        # "BELONG_DEALER_NAME": "经销商名称",
        # "PRODUCT_GROUP_NAME": "品项名称",
        # "EFFECTIVE_DATE": "生效日期",
        # "INACTIVE_DATE": "失效日期",
        # "DEALER_CODE": "经销商编码",
        # "PRODUCT_GROUP_CODE": "品项编码",
        # "AREA_NAME": "经营区域",
        # "PROVINCE": "省",
        # "CITY": "市",
        # "DISTRICT": "区县",
        # "STREET": "镇街",
        # "dealer_hotspot_sparse_count": "一级热点数量",
        # "dealer_remote_hotspot_sparse_count": "异地一级热点数量",
        # "remote_hotspot_sparse_ratio": "一级热点异地率",
        # "dealer_suspicious_hotspot_sparse_count": "可疑一级热点数量",
        # "suspicious_hotspot_sparse_ratio": "一级热点可疑率",
        # "dealer_hotspot_dense_count": "二级热点数量",
        # "dealer_remote_hotspot_dense_count": "异地二级热点数量",
        # "dealer_suspicious_hotspot_dense_count": "可疑二级热点数量",
    }

    df_model_parameters = df_suspicious_hotspots_parameters.copy().rename(
        columns=rename_dict
    )
    df_suspicious_dealers = df_dealer_results.loc[
        df_dealer_results.is_dealer_suspicious == 1, :
    ]
    df_suspicious_dealers = df_suspicious_dealers.sort_values(
        by="BELONG_DEALER_NO"
    ).reset_index(drop=True)
    product_group_name = df_suspicious_dealers.loc[0, "PRODUCT_GROUP_NAME"]

    # print(
    #     f"-------------------基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下-------------------"
    # )
    display_forth_title(
        "-" * 30
        + f"基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下"
        + "-" * 30
    )
    print()

    # print("---模型参数---")
    # print(tabulate(df_model_parameters, headers='keys', tablefmt='pretty', showindex=False))
    display_fifth_title("模型参数")
    df_model_parameters_styled = (
        df_model_parameters.style.format(
            {
                "热点扫码量占比阈值": "{:.1f}",  # 浮动数值保留一位小数
                "热点离散度阈值": "{:.1f}",  # 浮动数值保留一位小数
            }
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},
            ]  # 内容居中
        )
        .hide(axis="index")
    )
    display(df_model_parameters_styled)
    print()

    # print("---经销商数量统计---")
    display_fifth_title("经销商数量统计")
    df_region_dealer_statistics = pd.DataFrame(
        {
            "扫码经销商总数": df_total_scanning_locations.BELONG_DEALER_NO.nunique(),
            "经营范围未归档经销商数量": df_total_scanning_locations.loc[
                df_total_scanning_locations["is_dealer_within_archive"] == 0, :
            ].BELONG_DEALER_NO.nunique(),
            "当前规则下可疑经销商数量": df_suspicious_dealers.shape[0],
        },
        index=[0],
    )
    df_region_dealer_statistics_styled = (
        df_region_dealer_statistics.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},  # 内容居中
            ]
        ).hide(axis="index")
    )
    display(df_region_dealer_statistics_styled)
    print()

    # print(f"扫码经销商总数： {df_total_scanning_locations.BELONG_DEALER_NO.nunique()}")
    # print(
    #     f"经营范围未归档经销商总数： {df_total_scanning_locations.loc[df_total_scanning_locations['is_dealer_within_archive'] == 0, :].BELONG_DEALER_NO.nunique()}"
    # )
    # print(f"当前规则下可疑经销商数量: {df_suspicious_dealers.shape[0]}")
    # print()

    # print("---可疑经销商汇总表---")
    display_fifth_title("可疑经销商汇总表")
    df_suspicious_dealers_to_show = df_suspicious_dealers.loc[
        :, suspicious_dealers_overall_cols
    ]
    df_suspicious_dealers_to_show = df_suspicious_dealers_to_show.rename(
        columns=rename_dict
    )
    # print(tabulate(df_suspicious_dealers_to_show , headers='keys', tablefmt='pretty', showindex=False))

    df_suspicious_dealers_to_show_styled = (
        df_suspicious_dealers_to_show.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},  # 内容居中
            ]
        ).hide(axis="index")
    )
    display(df_suspicious_dealers_to_show_styled)
    print()
    print("*" * 150)

    print()
    print("可疑经销商详细信息如下:")
    print("=" * 100)
    print()

    ids_suspicious = list(df_suspicious_dealers.BELONG_DEALER_NO)
    for id in ids_suspicious:
        df_dealer_dealer_results = df_dealer_results.loc[
            df_dealer_results.BELONG_DEALER_NO == id, :
        ].reset_index(drop=True)
        df_dealer_total_scanning_locations = df_total_scanning_locations.loc[
            df_total_scanning_locations.BELONG_DEALER_NO == id, :
        ].reset_index(drop=True)
        df_dealer_total_centroids = df_total_centroids.loc[
            df_total_centroids.dealer_id == id, :
        ].reset_index(drop=True)

        main_show_dealer_results(
            df_dealer_dealer_results,
            df_dealer_total_scanning_locations,
            df_dealer_total_centroids,
            dealer_scope_dict_path,
        )


def main_show_results_special(
    df_dealer_results,
    df_total_scanning_locations,
    df_total_centroids,
    df_suspicious_hotspots_parameters,
    df_dealer_results_dense,
    df_total_scanning_locations_dense,
    df_total_centroids_dense,
    df_suspicious_hotspots_parameters_dense,
    dealer_scope_dict_path,
    start_date_str,
    end_date_str,
    dealer_region_name,
):

    suspicious_dealers_overall_cols = [
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "PRODUCT_GROUP_NAME",
    ]

    rename_dict = {
        "radius": "簇半径",
        "min_samples": "簇内最少样本数",
        "dis_hotspots_c_t": "距本地热点总质心的距离阈值",
        "dis_points_c_t": "距本地扫码点总质心的距离阈值",
        "dis_border_t": "距边界最小距离阈值",
        "ratio_scanning_t": "热点扫码量占比阈值",
        "scanning_count_t": "热点扫码量阈值",
        "std_distance_t": "热点离散度阈值",
        "box_count_t": "紧密热点的箱数阈值",
        "BELONG_DEALER_NO": "经销商编码",
        "BELONG_DEALER_NAME": "经销商名称",
        "PRODUCT_GROUP_NAME": "品项名称",
        # "BELONG_DEALER_NO": "经销商编码",
        # "BELONG_DEALER_NAME": "经销商名称",
        # "PRODUCT_GROUP_NAME": "品项名称",
        # "remote_ratio": "开瓶异地率",
        # "total_scanning_count": "扫码总量",
        # "remote_scanning_count": "异地扫码量",
        # "remote_hotspot_ratio": "热点异地率",
        # "dealer_hotspot_count": "热点总量",
        # "dealer_remote_hotspot_count": "异地热点数量",
        # "suspicious_hotspot_ratio": "热点可疑率",
        # "dealer_suspicious_hotspot_count": "可疑热点数量",
        # "cluster_label": "簇标签",
        # "province": "省",
        # "city": "市",
        # "district": "区",
        # "street": "镇街",
        # "scanning_count_within_cluster": "簇内点数量",
        # "is_remote": "是否异地",
        # "is_suspicious": "是否可疑",
        # "ratio_scanning_count": "扫码量占比",
        # "dis_to_overall_centroid": "热点质心到总质心的距离（质心距离）",
        # "dis_to_all_local_hotspots_centroid": "距本地热点总质心",
        # "dis_to_all_local_points_centroid": "距本地点总质心",
        # "dis_border": "距范围边界",
        # "avg_distance_within_cluster": "热点内各点到该热点质心距离的平均值（热点内距离均值）",
        # "std_distance_within_cluster": "簇内离散度",
        # "OPEN_PROVINCE": "开瓶省",
        # "OPEN_CITY": "开瓶市",
        # "count": "数量",
        # "is_remote_city": "是否为异地二级城市",
        # "dealer_total_box_count": "总开箱数",
        # "box_count_within_cluster": "箱数",
        # "box_count_ratio_for_cluster": "箱数占比",
        # "BELONG_DEALER_NO": "经销商编码",
        # "BELONG_DEALER_NAME": "经销商名称",
        # "PRODUCT_GROUP_NAME": "品项名称",
        # "EFFECTIVE_DATE": "生效日期",
        # "INACTIVE_DATE": "失效日期",
        # "DEALER_CODE": "经销商编码",
        # "PRODUCT_GROUP_CODE": "品项编码",
        # "AREA_NAME": "经营区域",
        # "PROVINCE": "省",
        # "CITY": "市",
        # "DISTRICT": "区县",
        # "STREET": "镇街",
        # "dealer_hotspot_sparse_count": "一级热点数量",
        # "dealer_remote_hotspot_sparse_count": "异地一级热点数量",
        # "remote_hotspot_sparse_ratio": "一级热点异地率",
        # "dealer_suspicious_hotspot_sparse_count": "可疑一级热点数量",
        # "suspicious_hotspot_sparse_ratio": "一级热点可疑率",
        # "dealer_hotspot_dense_count": "二级热点数量",
        # "dealer_remote_hotspot_dense_count": "异地二级热点数量",
        # "dealer_suspicious_hotspot_dense_count": "可疑二级热点数量",
    }

    df_model_parameters = df_suspicious_hotspots_parameters.copy().rename(
        columns=rename_dict
    )
    df_model_parameters_dense = df_suspicious_hotspots_parameters_dense.copy().rename(
        columns=rename_dict
    )
    df_model_parameters_dense.rename(
        columns={
            "簇半径": "二级分簇半径",
        }
    )

    df_suspicious_dealers = (
        df_dealer_results.loc[df_dealer_results["is_dealer_suspicious"] == 1, :]
        .sort_values(by="BELONG_DEALER_NO")
        .reset_index(drop=True)
    )
    df_suspicious_dealers_dense = (
        df_dealer_results_dense.loc[
            df_dealer_results_dense["is_dealer_suspicious"] == 1, :
        ]
        .sort_values(by="BELONG_DEALER_NO")
        .reset_index(drop=True)
    )
    product_group_name = df_suspicious_dealers.loc[0, "PRODUCT_GROUP_NAME"]

    # print(
    #     f"-------------------基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下-------------------"
    # )
    display_forth_title(
        "-" * 30
        + f"基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下"
        + "-" * 30
    )
    print()

    # print("---一级分簇模型参数---")
    # print(tabulate(df_model_parameters, headers='keys', tablefmt='pretty', showindex=False))
    display_fifth_title("一级分簇模型参数")
    df_model_parameters_styled = (
        df_model_parameters.style.format(
            {
                "热点扫码量占比阈值": "{:.1f}",  # 浮动数值保留一位小数
                "簇内离散度阈值": "{:.1f}",  # 浮动数值保留一位小数
            }
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},
            ]  # 内容居中
        )
        .hide(axis="index")
    )
    display(df_model_parameters_styled)
    print()

    # print("---二级分簇模型参数---")
    # print(tabulate(df_model_parameters_dense, headers='keys', tablefmt='pretty', showindex=False))
    display_fifth_title("二级分簇模型参数")
    df_model_parameters_dense_styled = (
        df_model_parameters_dense.style.format(
            {
                "该热点扫码量占比阈值": "{:.1f}",  # 浮动数值保留一位小数
                "簇内离散度阈值": "{:.1f}",  # 浮动数值保留一位小数
            }
        )
        .set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},
            ]  # 内容居中
        )
        .hide(axis="index")
    )
    display(df_model_parameters_dense_styled)
    print()

    ids_suspicious_total = set(df_suspicious_dealers.BELONG_DEALER_NO) | set(
        df_suspicious_dealers_dense.BELONG_DEALER_NO
    )
    # print(f"扫码经销商总数： {df_total_scanning_locations.BELONG_DEALER_NO.nunique()}")
    # print(
    #     f"经营范围未归档经销商总数： {df_total_scanning_locations.loc[df_total_scanning_locations['is_dealer_within_archive'] == 0, :].BELONG_DEALER_NO.nunique()}"
    # )
    # print(f"当前规则下可疑经销商数量: {len(ids_suspicious_total)}")
    display_fifth_title("经销商数量统计")
    df_region_dealer_statistics = pd.DataFrame(
        {
            "扫码经销商总数": df_total_scanning_locations.BELONG_DEALER_NO.nunique(),
            "经营范围未归档经销商数量": df_total_scanning_locations.loc[
                df_total_scanning_locations["is_dealer_within_archive"] == 0, :
            ].BELONG_DEALER_NO.nunique(),
            "当前规则下可疑经销商数量": len(ids_suspicious_total),
        },
        index=[0],
    )
    df_region_dealer_statistics_styled = (
        df_region_dealer_statistics.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},  # 内容居中
            ]
        ).hide(axis="index")
    )
    display(df_region_dealer_statistics_styled)
    print()

    # print("---可疑经销商汇总表---")
    display_fifth_title("可疑经销商汇总表")
    df_suspicious_dealers_to_show = df_suspicious_dealers.loc[
        :, suspicious_dealers_overall_cols
    ]
    df_suspicious_dealers_to_show_dense = df_suspicious_dealers_dense.loc[
        :, suspicious_dealers_overall_cols
    ]
    df_suspicious_dealers_to_show_total = pd.concat(
        [df_suspicious_dealers_to_show, df_suspicious_dealers_to_show_dense], axis=0
    ).drop_duplicates(ignore_index=True)
    df_suspicious_dealers_to_show_total = (
        df_suspicious_dealers_to_show_total.sort_values(
            by="BELONG_DEALER_NO", ignore_index=True
        )
    )
    ids_suspicious_total = list(df_suspicious_dealers_to_show_total.BELONG_DEALER_NO)
    df_suspicious_dealers_to_show_total = df_suspicious_dealers_to_show_total.rename(
        columns=rename_dict
    )
    # print(tabulate(df_suspicious_dealers_to_show_total, headers='keys', tablefmt='pretty', showindex=False))

    df_suspicious_dealers_to_show_total_styled = (
        df_suspicious_dealers_to_show_total.style.set_table_styles(
            [
                {"selector": "th", "props": [("text-align", "center")]},  # 表头居中
                {"selector": "td", "props": [("text-align", "center")]},
            ]  # 内容居中
        ).hide(axis="index")
    )
    display(df_suspicious_dealers_to_show_total_styled)
    print("*" * 150)

    print()
    print()
    print()
    print("可疑经销商详细信息如下:")
    print("=" * 100)
    print()

    for id in ids_suspicious_total:
        df_dealer_dealer_results = df_dealer_results.loc[
            df_dealer_results.BELONG_DEALER_NO == id, :
        ].reset_index(drop=True)
        df_dealer_total_scanning_locations = df_total_scanning_locations.loc[
            df_total_scanning_locations.BELONG_DEALER_NO == id, :
        ].reset_index(drop=True)
        df_dealer_total_centroids = df_total_centroids.loc[
            df_total_centroids.dealer_id == id, :
        ].reset_index(drop=True)

        df_dealer_dealer_results_dense = df_dealer_results_dense.loc[
            df_dealer_results_dense.BELONG_DEALER_NO == id, :
        ].reset_index(drop=True)
        df_dealer_total_scanning_locations_dense = (
            df_total_scanning_locations_dense.loc[
                df_total_scanning_locations_dense.BELONG_DEALER_NO == id, :
            ].reset_index(drop=True)
        )
        df_dealer_total_centroids_dense = df_total_centroids_dense.loc[
            df_total_centroids_dense.dealer_id == id, :
        ].reset_index(drop=True)

        main_show_dealer_results_special(
            df_dealer_dealer_results,
            df_dealer_total_scanning_locations,
            df_dealer_total_centroids,
            df_dealer_dealer_results_dense,
            df_dealer_total_scanning_locations_dense,
            df_dealer_total_centroids_dense,
            dealer_scope_dict_path,
        )


def main_show_region_short_results(
    df_dealer_results,
    df_total_scanning_locations,
    start_date_str,
    end_date_str,
    dealer_region_name,
):

    df_suspicious_dealers = df_dealer_results.loc[
        df_dealer_results["is_dealer_suspicious"] == 1, :
    ]
    product_group_name = df_suspicious_dealers.loc[0, "PRODUCT_GROUP_NAME"]
    print(
        f"-------------------基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下-------------------"
    )

    print(f"扫码经销商总数： {df_total_scanning_locations.BELONG_DEALER_NO.nunique()}")
    print(
        f"经营范围未归档经销商总数： {df_total_scanning_locations.loc[df_total_scanning_locations['is_dealer_within_archive'] == 0, :].BELONG_DEALER_NO.nunique()}"
    )
    print(f"当前规则下可疑经销商数量: {df_suspicious_dealers.shape[0]}")
    print()

    suspicious_dealers_overall_cols = [
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "PRODUCT_GROUP_NAME",
    ]
    rename_dict = {
        "BELONG_DEALER_NO": "经销商编码",
        "BELONG_DEALER_NAME": "经销商名称",
        "PRODUCT_GROUP_NAME": "品项名称",
    }
    print("---可疑经销商汇总表---")
    df_suspicious_dealers_to_show = df_suspicious_dealers.loc[
        :, suspicious_dealers_overall_cols
    ]
    df_suspicious_dealers_to_show = df_suspicious_dealers_to_show.rename(
        columns=rename_dict
    )
    print(
        tabulate(
            df_suspicious_dealers_to_show,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print("*" * 200)
    print()

    print("---经营范围未归档经销商---")
    df_unarchive_dealers_to_show = df_total_scanning_locations.loc[
        df_total_scanning_locations["is_dealer_within_archive"] == 0,
        suspicious_dealers_overall_cols,
    ].drop_duplicates(ignore_index=True)
    df_unarchive_dealers_to_show = df_unarchive_dealers_to_show.rename(
        columns=rename_dict
    )
    print(
        tabulate(
            df_unarchive_dealers_to_show,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print("*" * 200)


def main_show_region_short_results_special(
    df_dealer_results,
    df_total_scanning_locations,
    df_dealer_results_dense,
    start_date_str,
    end_date_str,
    dealer_region_name,
):

    df_suspicious_dealers = df_dealer_results.loc[
        df_dealer_results["is_dealer_suspicious"] == 1, :
    ]
    df_suspicious_dealers_dense = df_dealer_results_dense.loc[
        df_dealer_results_dense["is_dealer_suspicious"] == 1, :
    ]
    ids_suspicious_total = set(df_suspicious_dealers.BELONG_DEALER_NO) | set(
        df_suspicious_dealers_dense.BELONG_DEALER_NO
    )

    product_group_name = df_suspicious_dealers.loc[0, "PRODUCT_GROUP_NAME"]

    print(
        f"-------------------基于模型v1.0, ({start_date_str} - {end_date_str}),  ({dealer_region_name} - {product_group_name})的结果如下-------------------"
    )
    print()

    print(f"扫码经销商总数： {df_total_scanning_locations.BELONG_DEALER_NO.nunique()}")
    print(
        f"经营范围未归档经销商总数： {df_total_scanning_locations.loc[df_total_scanning_locations['is_dealer_within_archive'] == 0, :].BELONG_DEALER_NO.nunique()}"
    )
    print(f"当前规则下可疑经销商数量: {len(ids_suspicious_total)}")
    print()

    suspicious_dealers_overall_cols = [
        "BELONG_DEALER_NO",
        "BELONG_DEALER_NAME",
        "PRODUCT_GROUP_NAME",
    ]
    rename_dict = {
        "BELONG_DEALER_NO": "经销商编码",
        "BELONG_DEALER_NAME": "经销商名称",
        "PRODUCT_GROUP_NAME": "品项名称",
    }
    print("---可疑经销商汇总表---")
    df_suspicious_dealers_to_show = df_suspicious_dealers.loc[
        :, suspicious_dealers_overall_cols
    ]
    df_suspicious_dealers_to_show_dense = df_suspicious_dealers_dense.loc[
        :, suspicious_dealers_overall_cols
    ]
    df_suspicious_dealers_to_show_total = pd.concat(
        [df_suspicious_dealers_to_show, df_suspicious_dealers_to_show_dense], axis=0
    ).drop_duplicates(ignore_index=True)
    df_suspicious_dealers_to_show_total = df_suspicious_dealers_to_show_total.rename(
        columns=rename_dict
    )
    print(
        tabulate(
            df_suspicious_dealers_to_show_total,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print()

    print("---经营范围未归档经销商---")
    df_unarchive_dealers_to_show = df_total_scanning_locations.loc[
        df_total_scanning_locations["is_dealer_within_archive"] == 0,
        suspicious_dealers_overall_cols,
    ].drop_duplicates(ignore_index=True)
    df_unarchive_dealers_to_show = df_unarchive_dealers_to_show.rename(
        columns=rename_dict
    )
    print(
        tabulate(
            df_unarchive_dealers_to_show,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print("*" * 200)


def main_show_dealer_results(
    df_dealer_dealer_results,
    df_dealer_total_scanning_locations,
    df_dealer_total_centroids,
    dealer_scope_dict_path,
):

    if df_dealer_total_scanning_locations.empty:
        print("无记录")
        return

    dealer_info_cols = [
        "is_dealer_suspicious",
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
        "dis_to_all_local_points_centroid",
        "std_distance_within_cluster",
        "box_count_within_cluster",
        "dis_border",
    ]

    rename_dict = {
        "is_dealer_suspicious": "是否可疑",
        "remote_ratio": "开瓶异地率",
        "total_scanning_count": "扫码总量",
        "remote_scanning_count": "异地扫码量",
        "remote_hotspot_ratio": "热点异地率",
        "dealer_hotspot_count": "热点总量",
        "dealer_remote_hotspot_count": "异地热点数量",
        "dealer_suspicious_hotspot_count": "可疑热点数量",
        "suspicious_hotspot_ratio": "热点可疑率",
        "dealer_total_box_count": "总开箱数",
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
        "dis_to_all_local_hotspots_centroid": "距本地热点总质心",
        "dis_to_all_local_points_centroid": "距本地点总质心",
        "std_distance_within_cluster": "簇内离散度",
        "dis_border": "距范围边界",
        "avg_distance_within_cluster": "热点内各点到该热点质心距离的平均值（热点内距离均值）",
        "OPEN_PROVINCE": "开瓶省",
        "OPEN_CITY": "开瓶市",
        "count": "数量",
        "is_remote_city": "是否为异地二级城市",
        "box_count_within_cluster": "箱数",
        "box_count_ratio_for_cluster": "箱数占比",
        "BELONG_DEALER_NO": "经销商编码",
        "BELONG_DEALER_NAME": "经销商名称",
        "PRODUCT_GROUP_NAME": "品项名称",
        "EFFECTIVE_DATE": "生效日期",
        "INACTIVE_DATE": "失效日期",
        "DEALER_CODE": "经销商编码",
        "PRODUCT_GROUP_CODE": "品项编码",
        "AREA_NAME": "经营区域",
        "PROVINCE": "省",
        "CITY": "市",
        "DISTRICT": "区县",
        "STREET": "镇街",
        "dealer_hotspot_sparse_count": "一级热点数量",
        "dealer_remote_hotspot_sparse_count": "异地一级热点数量",
        "remote_hotspot_sparse_ratio": "一级热点异地率",
        "dealer_suspicious_hotspot_sparse_count": "可疑一级热点数量",
        "suspicious_hotspot_sparse_ratio": "一级热点可疑率",
        "dealer_hotspot_dense_count": "二级热点数量",
        "dealer_remote_hotspot_dense_count": "异地二级热点数量",
        "dealer_suspicious_hotspot_dense_count": "可疑二级热点数量",
    }

    dealer_id = df_dealer_total_scanning_locations.loc[0, "BELONG_DEALER_NO"]
    dealer_name = df_dealer_total_scanning_locations.loc[0, "BELONG_DEALER_NAME"]
    product_group_id = df_dealer_total_scanning_locations.loc[0, "PRODUCT_GROUP_CODE"]
    product_group_name = df_dealer_total_scanning_locations.loc[0, "PRODUCT_GROUP_NAME"]

    print(f"经销商: {dealer_id} - {dealer_name} - {product_group_name}")
    print("-" * 60)
    print()

    with open(dealer_scope_dict_path, "rb") as f:
        dealer_scope_dict = pickle.load(f)

    df_business_scope = dealer_scope_dict.get((dealer_id, product_group_id), None)
    if (dealer_id, product_group_id) not in dealer_scope_dict:
        print(
            f"{dealer_id} - {dealer_name} - {product_group_name}的经营范围并未记录在档! "
        )
        return

    df_business_scope = dealer_scope_dict[(dealer_id, product_group_id)]
    df_business_scope_to_show = df_business_scope.copy()
    df_business_scope_to_show["EFFECTIVE_DATE"] = df_business_scope_to_show[
        "EFFECTIVE_DATE"
    ].dt.strftime("%Y-%m-%d")
    df_business_scope_to_show["INACTIVE_DATE"] = df_business_scope_to_show[
        "INACTIVE_DATE"
    ].dt.strftime("%Y-%m-%d")
    df_business_scope_to_show = df_business_scope_to_show.drop(
        columns="AREA_CODE"
    ).rename(columns=rename_dict)
    print(f"---经营范围---")
    print(
        tabulate(
            df_business_scope_to_show,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print()

    df_dealer_info_to_show = df_dealer_dealer_results.loc[:, dealer_info_cols]
    df_dealer_info_to_show["is_dealer_suspicious"] = df_dealer_info_to_show[
        "is_dealer_suspicious"
    ].map({1: "是", 0: "否"})
    df_dealer_info_to_show = df_dealer_info_to_show.rename(columns=rename_dict)
    print(f"---范围内经销商异地信息---")
    print(
        tabulate(
            df_dealer_info_to_show, headers="keys", tablefmt="pretty", showindex=False
        )
    )
    print()

    df_dealer_cluster = df_dealer_total_centroids.loc[
        df_dealer_total_centroids.dealer_id == dealer_id, dealer_cluster_cols
    ]
    df_dealer_cluster = df_dealer_cluster.loc[
        ~(df_dealer_cluster["cluster_label"].isin([-2, -1])), :
    ]
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
    print(f"---经销商热点信息---")
    print("可疑热点标签:")
    print(suspicious_labels)
    df_dealer_cluster = df_dealer_cluster.rename(columns=rename_dict)
    print(
        tabulate(df_dealer_cluster, headers="keys", tablefmt="pretty", showindex=False)
    )
    print()

    df_city_count = (
        df_dealer_total_scanning_locations[["OPEN_PROVINCE", "OPEN_CITY"]]
        .value_counts()
        .reset_index()
    )
    # 扫码表格式 -> 经营范围表格式
    # ['天津市', '天津市'] -> ['天津', '天津市']
    df_city_count["OPEN_PROVINCE"] = df_city_count["OPEN_PROVINCE"].apply(
        lambda x: x[:2] if x in ["天津市", "北京市", "上海市", "重庆市"] else x
    )

    df_valid_scope = df_dealer_dealer_results.loc[0, "dealer_valid_scope"]
    df_count_merged = pd.DataFrame()

    if df_valid_scope.empty:
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
    print(f"开瓶二级城市数： {len(df_city_count)}")
    print(f"---可疑经销商开瓶城市统计表(大于五瓶的城市）---")
    print(
        tabulate(
            df_count_merged.loc[df_count_merged["数量"] > 5, :],
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print()

    polyline_points_list_total = df_dealer_dealer_results.loc[
        0, "dealer_polyline_points_list_total"
    ]
    # acodes = df_dealer_dealer_results.loc[0, 'dealer_acodes']
    m = plot_clusters_with_folium(
        df_dealer_total_scanning_locations,
        points_size=3,
        noise_size=1,
        polyline_points_list=polyline_points_list_total,
    )
    display(m)
    print()


def main_show_dealer_results_special(
    df_dealer_dealer_results,
    df_dealer_total_scanning_locations,
    df_dealer_total_centroids,
    df_dealer_dealer_results_dense,
    df_dealer_total_scanning_locations_dense,
    df_dealer_total_centroids_dense,
    dealer_scope_dict_path,
):

    if df_dealer_total_scanning_locations.empty:
        print("无记录")
        return

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
        "dis_to_all_local_points_centroid",
        "std_distance_within_cluster",
        "box_count_within_cluster",
        "dis_border",
    ]

    rename_dict = {
        "remote_ratio": "开瓶异地率",
        "total_scanning_count": "扫码总量",
        "remote_scanning_count": "异地扫码量",
        "remote_hotspot_ratio": "热点异地率",
        "dealer_hotspot_count": "热点总量",
        "dealer_remote_hotspot_count": "异地热点数量",
        "dealer_suspicious_hotspot_count": "可疑热点数量",
        "suspicious_hotspot_ratio": "热点可疑率",
        "dealer_total_box_count": "总开箱数",
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
        "dis_to_all_local_hotspots_centroid": "距本地热点总质心",
        "dis_to_all_local_points_centroid": "距本地点总质心",
        "std_distance_within_cluster": "簇内离散度",
        "dis_border": "距范围边界",
        "avg_distance_within_cluster": "热点内各点到该热点质心距离的平均值（热点内距离均值）",
        "OPEN_PROVINCE": "开瓶省",
        "OPEN_CITY": "开瓶市",
        "count": "数量",
        "is_remote_city": "是否为异地二级城市",
        "box_count_within_cluster": "箱数",
        "box_count_ratio_for_cluster": "箱数占比",
        "BELONG_DEALER_NO": "经销商编码",
        "BELONG_DEALER_NAME": "经销商名称",
        "PRODUCT_GROUP_NAME": "品项名称",
        "EFFECTIVE_DATE": "生效日期",
        "INACTIVE_DATE": "失效日期",
        "DEALER_CODE": "经销商编码",
        "PRODUCT_GROUP_CODE": "品项编码",
        "AREA_NAME": "经营区域",
        "PROVINCE": "省",
        "CITY": "市",
        "DISTRICT": "区县",
        "STREET": "镇街",
        "dealer_hotspot_sparse_count": "一级热点数量",
        "dealer_remote_hotspot_sparse_count": "异地一级热点数量",
        "remote_hotspot_sparse_ratio": "一级热点异地率",
        "dealer_suspicious_hotspot_sparse_count": "可疑一级热点数量",
        "suspicious_hotspot_sparse_ratio": "一级热点可疑率",
        "dealer_hotspot_dense_count": "二级热点数量",
        "dealer_remote_hotspot_dense_count": "异地二级热点数量",
        "dealer_suspicious_hotspot_dense_count": "可疑二级热点数量",
        "is_final_suspicious_dealer": "是否可疑(综合)",
    }

    dealer_id = df_dealer_total_scanning_locations.loc[0, "BELONG_DEALER_NO"]
    dealer_name = df_dealer_total_scanning_locations.loc[0, "BELONG_DEALER_NAME"]
    product_group_id = df_dealer_total_scanning_locations.loc[0, "PRODUCT_GROUP_CODE"]
    product_group_name = df_dealer_total_scanning_locations.loc[0, "PRODUCT_GROUP_NAME"]

    df_dealer_hotspots = df_dealer_total_centroids.loc[
        ~(df_dealer_total_centroids["cluster_label"].isin([-2, -1])), :
    ]
    df_dealer_hotspots_dense = df_dealer_total_centroids_dense.loc[
        ~(df_dealer_total_centroids_dense["cluster_label"].isin([-2, -1])), :
    ]

    print(f"经销商: {dealer_id} - {dealer_name} - {product_group_name}")
    print("-" * 60)
    print()

    with open(dealer_scope_dict_path, "rb") as f:
        dealer_scope_dict = pickle.load(f)

    df_business_scope = dealer_scope_dict.get((dealer_id, product_group_id), None)
    if (dealer_id, product_group_id) not in dealer_scope_dict:
        print(
            f"{dealer_id} - {dealer_name} - {product_group_name}的经营范围并未记录在档! "
        )
        return

    df_business_scope_to_show = (
        df_business_scope.copy()
    )  # 不知道为什么会有bug 会更改到字典里的数据类型,因此采用深拷贝
    df_business_scope_to_show["EFFECTIVE_DATE"] = df_business_scope_to_show[
        "EFFECTIVE_DATE"
    ].dt.strftime("%Y-%m-%d")
    df_business_scope_to_show["INACTIVE_DATE"] = df_business_scope_to_show[
        "INACTIVE_DATE"
    ].dt.strftime("%Y-%m-%d")
    df_business_scope_to_show = df_business_scope_to_show.drop(
        columns="AREA_CODE"
    ).rename(columns=rename_dict)
    print(f"---经营范围---")
    print(
        tabulate(
            df_business_scope_to_show,
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print()

    # df_dealer_info = pd.DataFrame(columns=['remote_ratio','total_scanning_count', 'remote_scanning_count', 'dealer_total_box_count'])
    df_dealer_info_sparse = pd.DataFrame(
        columns=[
            "dealer_hotspot_sparse_count",
            "dealer_remote_hotspot_sparse_count",
            "remote_hotspot_sparse_ratio",
            "dealer_suspicious_hotspot_sparse_count",
            "suspicious_hotspot_sparse_ratio",
        ]
    )
    df_dealer_info_dense = pd.DataFrame(
        columns=[
            "dealer_hotspot_dense_count",
            "dealer_remote_hotspot_dense_count",
            "dealer_suspicious_hotspot_dense_count",
        ]
    )

    # 大部分可疑都包含在了sparse的可疑名单里，如果不在 再从总数据里取
    # if not df_dealer_results.empty:
    #     suspicious_dealer_row = df_dealer_results.iloc[0]
    #     df_dealer_info = suspicious_dealer_row[['remote_ratio','total_scanning_count', 'remote_scanning_count', 'dealer_total_box_count']].to_frame().T.rename(columns=rename_dict)
    # else:
    #     # suspicious_dealer_dense_row = df_suspicious_dealer_dense_row.iloc[0]
    #     # df_dealer_info = suspicious_dealer_dense_row[['remote_ratio','total_scanning_count', 'remote_scanning_count', 'dealer_total_box_count']].to_frame().T.rename(columns=rename_dict)

    #     df_dealer_info.at[0, 'total_scanning_count'] = df_dealer_total_scanning_locations['BARCODE_BOTTLE'].nunique()
    #     df_dealer_info.at[0, 'remote_scanning_count'] = df_dealer_total_scanning_locations['point_remote_label_new'].nunique()
    #     df_dealer_info.at[0, 'remote_ratio'] = round((df_dealer_info.loc[0, 'remote_scanning_count'] / df_dealer_info.loc[0, 'total_scanning_count']), 2)
    #     df_dealer_info.at[0, 'dealer_total_box_count'] = df_dealer_total_scanning_locations['BARCODE_CORNER'].nunique()

    #     df_dealer_info = df_dealer_info.rename(columns=rename_dict)

    df_dealer_info = df_dealer_dealer_results[
        [
            "remote_ratio",
            "total_scanning_count",
            "remote_scanning_count",
            "dealer_total_box_count",
        ]
    ].copy()
    is_suspicious_dealer = df_dealer_dealer_results.loc[0, "is_dealer_suspicious"]

    df_dealer_info_sparse.at[0, "dealer_hotspot_sparse_count"] = df_dealer_hotspots[
        "cluster_label"
    ].nunique()
    df_dealer_info_sparse.at[0, "dealer_remote_hotspot_sparse_count"] = (
        df_dealer_hotspots["is_remote"].sum()
    )
    df_dealer_info_sparse.at[0, "remote_hotspot_sparse_ratio"] = round(
        df_dealer_hotspots["is_remote"].mean(), 2
    )
    df_dealer_info_sparse.at[0, "dealer_suspicious_hotspot_sparse_count"] = (
        df_dealer_hotspots["is_suspicious"].sum()
    )
    df_dealer_info_sparse.at[0, "suspicious_hotspot_sparse_ratio"] = round(
        df_dealer_hotspots["is_suspicious"].mean(), 2
    )
    df_dealer_info_sparse = df_dealer_info_sparse.rename(columns=rename_dict)

    dense_empty = df_dealer_dealer_results_dense.empty

    if not dense_empty:
        is_suspicious_dealer_dense = df_dealer_dealer_results_dense.loc[
            0, "is_dealer_suspicious"
        ]

        if is_suspicious_dealer or is_suspicious_dealer_dense:
            df_dealer_info["is_final_suspicious_dealer"] = 1
        else:
            df_dealer_info["is_final_suspicious_dealer"] = 0

        x = df_dealer_info.pop("is_final_suspicious_dealer")
        df_dealer_info.insert(0, "is_final_suspicious_dealer", x)
        df_dealer_info["is_final_suspicious_dealer"] = df_dealer_info[
            "is_final_suspicious_dealer"
        ].map({1: "是", 0: "否"})
        df_dealer_info = df_dealer_info.rename(columns=rename_dict)

        df_dealer_info_dense.at[0, "dealer_hotspot_dense_count"] = (
            df_dealer_hotspots_dense["cluster_label"].nunique()
        )
        df_dealer_info_dense.at[0, "dealer_remote_hotspot_dense_count"] = (
            df_dealer_hotspots_dense["is_remote"].sum()
        )
        df_dealer_info_dense.at[0, "dealer_suspicious_hotspot_dense_count"] = (
            df_dealer_hotspots_dense["is_suspicious"].sum()
        )
        df_dealer_info_dense = df_dealer_info_dense.rename(columns=rename_dict)
        print(f"---范围内经销商异地信息---")
        print(
            tabulate(df_dealer_info, headers="keys", tablefmt="pretty", showindex=False)
        )
        print("---一级热点信息---")
        print(
            tabulate(
                df_dealer_info_sparse,
                headers="keys",
                tablefmt="pretty",
                showindex=False,
            )
        )
        print("---二级级热点信息---")
        print(
            tabulate(
                df_dealer_info_dense, headers="keys", tablefmt="pretty", showindex=False
            )
        )
        print()
    else:
        df_dealer_info["is_final_suspicious_dealer"] = is_suspicious_dealer
        x = df_dealer_info.pop("is_final_suspicious_dealer")
        df_dealer_info.insert(0, "is_final_suspicious_dealer", x)
        df_dealer_info["is_final_suspicious_dealer"] = df_dealer_info[
            "is_final_suspicious_dealer"
        ].map({1: "是", 0: "否"})
        print(f"---范围内经销商异地信息---")
        print(
            tabulate(df_dealer_info, headers="keys", tablefmt="pretty", showindex=False)
        )
        print("---一级热点信息---")
        print(
            tabulate(
                df_dealer_info_sparse,
                headers="keys",
                tablefmt="pretty",
                showindex=False,
            )
        )
        print("---无二级热点---")

    # 打印开瓶二级城市统计
    df_city_count = (
        df_dealer_total_scanning_locations[["OPEN_PROVINCE", "OPEN_CITY"]]
        .value_counts()
        .reset_index()
    )
    # 扫码表格式 -> 经营范围表格式
    # ['天津市', '天津市'] -> ['天津', '天津市']
    df_city_count["OPEN_PROVINCE"] = df_city_count["OPEN_PROVINCE"].apply(
        lambda x: x[:2] if x in ["天津市", "北京市", "上海市", "重庆市"] else x
    )

    df_valid_scope = df_dealer_dealer_results.loc[0, "dealer_valid_scope"]
    df_count_merged = pd.DataFrame()

    if df_valid_scope.empty:
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
    print(f"开瓶二级城市数： {len(df_city_count)}")
    print(f"---可疑经销商开瓶城市统计表(大于五瓶的城市）---")
    print(
        tabulate(
            df_count_merged.loc[df_count_merged["数量"] > 5, :],
            headers="keys",
            tablefmt="pretty",
            showindex=False,
        )
    )
    print()

    polyline_points_list_total = df_dealer_dealer_results.loc[
        0, "dealer_polyline_points_list_total"
    ]

    df_dealer_cluster = df_dealer_hotspots.loc[:, dealer_cluster_cols]
    print(f"---经销商热点信息---")
    print()
    print("可疑一级热点标签:")
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
        tabulate(df_dealer_cluster, headers="keys", tablefmt="pretty", showindex=False)
    )
    print()
    print("---一级热点地图---")
    m = plot_clusters_with_folium(
        df_dealer_total_scanning_locations,
        points_size=3,
        noise_size=1,
        polyline_points_list=polyline_points_list_total,
    )
    display(m)

    df_dealer_cluster_dense = df_dealer_hotspots_dense.loc[:, dealer_cluster_cols]
    print("可疑二级热点标签:")
    suspicious_labels_dense = list(
        map(
            int,
            list(
                df_dealer_cluster_dense.loc[
                    df_dealer_cluster_dense["is_suspicious"] == 1, "cluster_label"
                ].values
            ),
        )
    )
    print(suspicious_labels_dense)
    df_dealer_cluster_dense = df_dealer_cluster_dense.rename(columns=rename_dict)
    print(
        tabulate(
            df_dealer_cluster_dense, headers="keys", tablefmt="pretty", showindex=False
        )
    )
    print()
    print("---二级热点地图---")
    m2 = plot_clusters_with_folium(
        df_dealer_total_scanning_locations_dense,
        points_size=3,
        noise_size=1,
        polyline_points_list=polyline_points_list_total,
    )
    display(m2)
    print()
