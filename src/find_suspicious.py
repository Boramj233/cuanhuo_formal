import pandas as pd
import numpy as np


def filter_suspicious_hotspot(df_total_centroids, radius, min_samples, **kwargs):

    hotspot_mask = ~(df_total_centroids["cluster_label"].isin([-2, -1]))

    local_hotspots_std = df_total_centroids.loc[
        hotspot_mask & (df_total_centroids["is_remote"] == 0),
        "std_distance_within_cluster",
    ]

    default_thresholds = {
        "dis_hotspots_c_t": 30,
        "dis_points_c_t": 100,
        "dis_border_t": 10,
        "ratio_scanning_t": 0.1,
        "scanning_count_t": 12,
        "std_quantile_t": 0.1,
        "box_count_t": 4,
    }

    thresholds = {**default_thresholds, **kwargs}

    dis_hotspots_c_t = thresholds["dis_hotspots_c_t"]
    dis_points_c_t = thresholds["dis_points_c_t"]
    dis_border_t = thresholds["dis_border_t"]
    ratio_scanning_t = thresholds["ratio_scanning_t"]
    scanning_count_t = thresholds["scanning_count_t"]

    if local_hotspots_std.size > 0:
        std_distance_within_cluster_threshold = np.quantile(
            local_hotspots_std, thresholds["std_quantile_t"]
        )
    else:
        # Handle the case when the array is empty, e.g., set a default threshold value or raise a warning
        std_distance_within_cluster_threshold = 0.05  # or other default value

    if std_distance_within_cluster_threshold < 0.05:
        std_distance_within_cluster_threshold = 0.05
    box_count_t = thresholds["box_count_t"]

    ##################################################################################################################################################
    # 设置可以热点条件

    suspicious_mask = (
        (hotspot_mask)
        & (df_total_centroids["is_remote"] == 1)
        & (
            (
                df_total_centroids["dis_to_all_local_hotspots_centroid"]
                >= dis_hotspots_c_t
            )
            | (np.isnan(df_total_centroids["dis_to_all_local_hotspots_centroid"]))
        )
        & (
            (df_total_centroids["dis_to_all_local_points_centroid"] >= dis_points_c_t)
            | (np.isnan(df_total_centroids["dis_to_all_local_points_centroid"]))
        )
        & (df_total_centroids["dis_border"] >= dis_border_t)
        & (
            (df_total_centroids["scanning_ratio_for_cluster"] >= ratio_scanning_t)
            | (df_total_centroids["scanning_count_within_cluster"] >= scanning_count_t)
        )
        & (
            (
                df_total_centroids["std_distance_within_cluster"]
                >= std_distance_within_cluster_threshold
            )
            | ((df_total_centroids["box_count_within_cluster"] >= box_count_t))
        )
    )

    ###################################################################################################################################################

    df_total_centroids_with_suspicious_label = df_total_centroids.copy()
    df_total_centroids_with_suspicious_label.loc[suspicious_mask, "is_suspicious"] = 1
    df_total_centroids_with_suspicious_label.loc[~suspicious_mask, "is_suspicious"] = 0
    df_total_centroids_with_suspicious_label.loc[~hotspot_mask, "is_suspicious"] = (
        df_total_centroids_with_suspicious_label.loc[~hotspot_mask, "cluster_label"]
    )  # 这个考虑更改下
    # df_total_centroids_with_suspicious_label['is_suspicious'] = df_total_centroids_with_suspicious_label['is_suspicious'].fillna(0).astype(int)

    df_suspicious_hotspots = (
        df_total_centroids_with_suspicious_label.loc[
            df_total_centroids_with_suspicious_label.is_suspicious == 1, :
        ]
        .drop(columns=["is_suspicious"])
        .reset_index(drop=True)
    )

    df_total_centroids_with_suspicious_label["is_suspicious"] = (
        df_total_centroids_with_suspicious_label["is_suspicious"].astype(int)
    )

    df_suspicious_hotspots_parameters = pd.DataFrame(
        {
            "radius": radius,
            "min_samples": min_samples,
            "dis_hotspots_c_t": dis_hotspots_c_t,
            "dis_points_c_t": dis_points_c_t,
            "dis_border_t": dis_border_t,
            "ratio_scanning_t": ratio_scanning_t,
            "scanning_count_t": scanning_count_t,
            "std_distance_t": round(std_distance_within_cluster_threshold, 2),
            "box_count_t": box_count_t,
        },
        index=[0],
    )

    return df_total_centroids_with_suspicious_label, df_suspicious_hotspots_parameters


def add_dealer_features_to_df_locations(
    df_total_scanning_locations, df_total_centroids
):
    df_total_scanning_locations_labels = df_total_scanning_locations.copy()
    df_total_scanning_locations_labels = df_total_scanning_locations_labels.merge(
        df_total_centroids[["dealer_id", "cluster_label", "is_suspicious"]],
        how="left",
        left_on=["BELONG_DEALER_NO", "cluster_label"],
        right_on=["dealer_id", "cluster_label"],
    )
    df_total_scanning_locations_labels = df_total_scanning_locations_labels.drop(
        columns=["dealer_id"]
    )
    df_total_scanning_locations_labels = df_total_scanning_locations_labels.rename(
        columns={"is_suspicious": "is_within_suspicious_hotspots"}
    )

    df_total_scanning_locations_labels["dealer_remote_ratio"] = (
        df_total_scanning_locations_labels.groupby("BELONG_DEALER_NO")[
            "is_remote_point_new"
        ].transform("mean")
    )
    df_total_scanning_locations_labels["dealer_total_scanning_count"] = (
        df_total_scanning_locations_labels.groupby("BELONG_DEALER_NO")[
            "BARCODE_BOTTLE"
        ].transform("nunique")
    )

    df_total_scanning_locations_labels["dealer_remote_scanning_count"] = (
        df_total_scanning_locations_labels.groupby("BELONG_DEALER_NO")[
            "is_remote_point_new"
        ].transform("sum")
    )

    # df_total_scanning_locations_labels['is_within_suspicious_hotspots'] = df_total_scanning_locations_labels['is_within_suspicious_hotspots'].apply(
    #     lambda x: 0 if x==-1 else x
    # )
    df_total_scanning_locations_labels.loc[
        df_total_scanning_locations_labels["is_within_suspicious_hotspots"] == -1,
        "is_within_suspicious_hotspots",
    ] = 0
    df_total_scanning_locations_labels["dealer_suspicious_points_ratio"] = (
        df_total_scanning_locations_labels.groupby("BELONG_DEALER_NO")[
            "is_within_suspicious_hotspots"
        ].transform("mean")
    )
    # df_total_scanning_locations_labels['dealer_suspicious_points_ratio'] = df_total_scanning_locations_labels['dealer_suspicious_points_ratio'].round(2)
    df_total_scanning_locations_labels["dealer_suspicious_points_count"] = (
        df_total_scanning_locations_labels.groupby("BELONG_DEALER_NO")[
            "is_within_suspicious_hotspots"
        ].transform("sum")
    )

    return df_total_scanning_locations_labels


def generate_df_dealer_results(df_total_scanning_locations, df_total_centroids):

    df_hotspots = df_total_centroids.loc[
        ~(df_total_centroids["cluster_label"].isin([-2, -1])), :
    ]
    df_hotspots = df_hotspots.drop_duplicates(
        subset=["dealer_id", "cluster_label"]
    )  # 可以不加drop_duplicates

    df_suspicious_hotspots = df_total_centroids.loc[
        df_total_centroids.is_suspicious == 1, :
    ].reset_index(drop=True)

    # df_total_centroids
    df_total_centroids_to_merge = df_total_centroids.loc[
        :,
        [
            "dealer_id",
            "dealer_total_box_count",
            "dealer_valid_scope",
            "dealer_polyline_points_list_total",
            "dealer_adcodes",
            "is_dealer_no_valid_scope",
        ],
    ].drop_duplicates(subset=["dealer_id"])
    df_total_centroids_to_merge = df_total_centroids_to_merge.rename(
        columns={"dealer_id": "BELONG_DEALER_NO"}
    )

    # df_hotspots
    df_hotspots["dealer_hotspot_count"] = df_hotspots.groupby("dealer_id")[
        "cluster_label"
    ].transform("count")
    df_hotspots["dealer_remote_hotspot_count"] = df_hotspots.groupby("dealer_id")[
        ["is_remote"]
    ].transform("sum")
    df_hotspots["dealer_remote_hotspot_ratio"] = df_hotspots.groupby("dealer_id")[
        ["is_remote"]
    ].transform("mean")
    df_hotspots_to_merge = df_hotspots.loc[
        :,
        [
            "dealer_id",
            "dealer_hotspot_count",
            "dealer_remote_hotspot_count",
            "dealer_remote_hotspot_ratio",
        ],
    ].drop_duplicates()
    df_hotspots_to_merge = df_hotspots_to_merge.rename(
        columns={"dealer_id": "BELONG_DEALER_NO"}
    )

    # df_suspicious_hotspots
    df_suspicious_hotspots["dealer_suspicious_hotspot_count"] = (
        df_suspicious_hotspots.groupby("dealer_id")["cluster_label"].transform("count")
    )
    df_suspicious_hotspots_to_merge = df_suspicious_hotspots.loc[
        :, ["dealer_id", "dealer_suspicious_hotspot_count"]
    ].drop_duplicates()
    df_suspicious_hotspots_to_merge = df_suspicious_hotspots_to_merge.rename(
        columns={"dealer_id": "BELONG_DEALER_NO"}
    )

    # sir this way
    # df_total_scanning_locations
    df_total_scanning_locations_to_merge = df_total_scanning_locations.loc[
        :,
        [
            "BELONG_DEALER_NO",
            "BELONG_DEALER_NAME",
            "PRODUCT_GROUP_CODE",
            "PRODUCT_GROUP_NAME",
            "dealer_total_scanning_count",
            "dealer_remote_scanning_count",
            "dealer_remote_ratio",
            "is_dealer_within_archive",
            "dealer_suspicious_points_ratio",
            "dealer_suspicious_points_count",
        ],
    ].drop_duplicates()

    df_result = pd.merge(
        df_total_scanning_locations_to_merge,
        df_total_centroids_to_merge,
        on="BELONG_DEALER_NO",
        how="left",
    )
    df_result = pd.merge(
        df_result, df_hotspots_to_merge, on="BELONG_DEALER_NO", how="left"
    )
    df_result = pd.merge(
        df_result, df_suspicious_hotspots_to_merge, on="BELONG_DEALER_NO", how="left"
    )

    # 将所有缺失值填充为 0
    df_result["dealer_suspicious_hotspot_ratio"] = (
        df_result["dealer_suspicious_hotspot_count"] / df_result["dealer_hotspot_count"]
    )

    df_result.fillna(0, inplace=True)

    df_result[
        [
            "dealer_hotspot_count",
            "dealer_remote_hotspot_count",
            "dealer_suspicious_hotspot_count",
            "dealer_total_box_count",
            "dealer_suspicious_points_count",
        ]
    ] = df_result[
        [
            "dealer_hotspot_count",
            "dealer_remote_hotspot_count",
            "dealer_suspicious_hotspot_count",
            "dealer_total_box_count",
            "dealer_suspicious_points_count",
        ]
    ].astype(
        int
    )

    df_result = df_result.reset_index(drop=True)
    return df_result


def add_suspicious_dealer_label(df_dealer_results):

    df_dealer_results_with_suspicious_label = df_dealer_results.copy()

    df_dealer_results_with_suspicious_label["is_dealer_suspicious"] = 0
    dealer_suspicious_mask = (
        df_dealer_results_with_suspicious_label["is_dealer_within_archive"] == 1
    ) & (df_dealer_results_with_suspicious_label["dealer_suspicious_hotspot_count"] > 0)

    df_dealer_results_with_suspicious_label.loc[
        dealer_suspicious_mask, "is_dealer_suspicious"
    ] = 1

    return df_dealer_results_with_suspicious_label


def find_suspicious_main(
    df_total_scanning_locations,
    df_total_centroids,
    radius,
    min_samples,
    **thresholds,
):
    df_total_centroids, df_suspicious_hotspots_parameters = filter_suspicious_hotspot(
        df_total_centroids, radius, min_samples, **thresholds
    )
    df_total_scanning_locations = add_dealer_features_to_df_locations(
        df_total_scanning_locations, df_total_centroids
    )
    df_dealer_results = generate_df_dealer_results(
        df_total_scanning_locations, df_total_centroids
    )
    df_dealer_results = add_suspicious_dealer_label(df_dealer_results)

    return (
        df_dealer_results,
        df_total_scanning_locations,
        df_total_centroids,
        df_suspicious_hotspots_parameters,
    )
