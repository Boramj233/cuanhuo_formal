from .data_clean import generate_clean_region_data_main
from .find_hotspots import find_hotspots_main, find_hotspots_continue_for_dense_main
from .find_suspicious import find_suspicious_main
from .functions import (
    get_month_start_end,
    extract_paths,
    read_outputs,
    save_model_outputs,
    load_model_parameters_config,
)
from .show_results import (
    show_results_main,
    show_results_special_main,
    show_region_short_results_main,
    show_region_short_results_special_main,
    show_dealer_results_main,
    show_dealer_results_special_main,
)
from .data_preprocessing import data_preprocessing_main

import os


def main_data_preprocessing(year_month_str, workspace_folder_path="./"):
    data_preprocessing_main(year_month_str, workspace_folder_path)


### IMPORTANT ###
def model_run(
    df_total_path,
    dealer_scope_dict_path,
    config_file_path,
    output_path,
    dealer_region_name,
    product_group_id,
    year_month_str,
    dbscan_parameters_tuple,
    save_outputs=True,
    show_results=True,
    save_results=False,
    thresholds={},
):
    """
    **thresholds 可选传入参数包括以下, 不传参时默认值为如下：
        - 'dis_hotspots_c_t': 35,
        - 'dis_points_c_t': 0,
        - 'dis_border' : 10
        - 'ratio_scanning_t': 0.1,
        - 'scanning_count_t': 12,
        - 'std_quantile_t': 0.1,
        - 'box_count_t': 4,

    """
    df_cleaned = generate_clean_region_data_main(
        dealer_region_name, df_total_path, product_group_id
    )
    print("generate clean data completed.")
    start_date_str, end_date_str = get_month_start_end(year_month_str)
    radius, min_samples = dbscan_parameters_tuple
    df_total_centroids, df_total_scanning_locations = find_hotspots_main(
        df_cleaned,
        product_group_id,
        start_date_str,
        end_date_str,
        radius,
        min_samples,
        config_file_path,
        dealer_scope_dict_path,
    )
    print(
        f"噪声点比： \
          {round((len(df_total_scanning_locations.loc[df_total_scanning_locations.cluster_label == -1]) / len(df_total_scanning_locations)), 2)}"
    )
    print("find hotspots completed.")

    (
        df_dealer_results,
        df_total_scanning_locations,
        df_total_centroids,
        df_suspicious_hotspots_parameters,
    ) = find_suspicious_main(
        df_total_scanning_locations,
        df_total_centroids,
        radius,
        min_samples,
        **thresholds,
    )
    print("find suspicious completed.")

    if save_outputs:
        save_model_outputs(
            df_dealer_results,
            df_total_scanning_locations,
            df_total_centroids,
            df_suspicious_hotspots_parameters,
            year_month_str,
            dealer_region_name,
            product_group_id,
            output_path,
            dense_model=False,
        )

    if show_results:
        show_results_main(
            df_dealer_results,
            df_total_scanning_locations,
            df_total_centroids,
            df_suspicious_hotspots_parameters,
            dealer_scope_dict_path,
            dealer_region_name,
            product_group_id,
            year_month_str,
            save_results=save_results,
        )


def model_run_special(
    df_total_path,
    dealer_scope_dict_path,
    config_file_path,
    output_path,
    dealer_region_name,
    product_group_id,
    year_month_str,
    dbscan_parameters_tuple,
    dbscan_parameters_tuple_dense,
    large_hotspots_threshold,
    save_outputs=True,
    show_results=False,
    save_results=False,
    dense_thresholds={},
    sparse_thresholds={},
):
    """
    total_thresholds = (sparse_thresholds_dict, dense_thresholds_dict)
    **thresholds 可选传入参数包括以下, 不传参时默认值为如下：
        - 'dis_hotspots_c_t': 35,
        - 'dis_points_c_t': 0,
        - 'dis_border_t': 5,
        - 'ratio_scanning_t': 0.1,
        - 'scanning_count_t': 12,
        - 'std_quantile_t': 0.1,
        - 'box_count_t': 4,
    """
    df_cleaned = generate_clean_region_data_main(
        dealer_region_name, df_total_path, product_group_id
    )

    start_date_str, end_date_str = get_month_start_end(year_month_str)
    radius, min_samples = dbscan_parameters_tuple
    radius_dense, min_samples_dense = dbscan_parameters_tuple_dense

    df_total_centroids_sparse, df_total_scanning_locations_sparse = find_hotspots_main(
        df_cleaned,
        product_group_id,
        start_date_str,
        end_date_str,
        radius,
        min_samples,
        config_file_path,
        dealer_scope_dict_path,
    )
    print(
        f"噪声点比： \
        {round((len(df_total_scanning_locations_sparse.loc[df_total_scanning_locations_sparse.cluster_label == -1]) / len(df_total_scanning_locations_sparse)), 2)}"
    )

    (
        df_total_centroids_dense,
        df_total_scanning_locations_dense,
        df_total_centroids_sparse,
    ) = find_hotspots_continue_for_dense_main(
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
    )

    (
        df_dealer_results_sparse,
        df_total_scanning_locations_sparse,
        df_total_centroids_sparse,
        df_suspicious_hotspots_parameters_sparse,
    ) = find_suspicious_main(
        df_total_scanning_locations_sparse,
        df_total_centroids_sparse,
        radius,
        min_samples,
        **sparse_thresholds,
    )

    (
        df_dealer_results_dense,
        df_total_scanning_locations_dense,
        df_total_centroids_dense,
        df_suspicious_hotspots_parameters_dense,
    ) = find_suspicious_main(
        df_total_scanning_locations_dense,
        df_total_centroids_dense,
        radius_dense,
        min_samples_dense,
        **dense_thresholds,
    )

    # 'dealer_suspicious_points_count_final'
    df_total_scanning_locations_sparse = df_total_scanning_locations_sparse.merge(
        df_total_scanning_locations_dense[
            ["BARCODE_BOTTLE", "is_within_suspicious_hotspots"]
        ],
        how="left",
        on="BARCODE_BOTTLE",
        suffixes=("_from_sparse", "_from_dense"),
    )
    df_total_scanning_locations_sparse["is_within_suspicious_hotspots_from_dense"] = (
        df_total_scanning_locations_sparse[
            "is_within_suspicious_hotspots_from_dense"
        ].fillna(0)
    )
    df_total_scanning_locations_sparse["is_within_suspicious_hotspots_final"] = (
        (
            df_total_scanning_locations_sparse[
                "is_within_suspicious_hotspots_from_sparse"
            ]
            == 1
        )
        | (
            df_total_scanning_locations_sparse[
                "is_within_suspicious_hotspots_from_dense"
            ]
            == 1
        )
    ).astype(int)
    df_dealer_suspicious_points_final = (
        df_total_scanning_locations_sparse.groupby(by="BELONG_DEALER_NO")[
            "is_within_suspicious_hotspots_final"
        ]
        .sum()
        .reset_index(name="dealer_suspicious_points_count")
    )
    dealer_suspicious_points_final = df_dealer_suspicious_points_final.set_index(
        "BELONG_DEALER_NO"
    )["dealer_suspicious_points_count"].to_dict()
    df_dealer_results_sparse["dealer_suspicious_points_count_final"] = (
        df_dealer_results_sparse["BELONG_DEALER_NO"].map(dealer_suspicious_points_final)
    )

    # 'dealer_suspicious_hotspot_count_final', 'dealer_hotspot_count_final'
    df_dealer_results_sparse = df_dealer_results_sparse.merge(
        df_dealer_results_dense[
            [
                "BELONG_DEALER_NO",
                "dealer_suspicious_hotspot_count",
                "dealer_hotspot_count",
            ]
        ],
        how="left",
        on="BELONG_DEALER_NO",
        suffixes=("_from_sparse", "_from_dense"),
    )
    df_dealer_results_sparse["dealer_suspicious_hotspot_count_from_dense"] = (
        df_dealer_results_sparse["dealer_suspicious_hotspot_count_from_dense"].fillna(0)
    )
    df_dealer_results_sparse["dealer_suspicious_hotspot_count_final"] = (
        df_dealer_results_sparse["dealer_suspicious_hotspot_count_from_sparse"]
        + df_dealer_results_sparse["dealer_suspicious_hotspot_count_from_dense"]
    ).astype(int)

    df_dealer_results_sparse["dealer_hotspot_count_from_dense"] = (
        df_dealer_results_sparse["dealer_hotspot_count_from_dense"].fillna(0)
    )
    df_dealer_results_sparse["dealer_hotspot_count_final"] = (
        df_dealer_results_sparse["dealer_hotspot_count_from_sparse"]
        + df_dealer_results_sparse["dealer_hotspot_count_from_dense"]
    ).astype(int)
    df_dealer_results_sparse["dealer_suspicious_hotspot_ratio_final"] = (
        df_dealer_results_sparse["dealer_suspicious_hotspot_count_final"]
        / df_dealer_results_sparse["dealer_hotspot_count_final"]
    )

    # 'dealer_suspicious_points_ratio_final'
    df_dealer_results_sparse["dealer_suspicious_points_ratio_final"] = (
        df_dealer_results_sparse["dealer_suspicious_points_count_final"]
        / df_dealer_results_sparse["dealer_total_scanning_count"]
    )

    df_dealer_results_sparse[
        [
            "dealer_suspicious_points_count_final",
            "dealer_suspicious_hotspot_count_final",
        ]
    ] = df_dealer_results_sparse[
        [
            "dealer_suspicious_points_count_final",
            "dealer_suspicious_hotspot_count_final",
        ]
    ].astype(
        int
    )

    df_dealer_results_sparse["is_dealer_suspicious_final"] = 0
    dealer_suspicious_mask = (
        df_dealer_results_sparse["is_dealer_within_archive"] == 1
    ) & (df_dealer_results_sparse["dealer_suspicious_points_count_final"] > 0)

    df_dealer_results_sparse.loc[
        dealer_suspicious_mask, "is_dealer_suspicious_final"
    ] = 1

    if save_outputs:
        save_model_outputs(
            df_dealer_results_sparse,
            df_total_scanning_locations_sparse,
            df_total_centroids_sparse,
            df_suspicious_hotspots_parameters_sparse,
            year_month_str,
            dealer_region_name,
            product_group_id,
            output_path,
            dense_model=False,
        )
        save_model_outputs(
            df_dealer_results_dense,
            df_total_scanning_locations_dense,
            df_total_centroids_dense,
            df_suspicious_hotspots_parameters_dense,
            year_month_str,
            dealer_region_name,
            product_group_id,
            output_path,
            dense_model=True,
        )

    if show_results:
        show_results_special_main(
            df_dealer_results_sparse,
            df_total_scanning_locations_sparse,
            df_total_centroids_sparse,
            df_suspicious_hotspots_parameters_sparse,
            df_dealer_results_dense,
            df_total_scanning_locations_dense,
            df_total_centroids_dense,
            df_suspicious_hotspots_parameters_dense,
            dealer_scope_dict_path,
            dealer_region_name,
            product_group_id,
            year_month_str,
            save_results=save_results,
        )


def show_results_from_raw_outputs(
    output_path,
    dealer_scope_dict_path,
    dealer_region_name,
    product_group_id,
    year_month_str,
    short_results=False,
    dense_model=False,
    save_results=True,
):

    start_date_str, end_date_str = get_month_start_end(year_month_str)
    folder_path = (
        f"{output_path}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )

    if not os.path.exists(folder_path):
        print(f"没有找到数据路径: {folder_path}")
    else:
        (
            df_dealer_results,
            df_total_centroids,
            df_total_scanning_locations,
            df_suspicious_hotspots_parameters,
        ) = read_outputs(folder_path, dense_model=False)

        if dense_model:
            (
                df_dealer_results_dense,
                df_total_centroids_dense,
                df_total_scanning_locations_dense,
                df_suspicious_hotspots_parameters_dense,
            ) = read_outputs(folder_path, dense_model=True)
            if short_results:
                show_region_short_results_special_main(
                    df_dealer_results,
                    df_total_scanning_locations,
                    start_date_str,
                    end_date_str,
                    dealer_region_name,
                )
            else:
                show_results_special_main(
                    df_dealer_results,
                    df_total_scanning_locations,
                    df_total_centroids,
                    df_suspicious_hotspots_parameters,
                    df_dealer_results_dense,
                    df_total_scanning_locations_dense,
                    df_total_centroids_dense,
                    df_suspicious_hotspots_parameters_dense,
                    dealer_scope_dict_path,
                    dealer_region_name,
                    product_group_id,
                    year_month_str,
                    save_results=save_results,
                )

        else:
            if short_results:
                show_region_short_results_main(
                    df_dealer_results,
                    df_total_scanning_locations,
                    start_date_str,
                    end_date_str,
                    dealer_region_name,
                )
            else:
                show_results_main(
                    df_dealer_results,
                    df_total_scanning_locations,
                    df_total_centroids,
                    df_suspicious_hotspots_parameters,
                    dealer_scope_dict_path,
                    dealer_region_name,
                    product_group_id,
                    year_month_str,
                    save_results=save_results,
                )


def main_run_model(
    dealer_region_name,
    product_group_id,
    year_month_str,
    workspace_folder_path="./",
    save_outputs=True,
    show_results=False,
    save_results=False,
):

    (
        df_total_path,
        dealer_scope_dict_path,
        config_file_path,
        parameters_config_file_path,
        output_folder,
    ) = extract_paths(workspace_folder_path, year_month_str)

    config, special_dealer_region_names = load_model_parameters_config(
        dealer_region_name, product_group_id, parameters_config_file_path
    )
    print(f"{dealer_region_name} - {product_group_id} \n config: {config}")
    dbscan_parameters_tuple = config.get("dbscan_parameters_tuple")
    dbscan_parameters_tuple_dense = config.get("dbscan_parameters_tuple_dense")
    large_hotspots_threshold = config.get("large_hotspots_threshold")
    dense_thresholds = config.get("dense_thresholds")
    thresholds = config.get("thresholds", {})

    # special_dealer_region_names = ["天津大区", "北京大区", "上海大区"]
    if [dealer_region_name, product_group_id] not in special_dealer_region_names:
        model_run(
            df_total_path,
            dealer_scope_dict_path,
            config_file_path,
            output_folder,
            dealer_region_name,
            product_group_id,
            year_month_str,
            dbscan_parameters_tuple,
            save_outputs,
            show_results,
            save_results,
            thresholds,
        )
    else:
        model_run_special(
            df_total_path,
            dealer_scope_dict_path,
            config_file_path,
            output_folder,
            dealer_region_name,
            product_group_id,
            year_month_str,
            dbscan_parameters_tuple,
            dbscan_parameters_tuple_dense,
            large_hotspots_threshold,
            save_outputs,
            show_results,
            save_results,
            dense_thresholds,
            thresholds,
        )

    print("Run model Ends!")
    print("_" * 100)


def main_show_region_results(
    dealer_region_name,
    product_group_id,
    year_month_str,
    short_results=False,
    workspace_folder_path="./",
    save_results=True,
):

    _, dealer_scope_dict_path, _, parameters_config_file_path, output_folder = extract_paths(
        workspace_folder_path, year_month_str
    )

    _, special_dealer_region_names = load_model_parameters_config(
        dealer_region_name, product_group_id, parameters_config_file_path
    )

    dense_model = False
    # special_dealer_region_names = [["天津大区", "01"], ["广东大区"， "01"]]

    if [dealer_region_name, product_group_id] in special_dealer_region_names:
        dense_model = True

    show_results_from_raw_outputs(
        output_folder,
        dealer_scope_dict_path,
        dealer_region_name,
        product_group_id,
        year_month_str,
        short_results=short_results,
        dense_model=dense_model,
        save_results=save_results,
    )


def main_show_dealer_results(
    dealer_region_name,
    product_group_id,
    year_month_str,
    dealer_id,
    workspace_folder_path="./",
):

    _, dealer_scope_dict_path, _, parameters_config_file_path, output_folder = extract_paths(
        workspace_folder_path, year_month_str
    )

    output_files_path = (
        f"{output_folder}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )

    if not os.path.exists(output_files_path):
        print(f"没有找到数据路径: {output_files_path}")

    else:
        (
            df_dealer_results,
            df_total_centroids,
            df_total_scanning_locations,
            df_suspicious_hotspots_parameters,
        ) = read_outputs(output_files_path, dense_model=False)

        df_dealer_dealer_results = df_dealer_results.loc[
            df_dealer_results.BELONG_DEALER_NO == dealer_id, :
        ].reset_index(drop=True)
        df_dealer_total_scanning_locations = df_total_scanning_locations.loc[
            df_total_scanning_locations.BELONG_DEALER_NO == dealer_id, :
        ].reset_index(drop=True)
        df_dealer_total_centroids = df_total_centroids.loc[
            df_total_centroids.dealer_id == dealer_id, :
        ].reset_index(drop=True)

        if df_dealer_total_scanning_locations.empty:
            print(
                f"{dealer_id}(品项：{product_group_id}) 不属于{dealer_region_name} 或 在{year_month_str}无扫码记录"
            )
            return

        ######## 要检查是否为空 #########
        _, special_dealer_region_names = load_model_parameters_config(
            dealer_region_name, product_group_id, parameters_config_file_path
        )
        # special_dealer_region_names = ["天津大区", "北京大区", "上海大区"]

        if [dealer_region_name, product_group_id] in special_dealer_region_names:
            (
                df_dealer_results_dense,
                df_total_centroids_dense,
                df_total_scanning_locations_dense,
                df_suspicious_hotspots_parameters_dense,
            ) = read_outputs(output_files_path, dense_model=True)

            df_dealer_dealer_results_dense = df_dealer_results_dense.loc[
                df_dealer_results_dense.BELONG_DEALER_NO == dealer_id, :
            ].reset_index(drop=True)
            df_dealer_total_scanning_locations_dense = (
                df_total_scanning_locations_dense.loc[
                    df_total_scanning_locations_dense.BELONG_DEALER_NO == dealer_id, :
                ].reset_index(drop=True)
            )
            df_dealer_total_centroids_dense = df_total_centroids_dense.loc[
                df_total_centroids_dense.dealer_id == dealer_id, :
            ].reset_index(drop=True)

            show_dealer_results_special_main(
                df_dealer_dealer_results,
                df_dealer_total_scanning_locations,
                df_dealer_total_centroids,
                df_dealer_dealer_results_dense,
                df_dealer_total_scanning_locations_dense,
                df_dealer_total_centroids_dense,
                dealer_scope_dict_path,
            )

        else:
            show_dealer_results_main(
                df_dealer_dealer_results,
                df_dealer_total_scanning_locations,
                df_dealer_total_centroids,
                dealer_scope_dict_path,
            )
