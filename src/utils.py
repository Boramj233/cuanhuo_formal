from .data_clean import main_generate_clean_region_data
from .find_hotspots import main_find_hotspots, main_find_hotspots_continue_for_dense
from .find_suspicious import main_find_suspicious
from .libs import get_month_start_end, load_from_raw_outputs, extract_paths
from .show_results import (
    main_show_results,
    main_show_results_special,
    main_show_region_short_results,
    main_show_region_short_results_special,
    main_show_dealer_results,
    main_show_dealer_results_special,
)

import os


### IMPORTANT ###
def main_model_run(
    df_total_path,
    dealer_scope_dict_path,
    config_file_path,
    output_path,
    dealer_region_name,
    product_group_id,
    year_month_str,
    dbscan_parameters_tuple,
    save_results=True,
    show_results=True,
    **thresholds,
):
    """
    **thresholds 可选传入参数包括以下, 不传参时默认值为如下：
        - 'dis_hotspots_c_t': 35,
        - 'dis_points_c_t': 0,
        - 'ratio_scanning_t': 0.1,
        - 'scanning_count_t': 12,
        - 'std_quantile_t': 0.1,
        - 'box_count_t': 4,

        当 is_cal_border_dis = True:
        - 'dis_border' : 5
    """
    df_cleaned = main_generate_clean_region_data(dealer_region_name, df_total_path)

    start_date_str, end_date_str = get_month_start_end(year_month_str)
    radius, min_samples = dbscan_parameters_tuple

    # print(f'cleaned: {len(df_cleaned)}')

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
    # print(f'find_hotspots: {len(df_total_scanning_locations)}')

    df_dealer_results, df_total_centroids, df_suspicious_hotspots_parameters = (
        main_find_suspicious(
            df_total_scanning_locations,
            df_total_centroids,
            year_month_str,
            dealer_region_name,
            product_group_id,
            output_path,
            radius,
            min_samples,
            save_results=save_results,
            dense_model=False,
            **thresholds,
        )
    )
    # print(f'find_suspicious: {len(df_total_scanning_locations)}')
    # print(df_dealers_without_archive)

    if show_results:
        main_show_results(
            df_dealer_results,
            df_total_scanning_locations,
            df_total_centroids,
            df_suspicious_hotspots_parameters,
            dealer_scope_dict_path,
            start_date_str,
            end_date_str,
            dealer_region_name,
        )


def main_model_run_special(
    df_total_path,
    dealer_scope_dict_path,
    config_file_path,
    output_path,
    dealer_region_name,
    product_group_id,
    year_month_str,
    dbscan_parameters_tuple,
    dbscan_parameters_tuple_dense,
    save_results=True,
    show_results=True,
    large_hotspots_threshold=60,
    total_thresholds=({}, {}),
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
    sparse_thresholds, dense_thresholds = total_thresholds[0], total_thresholds[1]
    df_cleaned = main_generate_clean_region_data(dealer_region_name, df_total_path)

    start_date_str, end_date_str = get_month_start_end(year_month_str)
    radius, min_samples = dbscan_parameters_tuple
    radius_dense, min_samples_dense = dbscan_parameters_tuple_dense

    df_total_centroids_sparse, df_total_scanning_locations_sparse = main_find_hotspots(
        df_cleaned,
        product_group_id,
        start_date_str,
        end_date_str,
        radius,
        min_samples,
        config_file_path,
        dealer_scope_dict_path,
    )

    df_total_centroids_dense, df_total_scanning_locations_dense = (
        main_find_hotspots_continue_for_dense(
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
    )

    (
        df_dealer_results_sparse,
        df_total_centroids_sparse,
        df_suspicious_hotspots_parameters_sparse,
    ) = main_find_suspicious(
        df_total_scanning_locations_sparse,
        df_total_centroids_sparse,
        year_month_str,
        dealer_region_name,
        product_group_id,
        output_path,
        radius,
        min_samples,
        save_results=save_results,
        dense_model=False,
        **sparse_thresholds,
    )

    (
        df_dealer_results_dense,
        df_total_centroids_dense,
        df_suspicious_hotspots_parameters_dense,
    ) = main_find_suspicious(
        df_total_scanning_locations_dense,
        df_total_centroids_dense,
        year_month_str,
        dealer_region_name,
        product_group_id,
        output_path,
        radius_dense,
        min_samples_dense,
        save_results=save_results,
        dense_model=True,
        **dense_thresholds,
    )

    if show_results:
        main_show_results_special(
            df_dealer_results_sparse,
            df_total_scanning_locations_sparse,
            df_total_centroids_sparse,
            df_suspicious_hotspots_parameters_sparse,
            df_dealer_results_dense,
            df_total_scanning_locations_dense,
            df_total_centroids_dense,
            df_suspicious_hotspots_parameters_dense,
            dealer_scope_dict_path,
            start_date_str,
            end_date_str,
            dealer_region_name,
        )


def show_results_from_raw_outputs(
    output_path,
    dealer_scope_dict_path,
    dealer_region_name,
    product_group_id,
    year_month_str,
    short_results=False,
    dense_model=False,
):

    start_date_str, end_date_str = get_month_start_end(year_month_str)

    folder_path = (
        f"{output_path}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )

    if not os.path.exists(folder_path):
        print(f"没有找到数据路径: {folder_path}")
    else:

        file_names = [
            "df_dealer_results.pkl",
            "df_total_centroids.pkl",
            "df_total_scanning_locations.parquet",
            "df_suspicious_hotspots_parameters.parquet",
        ]

        df_dealer_results_path = os.path.join(folder_path, file_names[0])
        df_total_centroids_path = os.path.join(folder_path, file_names[1])
        df_total_scanning_locations_path = os.path.join(folder_path, file_names[2])
        df_suspicious_hotspots_parameters_path = os.path.join(
            folder_path, file_names[3]
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

        if dense_model:

            dense_file_names = [
                "df_dealer_results_dense.pkl",
                "df_total_centroids_dense.pkl",
                "df_total_scanning_locations_dense.parquet",
                "df_suspicious_hotspots_parameters_dense.parquet",
            ]
            df_dealer_results_dense_path = os.path.join(
                folder_path, dense_file_names[0]
            )
            df_total_centroids_dense_path = os.path.join(
                folder_path, dense_file_names[1]
            )
            df_total_scanning_locations_dense_path = os.path.join(
                folder_path, dense_file_names[2]
            )
            df_suspicious_hotspots_parameters_dense_path = os.path.join(
                folder_path, dense_file_names[3]
            )

            (
                df_dealer_results_dense,
                df_total_centroids_dense,
                df_total_scanning_locations_dense,
                df_suspicious_hotspots_parameters_dense,
            ) = load_from_raw_outputs(
                df_dealer_results_dense_path,
                df_total_centroids_dense_path,
                df_total_scanning_locations_dense_path,
                df_suspicious_hotspots_parameters_dense_path,
            )

            if short_results:
                main_show_region_short_results_special(
                    df_dealer_results,
                    df_total_scanning_locations,
                    df_dealer_results_dense,
                    start_date_str,
                    end_date_str,
                    dealer_region_name,
                )

            else:
                main_show_results_special(
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
                )
        else:

            if short_results:
                main_show_region_short_results(
                    df_dealer_results,
                    df_total_scanning_locations,
                    start_date_str,
                    end_date_str,
                    dealer_region_name,
                )
            else:
                main_show_results(
                    df_dealer_results,
                    df_total_scanning_locations,
                    df_total_centroids,
                    df_suspicious_hotspots_parameters,
                    dealer_scope_dict_path,
                    start_date_str,
                    end_date_str,
                    dealer_region_name,
                )


def run_model(
    dealer_region_name,
    product_group_id,
    year_month_str,
    dbscan_parameters_tuple=(4, 6),
    dbscan_parameters_tuple_dense=(1, 12),
    large_hotspots_thresholds=60,
    workspace_folder_path="./",
    save_results=True,
    show_results=True,
    special_suspicious_thresholds=({}, {"dis_hotspots_c_t": 0, "dis_points_c_t": 15}),
    **thresholds,
):

    df_total_path, dealer_scope_dict_path, config_file_path, output_folder = (
        extract_paths(workspace_folder_path, year_month_str)
    )

    special_dealer_region_names = ["天津大区", "北京大区", "上海大区"]

    if dealer_region_name not in special_dealer_region_names:
        main_model_run(
            df_total_path,
            dealer_scope_dict_path,
            config_file_path,
            output_folder,
            dealer_region_name,
            product_group_id,
            year_month_str,
            dbscan_parameters_tuple,
            save_results,
            show_results,
            **thresholds,
        )
    else:

        main_model_run_special(
            df_total_path,
            dealer_scope_dict_path,
            config_file_path,
            output_folder,
            dealer_region_name,
            product_group_id,
            year_month_str,
            dbscan_parameters_tuple,
            dbscan_parameters_tuple_dense,
            save_results,
            show_results,
            large_hotspots_thresholds,
            special_suspicious_thresholds,
        )


def show_region_results(
    dealer_region_name,
    product_group_id,
    year_month_str,
    short_results=False,
    workspace_folder_path="./",
):

    _, dealer_scope_dict_path, _, output_folder = extract_paths(
        workspace_folder_path, year_month_str
    )

    dense_model = False
    special_dealer_region_names = ["天津大区", "北京大区", "上海大区"]
    if dealer_region_name in special_dealer_region_names:
        dense_model = True

    show_results_from_raw_outputs(
        output_folder,
        dealer_scope_dict_path,
        dealer_region_name,
        product_group_id,
        year_month_str,
        short_results=short_results,
        dense_model=dense_model,
    )


def show_dealer_results(
    dealer_region_name,
    product_group_id,
    year_month_str,
    dealer_id,
    workspace_folder_path="./",
):

    # start_date_str, end_date_str = get_month_start_end(year_month_str)

    _, dealer_scope_dict_path, _, output_folder = extract_paths(
        workspace_folder_path, year_month_str
    )

    output_files_path = (
        f"{output_folder}/{dealer_region_name}/{product_group_id}/{year_month_str}"
    )

    if not os.path.exists(output_files_path):
        print(f"没有找到数据路径: {output_files_path}")

    else:
        file_names = [
            "df_dealer_results.pkl",
            "df_total_centroids.pkl",
            "df_total_scanning_locations.parquet",
            "df_suspicious_hotspots_parameters.parquet",
        ]

        df_dealer_results_path = os.path.join(output_files_path, file_names[0])
        df_total_centroids_path = os.path.join(output_files_path, file_names[1])
        df_total_scanning_locations_path = os.path.join(
            output_files_path, file_names[2]
        )
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
        special_dealer_region_names = ["天津大区", "北京大区", "上海大区"]
        if dealer_region_name in special_dealer_region_names:
            dense_file_names = [
                "df_dealer_results.pkl",
                "df_total_centroids.pkl",
                "df_total_scanning_locations.parquet",
                "df_suspicious_hotspots_parameters.parquet",
            ]

            df_dealer_results_dense_path = os.path.join(
                output_files_path, dense_file_names[0]
            )
            df_total_centroids_dense_path = os.path.join(
                output_files_path, dense_file_names[1]
            )
            df_total_scanning_locations_dense_path = os.path.join(
                output_files_path, dense_file_names[2]
            )
            df_suspicious_hotspots_parameters_dense_path = os.path.join(
                output_files_path, dense_file_names[3]
            )

            (
                df_dealer_results_dense,
                df_total_centroids_dense,
                df_total_scanning_locations_dense,
                df_suspicious_hotspots_parameters_dense,
            ) = load_from_raw_outputs(
                df_dealer_results_dense_path,
                df_total_centroids_dense_path,
                df_total_scanning_locations_dense_path,
                df_suspicious_hotspots_parameters_dense_path,
            )

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

            # if df_dealer_total_scanning_locations.empty:
            #     print(f'{dealer_id}-{product_group_id} 不属于{dealer_region_name} 或 在{year_month_str}无扫码记录')
            #     return

            main_show_dealer_results_special(
                df_dealer_dealer_results,
                df_dealer_total_scanning_locations,
                df_dealer_total_centroids,
                df_dealer_dealer_results_dense,
                df_dealer_total_scanning_locations_dense,
                df_dealer_total_centroids_dense,
                dealer_scope_dict_path,
            )

        else:
            # if df_dealer_total_scanning_locations.empty:
            #     print(f'{dealer_id}-{product_group_id} 不属于{dealer_region_name} 或 在{year_month_str}无扫码记录')
            #     return
            main_show_dealer_results(
                df_dealer_dealer_results,
                df_dealer_total_scanning_locations,
                df_dealer_total_centroids,
                dealer_scope_dict_path,
            )
