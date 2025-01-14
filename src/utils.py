from .data_clean import main_generate_clean_region_data
from .find_hotspots import main_find_hotspots, main_find_hotspots_continue_for_dense
from .find_suspicious import main_find_suspicious
from .libs import get_month_start_end, extract_paths, read_outputs, save_model_results
from .show_results import (
    main_show_results,
    main_show_results_special,
    main_show_region_short_results,
    main_show_region_short_results_special,
    main_show_dealer_results,
    main_show_dealer_results_special,
)

import os
import pandas as pd


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

    df_dealer_results, df_total_scanning_locations, df_total_centroids, df_suspicious_hotspots_parameters = (
        main_find_suspicious(
            df_total_scanning_locations,
            df_total_centroids,
            radius,
            min_samples,
            **thresholds
        )
    )

    if save_results:
        save_model_results(
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

    df_total_centroids_dense, df_total_scanning_locations_dense, df_total_centroids_sparse = (
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

    
    df_dealer_results_sparse, df_total_scanning_locations_sparse, df_total_centroids_sparse, df_suspicious_hotspots_parameters_sparse = main_find_suspicious(
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
    ) = main_find_suspicious(
        df_total_scanning_locations_dense,
        df_total_centroids_dense,
        radius_dense,
        min_samples_dense,
        **dense_thresholds,
    )

    # 'dealer_suspicious_points_count_final'
    df_total_scanning_locations_sparse = df_total_scanning_locations_sparse.merge(
        df_total_scanning_locations_dense[['BARCODE_BOTTLE', 'is_within_suspicious_hotspots']],
        how='left', on='BARCODE_BOTTLE', suffixes=('_from_sparse', '_from_dense')
        )
    df_total_scanning_locations_sparse['is_within_suspicious_hotspots_from_dense'] =\
        df_total_scanning_locations_sparse['is_within_suspicious_hotspots_from_dense'].fillna(0)
    df_total_scanning_locations_sparse['is_within_suspicious_hotspots_final'] = (
        (df_total_scanning_locations_sparse['is_within_suspicious_hotspots_from_dense'] == 1) | 
        (df_total_scanning_locations_sparse['is_within_suspicious_hotspots_from_dense'] == 1)
        ).astype(int)
    df_dealer_suspicious_points_final = \
        df_total_scanning_locations_sparse.groupby(by='BELONG_DEALER_NO')['is_within_suspicious_hotspots_final'].sum()\
        .reset_index(name='dealer_suspicious_points_count')
    dealer_suspicious_points_final = df_dealer_suspicious_points_final.set_index('BELONG_DEALER_NO')['dealer_suspicious_points_count'].to_dict()
    df_dealer_results_sparse['dealer_suspicious_points_count_final'] = df_dealer_results_sparse['BELONG_DEALER_NO'].map(dealer_suspicious_points_final)

    # 'dealer_suspicious_hotspot_count_final'
    df_dealer_results_sparse = df_dealer_results_sparse.merge(
        df_dealer_results_dense[["BELONG_DEALER_NO", 'dealer_suspicious_hotspot_count']],
        how='left', on='BELONG_DEALER_NO', suffixes=('_from_sparse', '_from_dense')
        )
    df_dealer_results_sparse['dealer_suspicious_hotspot_count_from_dense'] = \
        df_dealer_results_sparse['dealer_suspicious_hotspot_count_from_dense'].fillna(0)
    df_dealer_results_sparse['dealer_suspicious_hotspot_count_final'] = \
        (df_dealer_results_sparse['dealer_suspicious_hotspot_count_from_sparse'] + df_dealer_results_sparse['dealer_suspicious_hotspot_count_from_dense']).astype(int)
    
    # 'dealer_suspicious_points_ratio_final'
    df_dealer_results_sparse['dealer_suspicious_points_ratio_final'] = \
        (df_dealer_results_sparse['dealer_suspicious_points_count_final'] / df_dealer_results_sparse['dealer_total_scanning_count'])
    
    df_dealer_results_sparse[['dealer_suspicious_points_count_final', 'dealer_suspicious_hotspot_count_final']] = \
        df_dealer_results_sparse[['dealer_suspicious_points_count_final', 'dealer_suspicious_hotspot_count_final']].astype(int)
    
    if save_results:
        save_model_results(
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
        save_model_results(
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
        df_dealer_results, df_total_centroids, df_total_scanning_locations, df_suspicious_hotspots_parameters =\
            read_outputs(folder_path, dense_model=False)

        if dense_model:
            df_dealer_results_dense, df_total_centroids_dense, df_total_scanning_locations_dense, df_suspicious_hotspots_parameters_dense =\
                read_outputs(folder_path, dense_model=True)
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
        df_dealer_results, df_total_centroids, df_total_scanning_locations, df_suspicious_hotspots_parameters =\
            read_outputs(output_files_path, dense_model=False)
        
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
            df_dealer_results_dense, df_total_centroids_dense, df_total_scanning_locations_dense, df_suspicious_hotspots_parameters_dense =\
                read_outputs(output_files_path, dense_model=False)

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
