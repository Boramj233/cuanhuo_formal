import pandas as pd
import numpy as np

from geopy.distance import geodesic
from .data_clean import main_generate_clean_region_data
from .find_hotspots import main_find_hotspots, find_hotspots_for_region
from .find_suspicious import main_find_suspicious
from .show_results import main_show_results, main_show_results_special

### IMPORTANT ###
def main_model_run(df_total_path, dealer_scope_dict_path, config_file_path, dealer_region_name, product_group_id, date_str_tuple, dbscan_parameters_tuple, **thresholds):
    """
        **thresholds 可选传入参数包括以下, 不传参时默认值为如下：
            - 'dis_to_all_local_hotspots_centroid_threshold': 35,
            - 'dis_to_all_local_points_centroid_threshold': 0,
            - 'ratio_scanning_count_threshold': 0.1,
            - 'scanning_count_within_cluster_threshold': 12,
            - 'std_distance_within_cluster_quantile_threshold': 0.1,
            - 'box_count_within_cluster_threshold': 4,
    """
    start_date_str, end_date_str = date_str_tuple
    radius, min_samples = dbscan_parameters_tuple

    df_cleaned = main_generate_clean_region_data(dealer_region_name, df_total_path)
    # print(f'cleaned: {len(df_cleaned)}')

    df_total_centroids, df_total_scanning_locations = \
    main_find_hotspots(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path)
    # print(f'find_hotspots: {len(df_total_scanning_locations)}')
 
    df_suspicious_dealers, df_total_centroids, df_suspicious_hotspots_parameters, df_dealers_without_archive=\
        main_find_suspicious(df_total_scanning_locations, df_total_centroids, **thresholds)
    # print(f'find_suspicious: {len(df_total_scanning_locations)}')
    # print(df_dealers_without_archive)
    
    main_show_results(df_total_scanning_locations, df_total_centroids, df_suspicious_dealers, df_suspicious_hotspots_parameters, \
                    dealer_scope_dict_path, start_date_str, end_date_str, dealer_region_name, radius, min_samples, config_file_path)
    



def main_model_run_special(df_total_path, dealer_scope_dict_path, config_file_path, dealer_region_name, product_group_id, date_str_tuple, \
                           dbscan_parameters_tuple, dbscan_parameters_tuple_dense, large_hotspots_threshold = 60, total_thresholds = ({}, {})):
    """
        total_thresholds
        **thresholds 可选传入参数包括以下, 不传参时默认值为如下：
            - 'dis_to_all_local_hotspots_centroid_threshold': 35,
            - 'dis_to_all_local_points_centroid_threshold': 0,
            - 'ratio_scanning_count_threshold': 0.1,
            - 'scanning_count_within_cluster_threshold': 12,
            - 'std_distance_within_cluster_quantile_threshold': 0.1,
            - 'box_count_within_cluster_threshold': 4,
    """
    sparse_thresholds, dense_thresholds = total_thresholds[0], total_thresholds[1]
    df_cleaned = main_generate_clean_region_data(dealer_region_name, df_total_path)

    start_date_str, end_date_str = date_str_tuple
    radius, min_samples = dbscan_parameters_tuple
    df_total_centroids_sparse, df_total_scanning_locations_sparse = \
    main_find_hotspots(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path)
    
    large_hotspots_mask = (~(df_total_centroids_sparse['cluster_label'].isin([-1, -2]))) &\
                        (df_total_centroids_sparse['scanning_count_within_cluster'] >= large_hotspots_threshold)
    
    df_large_hotspots= df_total_centroids_sparse.loc[large_hotspots_mask, :]
    df_large_hotspots_label = df_large_hotspots.loc[:, ['dealer_id', 'cluster_label']]
    df_large_hotspots_label = df_large_hotspots_label.reset_index(drop=True).rename(columns={
        'dealer_id': 'BELONG_DEALER_NO'
    })

    df_large_hotspots_scanning_locations = pd.merge(df_large_hotspots_label, df_total_scanning_locations_sparse, on=['BELONG_DEALER_NO', 'cluster_label'], how='left')
    # print(len(df_large_hotspots_scanning_locations))
    # print(len(df_total_scanning_locations_sparse))

    print(f'大型热点内的扫码点数量占比: {len(df_large_hotspots_scanning_locations) / len(df_total_scanning_locations_sparse)}')
    set_diff =  set(df_large_hotspots_scanning_locations.columns) - set(df_cleaned.columns) 
    df_large_hotspots_scanning_locations_cleaned = df_large_hotspots_scanning_locations.drop(columns=set_diff)


    # dense part

    radius_dense, min_samples_dense = dbscan_parameters_tuple_dense

    df_total_centroids_dense, df_total_scanning_locations_dense, dealers_not_within_archive_dense =\
        find_hotspots_for_region(df_large_hotspots_scanning_locations_cleaned, product_group_id, start_date_str, end_date_str, radius_dense, min_samples_dense, config_file_path,
                                    dealer_scope_dict_path)
    df_total_centroids_dense = df_total_centroids_dense.drop(columns=['total_scanning_count_for_dealer', 'box_count_ratio_for_cluster', 'ratio_scanning_count'])

    # dis_to_all_local_points_centroid, centroid_all_local_points_coordinate
    df_all_local_points_location = df_total_centroids_sparse.loc[:, ['dealer_id', 'centroid_all_local_points_coordinate']].drop_duplicates()
    df_total_centroids_dense = pd.merge(df_total_centroids_dense, df_all_local_points_location, on='dealer_id', how='left')

    def calculate_distance_points(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if pd.isna(row["LATITUDE"]) or pd.isna(row["LONGITUDE"]) or pd.isna(row["centroid_all_local_points_coordinate"]) :
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_points_coordinate"]
        ).kilometers

    df_total_centroids_dense["dis_to_all_local_points_centroid"] = df_total_centroids_dense.apply(
            calculate_distance_points,
            axis=1
        )
    df_total_centroids_dense["dis_to_all_local_points_centroid"] = np.round(df_total_centroids_dense["dis_to_all_local_points_centroid"], 2)

    # dis_to_all_local_hotspots_centroid, centroid_all_local_hotspots_coordinate
    df_all_local_hotspots_location = df_total_centroids_sparse.loc[:, ['dealer_id', 'centroid_all_local_hotspots_coordinate']].drop_duplicates()
    df_total_centroids_dense = pd.merge(df_total_centroids_dense, df_all_local_hotspots_location, on='dealer_id', how='left')
    # print(df_total_centroids_dense.shape)
    def calculate_distance_hotspots(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if pd.isna(row["LATITUDE"]) or pd.isna(row["LONGITUDE"]) or pd.isna(row["centroid_all_local_hotspots_coordinate"]) :
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_hotspots_coordinate"]
        ).kilometers

    df_total_centroids_dense["dis_to_all_local_hotspots_centroid"] = df_total_centroids_dense.apply(
            calculate_distance_hotspots,
            axis=1
        )
    df_total_centroids_dense["dis_to_all_local_hotspots_centroid"] = np.round(df_total_centroids_dense["dis_to_all_local_hotspots_centroid"], 2)

    # ratio_scanning_count ,'total_scanning_count_for_dealer'
    df_total_centroids_dense = pd.merge(df_total_centroids_dense, df_total_centroids_sparse[['dealer_id', 'total_scanning_count_for_dealer']].drop_duplicates(), 
                                        on='dealer_id', how='left')
    df_total_centroids_dense['ratio_scanning_count'] = (df_total_centroids_dense['scanning_count_within_cluster'] / df_total_centroids_dense['total_scanning_count_for_dealer']).round(2)
    # print(df_total_centroids_dense.shape)




    # df_suspicious_dealers_sparse, df_total_centroids_sparse, df_suspicious_hotspots_parameters_sparse, df_dealers_without_archive_sparse=\
    # main_find_suspicious(df_total_scanning_locations_sparse, df_total_centroids_sparse, dis_to_all_local_hotspots_centroid_threshold=20)

    # df_suspicious_dealers_dense, df_total_centroids_dense, df_suspicious_hotspots_parameters_dense, df_dealers_without_archive_dense = \
    # main_find_suspicious(df_total_scanning_locations_dense, df_total_centroids_dense, dis_to_all_local_hotspots_centroid_threshold = 0,
    #                           dis_to_all_local_points_centroid_threshold = 10)

    df_suspicious_dealers_sparse, df_total_centroids_sparse, df_suspicious_hotspots_parameters_sparse, df_dealers_without_archive_sparse=\
    main_find_suspicious(df_total_scanning_locations_sparse, df_total_centroids_sparse, **sparse_thresholds)

    df_suspicious_dealers_dense, df_total_centroids_dense, df_suspicious_hotspots_parameters_dense, df_dealers_without_archive_dense = \
    main_find_suspicious(df_total_scanning_locations_dense, df_total_centroids_dense, **dense_thresholds)
    
    main_show_results_special(df_total_scanning_locations_sparse, df_total_centroids_sparse, df_suspicious_dealers_sparse, df_suspicious_hotspots_parameters_sparse, 
                              df_total_scanning_locations_dense, df_total_centroids_dense, df_suspicious_dealers_dense, df_suspicious_hotspots_parameters_dense,
                    dealer_scope_dict_path, start_date_str, end_date_str, dealer_region_name, radius, min_samples, radius_dense, min_samples_dense,
                    config_file_path)