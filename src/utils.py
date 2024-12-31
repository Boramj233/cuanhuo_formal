
from concurrent.futures import ProcessPoolExecutor
from .data_clean import main_generate_clean_region_data
from .find_hotspots import main_find_hotspots, main_find_hotspots_special
from .find_suspicious import main_find_suspicious
from .show_results import main_show_results, main_show_results_special


### IMPORTANT ###
def main_model_run(df_total_path, dealer_scope_dict_path, config_file_path, dealer_region_name, product_group_id, date_str_tuple, dbscan_parameters_tuple, 
                   is_cal_border_dis=False, **thresholds):
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
    start_date_str, end_date_str = date_str_tuple
    radius, min_samples = dbscan_parameters_tuple

    df_cleaned = main_generate_clean_region_data(dealer_region_name, df_total_path)
    # print(f'cleaned: {len(df_cleaned)}')

    df_total_centroids, df_total_scanning_locations = \
    main_find_hotspots(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, cal_border=is_cal_border_dis)
    # print(f'find_hotspots: {len(df_total_scanning_locations)}')
 
    df_suspicious_dealers, df_total_centroids, df_suspicious_hotspots_parameters, df_dealers_without_archive=\
        main_find_suspicious(df_total_scanning_locations, df_total_centroids, is_cal_border_dis=is_cal_border_dis, **thresholds)
    # print(f'find_suspicious: {len(df_total_scanning_locations)}')
    # print(df_dealers_without_archive)
    
    main_show_results(df_total_scanning_locations, df_total_centroids, df_suspicious_dealers, df_suspicious_hotspots_parameters, \
                    dealer_scope_dict_path, start_date_str, end_date_str, dealer_region_name, radius, min_samples, config_file_path, is_cal_border_dis=is_cal_border_dis)
    



def main_model_run_special(df_total_path, dealer_scope_dict_path, config_file_path, dealer_region_name, product_group_id, date_str_tuple, \
                           dbscan_parameters_tuple, dbscan_parameters_tuple_dense, large_hotspots_threshold = 60, total_thresholds = ({}, {})):
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

    start_date_str, end_date_str = date_str_tuple
    radius, min_samples = dbscan_parameters_tuple
    radius_dense, min_samples_dense = dbscan_parameters_tuple_dense

 
    # with ProcessPoolExecutor() as executor:
    #     future_sparse = executor.submit(main_find_hotspots, df_cleaned, product_group_id, start_date_str, end_date_str,
    #                                     radius, min_samples, config_file_path, dealer_scope_dict_path, cal_border=True)

    #     # 等待 sparse 结果后执行 dense
    #     df_total_centroids_sparse, df_total_scanning_locations_sparse = future_sparse.result()

    #     future_dense = executor.submit(main_find_hotspots_special, df_total_centroids_sparse, df_total_scanning_locations_sparse,
    #                                 large_hotspots_threshold, df_cleaned, product_group_id, start_date_str, end_date_str,
    #                                 radius_dense, min_samples_dense, config_file_path, dealer_scope_dict_path)

    #     df_total_centroids_dense, df_total_scanning_locations_dense = future_dense.result()

    df_total_centroids_sparse, df_total_scanning_locations_sparse = \
    main_find_hotspots(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, cal_border=True)

    df_total_centroids_dense, df_total_scanning_locations_dense = main_find_hotspots_special(df_total_centroids_sparse, df_total_scanning_locations_sparse, large_hotspots_threshold,
                                                                                    df_cleaned, product_group_id, start_date_str, end_date_str, 
                                                                                    radius_dense, min_samples_dense, config_file_path, dealer_scope_dict_path)

    df_suspicious_dealers_sparse, df_total_centroids_sparse, df_suspicious_hotspots_parameters_sparse, df_dealers_without_archive_sparse=\
    main_find_suspicious(df_total_scanning_locations_sparse, df_total_centroids_sparse, is_cal_border_dis=True, **sparse_thresholds)

    df_suspicious_dealers_dense, df_total_centroids_dense, df_suspicious_hotspots_parameters_dense, df_dealers_without_archive_dense = \
    main_find_suspicious(df_total_scanning_locations_dense, df_total_centroids_dense, is_cal_border_dis=True, **dense_thresholds)
    
    main_show_results_special(df_total_scanning_locations_sparse, df_total_centroids_sparse, df_suspicious_dealers_sparse, df_suspicious_hotspots_parameters_sparse, 
                              df_total_scanning_locations_dense, df_total_centroids_dense, df_suspicious_dealers_dense, df_suspicious_hotspots_parameters_dense,
                    dealer_scope_dict_path, start_date_str, end_date_str, dealer_region_name, radius, min_samples, radius_dense, min_samples_dense,
                    config_file_path)
    