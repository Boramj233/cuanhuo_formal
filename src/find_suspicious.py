from geopy.distance import geodesic

import pandas as pd
import numpy as np


# 考虑增加 all local points 距离作为大簇的计算距离点
def filter_suspicious_hotspot(df_total_centroids, **kwargs):

    hotspot_mask = ~(df_total_centroids['cluster_label'].isin([-2, -1]))

    local_hotspots_std = df_total_centroids.loc[hotspot_mask & (df_total_centroids['is_remote'] == 0), 'std_distance_within_cluster']

    default_thresholds = {
        'dis_to_all_local_hotspots_centroid_threshold': 35,
        'dis_to_all_local_points_centroid_threshold': 0,
        'ratio_scanning_count_threshold': 0.1,
        'scanning_count_within_cluster_threshold': 12,
        'std_distance_within_cluster_quantile_threshold': 0.1,
        'box_count_within_cluster_threshold': 4,
    }

    thresholds = {**default_thresholds, **kwargs}

    dis_to_all_local_hotspots_centroid_threshold = thresholds['dis_to_all_local_hotspots_centroid_threshold']
    dis_to_all_local_points_centroid_threshold = thresholds['dis_to_all_local_points_centroid_threshold']
    ratio_scanning_count_threshold = thresholds['ratio_scanning_count_threshold']
    scanning_count_within_cluster_threshold = thresholds['scanning_count_within_cluster_threshold']

    std_distance_within_cluster_threshold = np.quantile(local_hotspots_std, thresholds['std_distance_within_cluster_quantile_threshold'])
    box_count_within_cluster_threshold = thresholds['box_count_within_cluster_threshold']

    ##################################################################################################################################################
    # 设置可以热点条件
    # verison 1.0

    suspicious_mask = (hotspot_mask) & (df_total_centroids['is_remote'] == 1) &\
        (df_total_centroids['dis_to_all_local_hotspots_centroid'] >= dis_to_all_local_hotspots_centroid_threshold ) & \
        (df_total_centroids['dis_to_all_local_points_centroid'] >= dis_to_all_local_points_centroid_threshold) &\
             ((df_total_centroids['ratio_scanning_count'] >= ratio_scanning_count_threshold) | (df_total_centroids['scanning_count_within_cluster'] >= scanning_count_within_cluster_threshold)) &\
              (
                (df_total_centroids['std_distance_within_cluster'] >= std_distance_within_cluster_threshold) | \
               ((df_total_centroids['std_distance_within_cluster'] < std_distance_within_cluster_threshold) & (df_total_centroids['box_count_within_cluster'] >= box_count_within_cluster_threshold))
              )
    

    ###################################################################################################################################################
    
    df_total_centroids_with_suspicious_label = df_total_centroids.copy()
    df_total_centroids_with_suspicious_label.loc[suspicious_mask, 'is_suspicious'] = 1
    df_total_centroids_with_suspicious_label.loc[~suspicious_mask, 'is_suspicious'] = 0
    df_total_centroids_with_suspicious_label.loc[~hotspot_mask, 'is_suspicious'] = df_total_centroids_with_suspicious_label.loc[~hotspot_mask, 'cluster_label']
    # df_total_centroids_with_suspicious_label['is_suspicious'] = df_total_centroids_with_suspicious_label['is_suspicious'].fillna(0).astype(int)
 
    df_suspicious_hotspots = df_total_centroids_with_suspicious_label.loc[
        df_total_centroids_with_suspicious_label.is_suspicious == 1, :].drop(columns= ['is_suspicious']).reset_index(drop=True)
    
    df_total_centroids_with_suspicious_label['is_suspicious'] = df_total_centroids_with_suspicious_label['is_suspicious'].astype(int)

    df_suspicious_hotspots_parameters = pd.DataFrame(
        {
            '距本地热点总质心的距离阈值':  dis_to_all_local_hotspots_centroid_threshold,
            '距本地扫码点总质心的距离阈值': dis_to_all_local_points_centroid_threshold,
            '该热点扫码量占比阈值': ratio_scanning_count_threshold,
            '簇内扫码量阈值': scanning_count_within_cluster_threshold,
            '簇内离散度阈值': round(std_distance_within_cluster_threshold, 2),
            '紧密热点的箱数阈值': box_count_within_cluster_threshold,
        }, index=[0]
    )
    
    return df_suspicious_hotspots, df_total_centroids_with_suspicious_label, df_suspicious_hotspots_parameters



def find_suspicious_hotspots_generate_dealer_info(df_total_scanning_locations, df_total_centroids, **thresholds):
    df_total_scanning_locations_features = df_total_scanning_locations.copy()
    df_hotspots = df_total_centroids.loc[~(df_total_centroids['cluster_label'].isin([-2, -1])), :]
    df_hotspots = df_hotspots.drop_duplicates(subset=['dealer_id', 'cluster_label']) # 可以不加drop_duplicates

    df_suspicious_hotspots, df_total_centroids, df_suspicious_hotspots_parameters = filter_suspicious_hotspot(df_total_centroids, **thresholds)
    # print(f'std_distance_within_cluster_threshold: {std_distance_within_cluster_threshold}')
    print(f'当前规则下可疑热点数： {len(df_suspicious_hotspots)}')
    # print("=" * 150)
    print()
    df_total_scanning_locations_features['remote_ratio'] = df_total_scanning_locations_features.groupby('BELONG_DEALER_NO')['point_remote_label_new'].transform('mean')
    df_total_scanning_locations_features['total_scanning_count'] = df_total_scanning_locations_features.groupby('BELONG_DEALER_NO')['BARCODE_BOTTLE'].transform('nunique')
    df_total_scanning_locations_features['remote_scanning_count'] = df_total_scanning_locations_features.groupby('BELONG_DEALER_NO')['point_remote_label_new'].transform('sum')

    df_hotspots['dealer_hotspot_count'] = df_hotspots.groupby('dealer_id')['cluster_label'].transform('count')
    df_hotspots['dealer_remote_hotspot_count'] = df_hotspots.groupby('dealer_id')[['is_remote']].transform('sum')
    df_hotspots['remote_hotspot_ratio'] = df_hotspots.groupby('dealer_id')[['is_remote']].transform('mean')
    df_hotspots_to_merge = df_hotspots.loc[:, ['dealer_id', 'dealer_total_box_count', 'dealer_hotspot_count', 'dealer_remote_hotspot_count', 'remote_hotspot_ratio']].drop_duplicates()
    df_hotspots_to_merge = df_hotspots_to_merge.rename(
        columns={
            'dealer_id': 'BELONG_DEALER_NO'
        }
    )

    df_suspicious_hotspots['dealer_suspicious_hotspot_count'] = df_suspicious_hotspots.groupby('dealer_id')['cluster_label'].transform('count')
    df_suspicious_hotspots_to_merge = df_suspicious_hotspots.loc[:, ['dealer_id', 'dealer_suspicious_hotspot_count']].drop_duplicates()
    df_suspicious_hotspots_to_merge = df_suspicious_hotspots_to_merge.rename(
        columns={
            'dealer_id': 'BELONG_DEALER_NO'
        }
    )


    df_total_scanning_locations_features_to_merge = df_total_scanning_locations_features.loc[:, [
        'BELONG_DEALER_NO', 'BELONG_DEALER_NAME','PRODUCT_GROUP_CODE', 'PRODUCT_GROUP_NAME', 'total_scanning_count', 'remote_scanning_count', 'remote_ratio', 'is_dealer_within_archive']].drop_duplicates()

    df_result = pd.merge(df_total_scanning_locations_features_to_merge, df_hotspots_to_merge, on='BELONG_DEALER_NO', how='left')
    df_result = pd.merge(df_result, df_suspicious_hotspots_to_merge, on='BELONG_DEALER_NO', how='left')

    # 将所有缺失值填充为 0
    
    df_result['suspicious_hotspot_ratio'] = round(df_result['dealer_suspicious_hotspot_count'] /  df_result['dealer_hotspot_count'], 2)
    df_result.fillna(0, inplace=True)

    df_result[['dealer_hotspot_count', 'dealer_remote_hotspot_count', 'dealer_suspicious_hotspot_count',  'dealer_total_box_count']] = \
        df_result[['dealer_hotspot_count', 'dealer_remote_hotspot_count', 'dealer_suspicious_hotspot_count', 'dealer_total_box_count']].astype(int)  
 

    df_result[['remote_ratio', 'remote_hotspot_ratio']] = df_result[['remote_ratio', 'remote_hotspot_ratio']].round(2)
    # df_result['col1'] = (df_result['col1'] * 100).apply(lambda x: f'{x:.2f}%')

    df_result_within_archive = df_result.loc[df_result['is_dealer_within_archive'] == 1, :].reset_index(drop=True)
    df_result_without_archive = df_result.loc[df_result['is_dealer_within_archive'] == 0, :].reset_index(drop=True)

    return df_result_within_archive, df_result_without_archive, df_total_centroids, df_suspicious_hotspots_parameters
    
    

def filter_suspicious_dealers(df_result_within_archive):
    ############## 可以经销商条件 ############
    df_suspicious_dealers = df_result_within_archive.loc[
        df_result_within_archive['dealer_suspicious_hotspot_count'] > 0, :
    ].reset_index(drop=True)

    return df_suspicious_dealers


def main_find_suspicious(df_total_scanning_locations, df_total_centroids, **thresholds):
    # df_total_centroids = calculate_distances_to_local_centroids_for_centroids(df_total_scanning_locations, df_total_centroids, dealers_not_within_archive)
    
    df_dealers_within_archive, df_dealers_without_archive, df_total_centroids,  df_suspicious_hotspots_parameters = \
    find_suspicious_hotspots_generate_dealer_info(df_total_scanning_locations, df_total_centroids, **thresholds)

    df_suspicious_dealers = filter_suspicious_dealers(df_dealers_within_archive)

    df_suspicious_dealers = df_suspicious_dealers.sort_values(by='BELONG_DEALER_NO')

    # return df_suspicious_dealers, df_total_centroids, df_suspicious_hotspots_parameters, df_dealers_within_archive, df_dealers_without_archive,  df_suspicious_hotspots
    return df_suspicious_dealers, df_total_centroids, df_suspicious_hotspots_parameters, df_dealers_without_archive