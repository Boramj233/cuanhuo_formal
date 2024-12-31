from datetime import datetime
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

from .libs import get_address_from_lat_lon, get_acode, get_polyline_points

import numpy as np
import pandas as pd
import pickle

### Find hotspots
def find_clusters_for_dealer(df_cleaned, dealer_id, product_group_id, start_date_str, end_date_str, radius, min_samples):

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()

    df_scanning_locations = df_cleaned.loc[((df_cleaned['BELONG_DEALER_NO'] == dealer_id) & \
                          (df_cleaned['CUST_SCAN_DATE'] >= start_date) & \
                          (df_cleaned['CUST_SCAN_DATE'] <= end_date) & (df_cleaned['PRODUCT_GROUP_CODE'] == product_group_id)), :].copy()
    
    if df_scanning_locations.empty:
        print(f"No scanning data for dealer ID {dealer_id} in the specified period.")

    else:
        # 先维度lat, 再经度lon
        coords = df_scanning_locations.loc[:, ['LATITUDE', 'LONGITUDE']].values
        # 将经纬度转换为弧度
        data_rad = np.radians(coords)
        # 使用 DBSCAN 并指定 Haversine 距离
        dbscan = DBSCAN(eps=radius/ 6371, min_samples=min_samples, metric='haversine') # 地球半径6371公里
        df_scanning_locations['cluster_label'] = dbscan.fit_predict(data_rad)

    return df_scanning_locations


def get_centroids(df_scanning_locations, config_file_path):

    if not df_scanning_locations.empty:
        centroids = df_scanning_locations.groupby('cluster_label')[['LATITUDE', 'LONGITUDE']].mean()
        df_centroids = centroids.reset_index()

        # 增加一个所有点位的总质心
        overall_centroid = df_scanning_locations[['LATITUDE', 'LONGITUDE']].mean()

        # 创建新的 DataFrame 行，cluster_label 可以用 '-2' 或者其他标识
        # cluster_label -2 (int) 在 df_centroids 代表整个dealer 所有点的locationd的总质心
        overall_centroid_row = pd.DataFrame({
            'cluster_label': [-2],
            'LATITUDE': [overall_centroid['LATITUDE']],
            'LONGITUDE': [overall_centroid['LONGITUDE']]
        })

        # 将总质心行追加到 df_centroids
        df_centroids = pd.concat([overall_centroid_row, df_centroids], ignore_index=True)

        df_centroids['formatted_address'] = '-1'
        df_centroids['province'] = '-1'
        df_centroids['city'] = '-1'
        df_centroids['district'] = '-1'
        df_centroids['street'] = '-1'

        for i in range(len(df_centroids)):   
            location = f'{round(df_centroids.loc[i, 'LONGITUDE'], 2)}, {round(df_centroids.loc[i, 'LATITUDE'], 2)}'  # Gaode api 经度在前，纬度在后
            address = get_address_from_lat_lon(location, config_file_path)            
            df_centroids.at[i, 'formatted_address'] = address['regeocode']['formatted_address']
            df_centroids.at[i, 'province'] = address['regeocode']['addressComponent']['province']
            df_centroids.at[i, 'city'] = address['regeocode']['addressComponent']['city']
            df_centroids.at[i, 'district'] = address['regeocode']['addressComponent']['district']
            df_centroids.at[i, 'street'] = address['regeocode']['addressComponent']['township']

        return df_centroids
    return pd.DataFrame()


def get_scanning_locations_and_centroids(df_cleaned, dealer_id, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path):
    df_scanning_loactions = find_clusters_for_dealer(df_cleaned, dealer_id, product_group_id, start_date_str, end_date_str, radius, min_samples)
    if df_scanning_loactions.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_centroids = get_centroids(df_scanning_loactions, config_file_path)
    df_scanning_loactions = df_scanning_loactions.reset_index(drop=True)

    return df_scanning_loactions,df_centroids


def find_valid_regions(dealer_id, product_group_id, query_date, dealer_scope_dict):

    # 转换为 datetime 对象
    query_date = datetime.strptime(query_date, '%Y-%m-%d')

    flag = 0  # if (dealer_id, product_group_id) 在当前经销商合同范围表
    if (dealer_id, product_group_id) in dealer_scope_dict:
        flag = 1 
        df_dealer_scope = dealer_scope_dict[(dealer_id, product_group_id)]
        df_valid_region = df_dealer_scope[(query_date >= df_dealer_scope['EFFECTIVE_DATE']) & (query_date <= df_dealer_scope['INACTIVE_DATE'])]
        
        if not df_valid_region.empty:
            df_scope = df_valid_region[['AREA_CODE', 'AREA_NAME', 'PROVINCE', 'CITY', 'DISTRICT', 'STREET']]
            return df_scope.reset_index(drop=True), flag

        return pd.DataFrame(columns=['AREA_CODE', 'AREA_NAME', 'PROVINCE', 'CITY', 'DISTRICT', 'STREET']), flag
    return pd.DataFrame(columns=['AREA_CODE', 'AREA_NAME', 'PROVINCE', 'CITY', 'DISTRICT', 'STREET']), flag


def is_belong_to (address_list, scope_list):
    # 将高德api 返回的centroid位置格式 转化成 经营范围表的格式
    # ['天津市', '[]'] -> ['天津', '天津市']
    if address_list[0] in ['北京市', '上海市', '天津市', '重庆市'] and address_list[1] == []:
        address_list[1] = address_list[0]
        address_list[0] = address_list[0][:2]

    if address_list[1] == []:
        address_list[1] = address_list[2]
        
    if '-1' in scope_list:
        level = scope_list.index('-1')
    else:
        level = len(scope_list)

    if level == 0:
        return False
    
    for i in range(level):
        if address_list[i] != scope_list[i]:
            return False
    return True


def verify_centroids_within_scope(df_centroids, df_scope):
    df_centroids_with_remote = df_centroids.copy()
    df_centroids_with_remote['is_remote'] = -100
    # -100: 未处理；-2： overall; -1: noise; 
    # 0：本地；1：异地；

    for i in range(len(df_centroids_with_remote)):
        cluster_label = df_centroids_with_remote.loc[i, 'cluster_label']

        if cluster_label != -1:
            address_list = df_centroids_with_remote.loc[i, ['province', 'city', 'district', 'street']].values.flatten().tolist()
            flag = True
            for j in range(len(df_scope)): # if df_scope 是空，不会进此循环， flag 永远是True.
                scope_list = df_scope.loc[j, :].values.flatten().tolist()
                if is_belong_to(address_list, scope_list):
                    flag = False  # 只要在一个有效经营范围内，就不是异地
                    break
                
            if flag:
                df_centroids_with_remote.loc[i, 'is_remote'] = 1

            else:
                df_centroids_with_remote.loc[i, 'is_remote'] = 0
        else:
            df_centroids_with_remote.loc[i, 'is_remote'] = cluster_label
    return df_centroids_with_remote


def verify_points_within_scope(df_scanning_locations, df_scope):

    df_scanning_locations_with_new_remote_label = df_scanning_locations.copy().reset_index(drop=True)
    df_scanning_locations_with_new_remote_label['point_remote_label_new'] = -100
    # -100: 未处理；-2： overall; -1: noise; 
    # 0：本地；1：异地；

    for i in range(len(df_scanning_locations_with_new_remote_label)):
        address_list = df_scanning_locations_with_new_remote_label.loc[i, ['OPEN_PROVINCE', 'OPEN_CITY', 'OPEN_DISTRICT', 'OPEN_TOWN']].values.flatten().tolist()
        # print(address_list)

        # 红包扫码表格式 -> 经营范围表格式
        # ['天津市', '天津市'] -> ['天津', '天津市']

        if address_list[0] in ['天津市', '北京市', '上海市', '重庆市']:
            address_list[0] = address_list[0][:2]

        flag = True
        for j in range(len(df_scope)):
            scope_list = df_scope.loc[j, :].values.flatten().tolist()
            if is_belong_to(address_list, scope_list):
                flag = False
                break
        # print(flag)
        if flag:
            df_scanning_locations_with_new_remote_label.loc[i, 'point_remote_label_new'] = 1
        else:
            df_scanning_locations_with_new_remote_label.loc[i, 'point_remote_label_new'] = 0

    return df_scanning_locations_with_new_remote_label



def find_remote_clusters_for_dealers(df_cleaned, dealer_id, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path, dealer_scope_dict_path, 
                                     end_scope = False):
        with open(dealer_scope_dict_path, 'rb') as f:
            dealer_scope_dict = pickle.load(f)
              
        df_scanning_locations_with_labels, df_centroids = \
        get_scanning_locations_and_centroids(df_cleaned, dealer_id, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path)

        df_scope_start, is_within_archive = find_valid_regions(dealer_id, product_group_id, start_date_str, dealer_scope_dict) # 如果在数据里找不到这条，被记录下来
        df_scope_end, is_within_archive = find_valid_regions(dealer_id, product_group_id, end_date_str, dealer_scope_dict)

        ##########
        df_scope_start = df_scope_start[['PROVINCE', 'CITY', 'DISTRICT', 'STREET']]
        df_scope_end = df_scope_end[['PROVINCE', 'CITY', 'DISTRICT', 'STREET']]
        #########

        dealer_change_remote_scope = {}
        df_centroids_with_remote_label = pd.DataFrame()

        if df_scope_start.equals(df_scope_end):
            df_centroids_with_remote_label = verify_centroids_within_scope(df_centroids, df_scope_end)
            df_scanning_locations_with_remote_labels = verify_points_within_scope(df_scanning_locations_with_labels, df_scope_end)

        else:
            business_scope ={
                    'start' : df_scope_start,
                    'end': df_scope_end
                }

            dealer_change_remote_scope[(dealer_id, product_group_id)] = business_scope
            # print(f'经销商：{dealer_id}的经营范围 在{start_date_str} - {end_date_str} 之间发生了变化！')

            # 起始时间的有效范围 可能只有一个为空，另一个不为空。这时候取不为空的。
            if df_scope_end.empty and not df_scope_start.empty:
                df_centroids_with_remote_label = verify_centroids_within_scope(df_centroids, df_scope_start)
                df_scanning_locations_with_remote_labels = verify_points_within_scope(df_scanning_locations_with_labels, df_scope_start)
                # print(f'{dealer_id} start_date的有效经营范围为空, end_date不为空, 按照end_date范围计算.')

            elif df_scope_start.empty and not df_scope_end.empty:
                df_centroids_with_remote_label = verify_centroids_within_scope(df_centroids, df_scope_end)
                df_scanning_locations_with_remote_labels = verify_points_within_scope(df_scanning_locations_with_labels, df_scope_end)
                # print(f'{dealer_id} end_date的有效经营范围为空, start_date不为空, 按照start_date范围计算.')

            else:
                if end_scope:
                    df_scanning_locations_with_remote_labels = verify_points_within_scope(df_scanning_locations_with_labels, df_scope_end)
                    # print('按照时间区间末端计算')
                    df_centroids_with_remote_label = verify_centroids_within_scope(df_centroids, df_scope_end)   
                else:
                    df_scanning_locations_with_remote_labels = verify_points_within_scope(df_scanning_locations_with_labels, df_scope_start)
                    # print('按照时间区间始端计算')
                    df_centroids_with_remote_label = verify_centroids_within_scope(df_centroids, df_scope_start)

         
       # df_remote_cluster = df_centroids_with_remote_label[df_centroids_with_remote_label.is_remote == 1]
       # print('--------------------------------------------')

        df_centroids_with_remote_label['product_group_id'] = product_group_id
        df_centroids_with_remote_label['total_scanning_count_for_dealer'] = len(df_scanning_locations_with_remote_labels)
        df_centroids_with_remote_label['overall_centroid_for_dealer_latitude'] = df_centroids.loc[df_centroids['cluster_label'] == -2, 'LATITUDE'].iloc[0]
        df_centroids_with_remote_label['overall_centroid_for_dealer_longitude'] = df_centroids.loc[df_centroids['cluster_label'] == -2, 'LONGITUDE'].iloc[0]

        centroid = (df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == -2, 'LATITUDE'].iloc[0],\
              df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == -2, 'LONGITUDE'].iloc[0])

        df_centroids_with_remote_label.loc[:, 'dealer_total_box_count'] = df_scanning_locations_with_remote_labels['BARCODE_CORNER'].nunique()

        for label in df_centroids_with_remote_label.cluster_label:

              df_locations_label = df_scanning_locations_with_remote_labels[df_scanning_locations_with_remote_labels.cluster_label == label].copy()
              cluster_mask = df_centroids_with_remote_label['cluster_label'] == label

              if label not in [-2]:
                     # 每个label内的扫码点数
                     df_centroids_with_remote_label.loc[cluster_mask, 'scanning_count_within_cluster'] = len(df_locations_label)

                     df_centroids_with_remote_label.loc[cluster_mask, 'box_count_within_cluster'] =\
                              df_locations_label['BARCODE_CORNER'].nunique()

              if label not in [-2, -1]:
                     points = list(zip(df_locations_label.LATITUDE, df_locations_label.LONGITUDE))
                     centroid_for_label = (df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == label, 'LATITUDE'].iloc[0],\
                                          df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == label, 'LONGITUDE'].iloc[0])
                     # 到所有点的质心 的距离
                     df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == label, 'dis_to_overall_centroid'] = np.round(geodesic(centroid_for_label, centroid).kilometers, 2)

                     distances = [geodesic(point, centroid_for_label).kilometers for point in points]
                     df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == label, 'avg_distance_within_cluster'] = round(np.mean(distances), 2)
                     df_centroids_with_remote_label.loc[df_centroids_with_remote_label['cluster_label'] == label, 'std_distance_within_cluster'] = round(np.std(distances), 2)

        df_centroids_with_remote_label['is_dealer_within_archive'] = is_within_archive
        df_centroids_with_remote_label['box_count_ratio_for_cluster'] = (df_centroids_with_remote_label['box_count_within_cluster'] / df_centroids_with_remote_label['dealer_total_box_count']).round(2)
        df_scanning_locations_with_remote_labels['is_dealer_within_archive'] = is_within_archive

        # return df_centroids_with_remote_label, df_scanning_locations_with_remote_labels, dealer_change_remote_scope, is_within_archive
        return df_centroids_with_remote_label, df_scanning_locations_with_remote_labels, is_within_archive


def find_hotspots_for_region(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, end_scope = False):
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    ids_within_region = df_cleaned[(df_cleaned['CUST_SCAN_DATE'] >= start_date) & \
                        (df_cleaned['CUST_SCAN_DATE'] <= end_date) & \
                        (df_cleaned['PRODUCT_GROUP_CODE'] == product_group_id)]['BELONG_DEALER_NO'].unique()
    
    centroids_list = []
    scanning_locations_list = []
    # dealers_change_remote_scope = {}
    dealers_not_within_archive = []

    for i, dealer_id in enumerate(ids_within_region):
        # print(f'进度: {i}/{len(ids_within_region)}')
        # df_centroids_with_remote_label, df_scanning_locations_with_remote_labels, dealer_change_remote_scope, is_within_archive = \
        df_centroids_with_remote_label, df_scanning_locations_with_remote_labels, is_within_archive  = \
        find_remote_clusters_for_dealers(df_cleaned, dealer_id, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path, \
                                         dealer_scope_dict_path, end_scope)

        df_centroids_with_remote_label['dealer_id'] = dealer_id
 
        if not df_centroids_with_remote_label.empty:
            centroids_list.append(df_centroids_with_remote_label)

        if not df_scanning_locations_with_remote_labels.empty:
            scanning_locations_list.append(df_scanning_locations_with_remote_labels)

        # if dealer_change_remote_scope:
        #     dealers_change_remote_scope.update(dealer_change_remote_scope)
        
        if not is_within_archive:
            dealers_not_within_archive.append((dealer_id, product_group_id))

    try:
        df_total_centroids = pd.concat(centroids_list).reset_index(drop=True)

    except Exception as e:
        df_total_centroids = pd.DataFrame()  # 创建一个空的 DataFrame
        print(f"发生错误: {e}")


    try:
        df_total_scanning_locations = pd.concat(scanning_locations_list).reset_index(drop=True)
    except Exception as e:
        df_total_scanning_locations = pd.DataFrame()  # 创建一个空的 DataFrame
        print(f"发生错误: {e}")

    # df_hotspots = df_total_centroids.loc[~(df_total_centroids['cluster_label'].isin([-2, -1])), :]
    # df_local_hotpots_within_archive = df_hotspots.loc[
    #     ((df_hotspots['is_dealer_within_archive'] == 1) & (df_hotspots['is_remote'] == 0)), :
    #         ]

    # average_dis_to_overall_centroid_among_all_local_centroids = np.mean(df_local_hotpots_within_archive.dis_to_overall_centroid)
    # average_avg_distance_within_cluster_among_all_local_centroids = np.mean(df_local_hotpots_within_archive.avg_distance_within_cluster)
    # average_std_distance_within_cluster_among_all_local_centroids = np.mean(df_local_hotpots_within_archive.std_distance_within_cluster)

    # df_total_centroids['ratio_dis_to_overall_centroid'] = (df_total_centroids['dis_to_overall_centroid'] / average_dis_to_overall_centroid_among_all_local_centroids).round(2)
    # df_total_centroids['ratio_avg_distance_within_cluster'] = (df_total_centroids['avg_distance_within_cluster'] / average_avg_distance_within_cluster_among_all_local_centroids).round(2)
    # df_total_centroids['ratio_std_distance_within_cluster'] = (df_total_centroids['std_distance_within_cluster'] / average_std_distance_within_cluster_among_all_local_centroids).round(2)
    df_total_centroids['ratio_scanning_count'] = (df_total_centroids['scanning_count_within_cluster'] / df_total_centroids['total_scanning_count_for_dealer']).round(2)

    # return df_total_centroids, df_total_scanning_locations, dealers_change_remote_scope, dealers_not_within_archive
    return df_total_centroids, df_total_scanning_locations, dealers_not_within_archive



def calculate_distances_to_local_centroids_for_centroids(df_total_scanning_locations, df_total_centroids, dealers_not_within_archive):
    ids_within_archive = set(df_total_centroids.dealer_id.unique()) - set([item[0] for item in dealers_not_within_archive])
    centroids_all_local_points_dict = {}
    centroids_all_local_hotspots_dict = {}

    for id in ids_within_archive:

        df_locations = df_total_scanning_locations.loc[df_total_scanning_locations['BELONG_DEALER_NO'] == id, :]
        df_centroids = df_total_centroids.loc[df_total_centroids['dealer_id'] == id, :]

        # 所有为本地的 扫码点 的质心
        df_all_local_points = df_locations.loc[df_total_scanning_locations['point_remote_label_new'] == 0, ['LATITUDE', 'LONGITUDE']]
        if len(df_all_local_points) != 0:
            centroid_all_local_points = df_all_local_points.mean()
            centroids_all_local_points_dict[id] = (centroid_all_local_points['LATITUDE'], centroid_all_local_points['LONGITUDE'])   
        else:
            centroids_all_local_points_dict[id] = np.nan
        
        # 所有在本地热点内的 扫码点 的质心
        local_hotspot_labels = df_centroids.loc[(~df_centroids['cluster_label'].isin([-2, -1])) & (df_centroids['is_remote'] == 0), 'cluster_label'].unique()
        df_all_local_hotspots = df_locations.loc[df_locations['cluster_label'].isin(local_hotspot_labels), ['LATITUDE', 'LONGITUDE']]
        if len(df_all_local_hotspots) != 0:
            centroid_all_local_hotspots = df_all_local_hotspots.mean()
            centroids_all_local_hotspots_dict[id] = (centroid_all_local_hotspots['LATITUDE'], centroid_all_local_hotspots['LONGITUDE'])
        else:
            centroids_all_local_hotspots_dict[id] = np.nan
    # 将质心映射到 df_total_centroids 中
    df_total_centroids['centroid_all_local_points_coordinate'] = df_total_centroids['dealer_id'].map(centroids_all_local_points_dict)
    df_total_centroids['centroid_all_local_hotspots_coordinate'] = df_total_centroids['dealer_id'].map(centroids_all_local_hotspots_dict)

    def calculate_distance_hotspots(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if pd.isna(row["LATITUDE"]) or pd.isna(row["LONGITUDE"]) or pd.isna(row["centroid_all_local_hotspots_coordinate"]) :
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_hotspots_coordinate"]
        ).kilometers
    
    def calculate_distance_points(row):
        # 如果有缺失值，返回 NaN（不进行计算）

        if pd.isna(row["LATITUDE"]) or pd.isna(row["LONGITUDE"]) or pd.isna(row["centroid_all_local_points_coordinate"]) :
            return np.nan  # 或者返回 None, NaN: `return pd.NA`
        # 否则，计算距离
        return geodesic(
            (row["LATITUDE"], row["LONGITUDE"]),
            row["centroid_all_local_points_coordinate"]
        ).kilometers
        
    # 使用 apply 方法计算距离，并保留缺失值的行
    df_total_centroids["dis_to_all_local_hotspots_centroid"] = df_total_centroids.apply(
        calculate_distance_hotspots,
        axis=1
    )
    df_total_centroids["dis_to_all_local_hotspots_centroid"] = np.round(df_total_centroids["dis_to_all_local_hotspots_centroid"], 2)

    df_total_centroids["dis_to_all_local_points_centroid"] = df_total_centroids.apply(
        calculate_distance_points,
        axis=1
    )
    df_total_centroids["dis_to_all_local_points_centroid"] = np.round(df_total_centroids["dis_to_all_local_points_centroid"], 2)

    return df_total_centroids


def find_closest_point_geodesic(fixed_point, coordinates):
    # min_distance = float('inf')
    min_distance = float('99999')
    closest_point = None

    for coord in coordinates:
        coordinate = (coord[0], coord[1])
        distance = geodesic(fixed_point, coordinate).kilometers  # Distance in kilometers
        if distance < min_distance:
            min_distance = distance
            closest_point = coord

    return closest_point, min_distance


def calculate_min_distance_to_border(df_total_centroids, product_group_id, start_date_str, end_date_str, config_file_path, dealer_scope_dict_path):

    with open(dealer_scope_dict_path, 'rb') as f:
        dealer_scope_dict = pickle.load(f)
    
    df_total_hotspots = df_total_centroids.loc[(~(df_total_centroids['cluster_label'].isin([-1, -2]))) & (df_total_centroids['is_dealer_within_archive']==1) , :].reset_index(drop=True)
    # df_total_hotspots['dis_border'] = float('inf')

    for i in range(len(df_total_hotspots)):
        dealer_id = df_total_hotspots.loc[i, 'dealer_id']

        df_valid_scope, _ = find_valid_regions(dealer_id, product_group_id, start_date_str, dealer_scope_dict)

        if df_valid_scope.empty:
            df_valid_scope_end, _ = find_valid_regions(dealer_id, product_group_id, end_date_str, dealer_scope_dict)

            if df_valid_scope_end.empty:
                df_total_hotspots.loc[i, 'dis_border'] = float(('inf'))
                # print(f'{dealer_id} 有效经营范围为空')
                continue
            else:
                df_valid_scope = df_valid_scope_end
    
        # acodes = list(df_valid_scope['AREA_CODE']) # 有一部分公司的area_Code 与高德api的acode 不一致！！！！
        # 添加当前（月初或月末）有效经营范围的区域划分线

        df_valid_scope = df_valid_scope.reset_index(drop=True)
        acodes = []
        for j in range(len(df_valid_scope)):
            address = df_valid_scope.loc[j, ['PROVINCE', 'CITY', 'DISTRICT', 'STREET']].tolist()
            area_name=''
            for item in address:
                if item != '-1':
                    area_name += item
                else:
                    break
            acode = get_acode(config_file_path, area_name, sleep=0.1)['geocodes'][0]['adcode']
            acodes.append(acode)
        
        polyline_points_list_total = []
        if acodes:
            for acode in acodes:
                polyline_points_list = get_polyline_points(config_file_path, acode, sleep=0.1)
                for x in polyline_points_list:
                    polyline_points_list_total.append(x)

        fixed_point = (df_total_hotspots.loc[i, 'LATITUDE'], df_total_hotspots.loc[i,'LONGITUDE'])
        flat_coordinates = [point for sublist in polyline_points_list_total for point in sublist]

        closest_point, min_distance = find_closest_point_geodesic(fixed_point, flat_coordinates)

        df_total_hotspots.loc[i, 'dis_border'] = round(min_distance, 2)

    df_total_hotspots_to_merge = df_total_hotspots.loc[:, ['dealer_id', 'cluster_label', 'dis_border']]
    df_total_centroids_new = pd.merge(df_total_centroids, df_total_hotspots_to_merge, on=['dealer_id', 'cluster_label'], how='left')

    df_total_centroids_new.loc[(df_total_centroids_new['is_remote'] == 0) & (~(df_total_centroids_new['cluster_label'].isin([-1, -2]))), 'dis_border'] = -df_total_centroids_new['dis_border']

    return df_total_centroids_new
        


def main_find_hotspots(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, end_scope = False, cal_border=False):
    
    df_total_centroids, df_total_scanning_locations, dealers_not_within_archive = \
        find_hotspots_for_region(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, end_scope)
    
    df_total_centroids =\
        calculate_distances_to_local_centroids_for_centroids(df_total_scanning_locations, df_total_centroids, dealers_not_within_archive)
    
    if cal_border:
        df_total_centroids =\
            calculate_min_distance_to_border(df_total_centroids, product_group_id, start_date_str, end_date_str, config_file_path, dealer_scope_dict_path)
    
    return df_total_centroids, df_total_scanning_locations
    

def main_find_hotspots_special(df_total_centroids_sparse, df_total_scanning_locations_sparse, large_hotspots_threshold,
                               df_cleaned, product_group_id, start_date_str, end_date_str, radius_dense, min_samples_dense,
                               config_file_path, dealer_scope_dict_path):

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
    set_diff = set(df_large_hotspots_scanning_locations.columns) - set(df_cleaned.columns) 
    df_large_hotspots_scanning_locations_cleaned = df_large_hotspots_scanning_locations.drop(columns=set_diff)


    # dense part
    # radius_dense, min_samples_dense = dbscan_parameters_tuple_dense

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
    df_total_centroids_dense = calculate_min_distance_to_border(df_total_centroids_dense, product_group_id, start_date_str, end_date_str,
                                                                config_file_path, dealer_scope_dict_path)

    return df_total_centroids_dense, df_total_scanning_locations_dense