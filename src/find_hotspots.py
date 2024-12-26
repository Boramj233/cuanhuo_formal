from datetime import datetime
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
import requests
import yaml

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


def get_address_from_lat_lon(location, config_file_path):  

    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get('gaode_api_key')

    url = 'https://restapi.amap.com/v3/geocode/regeo?parameters'
    params = {
        'key': gaode_api_key,
        'location':  location, # 经度在前，纬度在后 （lon, lat)
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # 检查请求是否成功
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    

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



def plot_clusters_with_folium(df_scanning_locations, points_size = 5, noise_size=5):

    gdf = gpd.GeoDataFrame(df_scanning_locations, geometry=gpd.points_from_xy(df_scanning_locations.LONGITUDE, df_scanning_locations.LATITUDE)) # x -> longitude ; y -> latitude

    m = folium.Map(location=[gdf['LATITUDE'].mean(), gdf['LONGITUDE'].mean()], 
                    tiles= 'https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=7',
                    attr='高德-常规图',
                    zoom_start=7,
                )

    unique_clusters = gdf['cluster_label'].sort_values().unique()

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#aab7ff', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
        '#f7b6d2', '#dbdb8d', '#9edae5', '#f5b041',  '#d62728'
    ]

    for i, cluster in enumerate(unique_clusters):
        if cluster != -1:  # 排除噪声点
            cluster_points = gdf[gdf['cluster_label'] == cluster]

            color_index = cluster % 20  
            color = colors[color_index]
            for _, row in cluster_points.iterrows():
                
                folium.CircleMarker(
                    location=(row['LATITUDE'], row['LONGITUDE']),
                    radius=points_size,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.6,
                    popup=f'Cluster: {cluster}'
                ).add_to(m)

    # 绘制噪声点（可选）
    noise_points = gdf[gdf['cluster_label'] == -1]
    for _, row in noise_points.iterrows():
        folium.CircleMarker(
            location=(row['LATITUDE'], row['LONGITUDE']),
            radius=noise_size,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup='Noise'
        ).add_to(m)

    legend_html = '''
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
    '''
    for i, cluster in enumerate(unique_clusters):
        if cluster != -1:
            color_index = cluster % 20
            color = colors[color_index]
            legend_html += f'<i style="background:{color}; width: 15px; height: 15px; display:inline-block; margin-right: 5px;"></i> Cluster {cluster}<br>'
    legend_html += '</div>'

    # 将图例添加到地图
    folium.Marker(
        location=[gdf['LATITUDE'].mean(), gdf['LONGITUDE'].mean()],
        icon=folium.DivIcon(html=legend_html)
    ).add_to(m)

    return m


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


def main_find_hotspots(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, end_scope = False):
    
    df_total_centroids, df_total_scanning_locations, dealers_not_within_archive = \
        find_hotspots_for_region(df_cleaned, product_group_id, start_date_str, end_date_str, radius, min_samples, config_file_path,
                                    dealer_scope_dict_path, end_scope)
    
    df_total_centroids =\
        calculate_distances_to_local_centroids_for_centroids(df_total_scanning_locations, df_total_centroids, dealers_not_within_archive)
    
    return df_total_centroids, df_total_scanning_locations
    