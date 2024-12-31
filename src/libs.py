import requests
import time
import yaml



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

   
def get_acode(config_file_path, area_name, sleep=0):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get('gaode_api_key')

    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'

    params = {
        'key': gaode_api_key,
        'address': area_name,
        'city': area_name,
    }

    try: 
        if sleep:
            time.sleep(sleep)
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None



def get_region_polyline(config_file_path, acode, sleep=0):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    gaode_api_key = config.get('gaode_api_key')

    url = 'https://restapi.amap.com/v3/config/district?parameters'
    params = {
        'key': gaode_api_key,
        'keywords': acode,
        'subdistrict': 0,
        'filter': acode,
        'extensions': 'all',
    }

    try: 
        if sleep:
            time.sleep(sleep)
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None
    

    
def get_polyline_points(config_file_path, acode, sleep=0):
    if len(acode) > 6:
        acode = acode[:6]
    data = get_region_polyline(config_file_path, acode)
    polyline = data['districts'][0]['polyline']
    polyline_points_list = []
    # if '|' in polyline:
    polylines = polyline.split('|')

    for polyline in polylines:
        polyline = polyline.split(';')
        polyline_points = []
        for i, coordinate in enumerate(polyline):
            coordinate = coordinate.split(',')
            coordinate[0], coordinate[1] = float(coordinate[1]), float(coordinate[0]) # folium (lat, lon)

            # polyline_points.append((coordinate[0], coordinate[1]))
            polyline_points.append(coordinate)
        polyline_points_list.append(polyline_points)
    return polyline_points_list