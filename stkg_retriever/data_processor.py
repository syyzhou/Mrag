# stkg_retriever/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
import pickle

class POIDataProcessor:
    """POI签到数据处理器"""
    
    def __init__(self, time_unit=0.5, distance_unit=100):
        """
        Args:
            time_unit: 时间间隔单位（小时），默认0.5小时
            distance_unit: 距离间隔单位（米），默认100米
        """
        self.time_unit = time_unit
        self.distance_unit = distance_unit
        
        # ID映射
        self.user2id = {}
        self.poi2id = {}
        self.category2id = {}
        self.region2id = {}  # 可以基于经纬度划分区域
        
        # 反向映射
        self.id2user = {}
        self.id2poi = {}
        self.id2category = {}
        
        # POI信息
        self.poi_info = {}  # poi_id -> {category, lat, lon, region}
        
    def load_and_process(self, csv_path: str) -> Dict:
        """
        加载并处理CSV数据
        
        Returns:
            处理后的数据字典
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # 解析时间
        df['timestamp'] = pd.to_datetime(df['UTCTimeOffset'])
        df = df.sort_values(['UserId', 'trajectory_id', 'UTCTimeOffsetEpoch'])
        
        # 构建ID映射
        self._build_id_mappings(df)
        
        # 提取轨迹
        trajectories = self._extract_trajectories(df)
        
        # 划分区域
        self._assign_regions(df)
        
        print(f"Processed {len(self.user2id)} users, {len(self.poi2id)} POIs, "
              f"{len(self.category2id)} categories, {len(trajectories)} trajectories")
        
        return {
            'trajectories': trajectories,
            'user2id': self.user2id,
            'poi2id': self.poi2id,
            'category2id': self.category2id,
            'region2id': self.region2id,
            'poi_info': self.poi_info
        }
    
    def _build_id_mappings(self, df: pd.DataFrame):
        """构建实体ID映射"""
        # 用户映射
        for user_id in df['UserId'].unique():
            if user_id not in self.user2id:
                idx = len(self.user2id)
                self.user2id[user_id] = idx
                self.id2user[idx] = user_id
        
        # POI映射
        for _, row in df.drop_duplicates('PoiId').iterrows():
            poi_id = row['PoiId']
            if poi_id not in self.poi2id:
                idx = len(self.poi2id)
                self.poi2id[poi_id] = idx
                self.id2poi[idx] = poi_id
                
                # 存储POI信息
                self.poi_info[idx] = {
                    'original_id': poi_id,
                    'lat': row['Latitude'],
                    'lon': row['Longitude'],
                    'category_id': row['PoiCategoryId'],
                    'category_name': row['PoiCategoryName']
                }
        
        # 类别映射
        for cat_id in df['PoiCategoryId'].unique():
            if cat_id not in self.category2id:
                idx = len(self.category2id)
                self.category2id[cat_id] = idx
                self.id2category[idx] = cat_id
    
    def _assign_regions(self, df: pd.DataFrame, grid_size=0.01):
        """基于经纬度网格划分区域"""
        for poi_idx, info in self.poi_info.items():
            lat_grid = int(info['lat'] / grid_size)
            lon_grid = int(info['lon'] / grid_size)
            region_key = f"{lat_grid}_{lon_grid}"
            
            if region_key not in self.region2id:
                self.region2id[region_key] = len(self.region2id)
            
            info['region_id'] = self.region2id[region_key]
    
    def _extract_trajectories(self, df: pd.DataFrame) -> List[Dict]:
        """提取用户轨迹"""
        trajectories = []
        
        for traj_id, group in df.groupby('trajectory_id'):
            group = group.sort_values('UTCTimeOffsetEpoch')
            user_id = group['UserId'].iloc[0]
            
            checkins = []
            for _, row in group.iterrows():
                checkins.append({
                    'poi_id': self.poi2id[row['PoiId']],
                    'timestamp': row['timestamp'],
                    'epoch': row['UTCTimeOffsetEpoch'],
                    'lat': row['Latitude'],
                    'lon': row['Longitude'],
                    'category_id': self.category2id[row['PoiCategoryId']]
                })
            
            trajectories.append({
                'trajectory_id': traj_id,
                'user_id': self.user2id[user_id],
                'checkins': checkins
            })
        
        return trajectories
    
    def save(self, save_path: str):
        """保存处理结果"""
        data = {
            'user2id': self.user2id,
            'poi2id': self.poi2id,
            'category2id': self.category2id,
            'region2id': self.region2id,
            'poi_info': self.poi_info,
            'id2user': self.id2user,
            'id2poi': self.id2poi,
            'id2category': self.id2category
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processor to {save_path}")
    
    @classmethod
    def load(cls, load_path: str):
        """加载处理器"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        processor = cls()
        processor.user2id = data['user2id']
        processor.poi2id = data['poi2id']
        processor.category2id = data['category2id']
        processor.region2id = data['region2id']
        processor.poi_info = data['poi_info']
        processor.id2user = data['id2user']
        processor.id2poi = data['id2poi']
        processor.id2category = data['id2category']
        
        return processor