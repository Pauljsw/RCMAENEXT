import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, optimize
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ sklearn not available. Using simplified material analysis.")
    SKLEARN_AVAILABLE = False
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

class RealDataIntensityAnalyzer:
    """
    Real Velodyne 데이터에서 intensity 패턴을 분석하고 
    Synthetic 데이터에 적용할 수 있는 모델을 생성
    """
    
    def __init__(self, real_data_path: str):
        self.real_data_path = Path(real_data_path)
        self.intensity_model = None
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
        
        # 분석 결과 저장
        self.analysis_results = {
            'distance_intensity_curve': None,
            'height_intensity_map': None,
            'angular_intensity_map': None,
            'material_signatures': None,
            'noise_characteristics': None
        }
    
    def load_real_pointcloud(self, file_path: Path) -> np.ndarray:
        """Real Velodyne 데이터 로드 (x,y,z,intensity)"""
        try:
            # .bin 파일 (Velodyne 형식)
            if file_path.suffix == '.bin':
                points = np.fromfile(file_path, dtype=np.float32)
                if len(points) % 4 == 0:
                    return points.reshape(-1, 4)
                else:
                    print(f"Warning: {file_path} has unexpected format")
                    return None
            
            # .npy 파일
            elif file_path.suffix == '.npy':
                points = np.load(file_path)
                if points.shape[1] == 4:  # x,y,z,intensity
                    return points.astype(np.float32)
                elif points.shape[1] == 3:  # x,y,z only
                    print(f"Warning: {file_path} has no intensity, skipping")
                    return None
                else:
                    print(f"Warning: {file_path} has unexpected shape {points.shape}")
                    return None
            
            # .pcd 파일
            elif file_path.suffix == '.pcd':
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    
                # 헤더 건너뛰기
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('DATA'):
                        data_start = i + 1
                        break
                
                # 데이터 파싱
                points = []
                for line in lines[data_start:]:
                    if line.strip():
                        vals = line.strip().split()
                        if len(vals) >= 4:
                            points.append([float(v) for v in vals[:4]])
                
                return np.array(points, dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def analyze_distance_intensity_relationship(self, points_list: List[np.ndarray]):
        """거리-intensity 관계 분석"""
        print("Analyzing distance-intensity relationship...")
        
        all_distances = []
        all_intensities = []
        
        for points in points_list:
            if points is not None and points.shape[1] >= 4:
                distances = np.linalg.norm(points[:, :3], axis=1)
                intensities = points[:, 3]
                
                # 기본 필터링: intensity > 1 (거의 0값만 제거)
                valid_mask = (distances > 1.0) & (distances < 60.0) & (intensities > 1.0)
                
                valid_distances = distances[valid_mask]
                valid_intensities = intensities[valid_mask]
                
                print(f"  File points: {len(valid_intensities)}/{len(intensities)} "
                      f"(mean intensity: {np.mean(valid_intensities):.1f})")
                
                all_distances.extend(valid_distances)
                all_intensities.extend(valid_intensities)
        
        all_distances = np.array(all_distances)
        all_intensities = np.array(all_intensities)
        
        print(f"Total valid points: {len(all_distances)}")
        print(f"Distance range: {all_distances.min():.1f} - {all_distances.max():.1f}m")
        print(f"Intensity stats: mean={np.mean(all_intensities):.1f}, "
              f"std={np.std(all_intensities):.1f}")
        
        # 거리 구간별 통계 계산 (적응적 구간)
        max_dist = min(all_distances.max(), 50.0)  # 최대 50m로 제한
        distance_bins = np.linspace(1, max_dist, 20)  # 20개 구간
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        mean_intensities = []
        std_intensities = []
        point_counts = []
        
        for i in range(len(distance_bins) - 1):
            mask = (all_distances >= distance_bins[i]) & (all_distances < distance_bins[i+1])
            count = np.sum(mask)
            point_counts.append(count)
            
            if count > 50:  # 충분한 포인트가 있는 경우
                intensities_in_bin = all_intensities[mask]
                mean_intensities.append(np.mean(intensities_in_bin))
                std_intensities.append(np.std(intensities_in_bin))
            else:
                mean_intensities.append(np.nan)
                std_intensities.append(np.nan)
        
        # NaN 제거
        valid_mask = ~np.isnan(mean_intensities)
        valid_distances = bin_centers[valid_mask]
        valid_means = np.array(mean_intensities)[valid_mask]
        valid_stds = np.array(std_intensities)[valid_mask]
        
        print(f"Valid distance bins: {len(valid_distances)}/20")
        print(f"Distance bins with data: {[f'{d:.1f}m({c}pts)' for d, c in zip(bin_centers, point_counts) if c > 50]}")
        
        if len(valid_distances) >= 3:  # 최소 3개 구간 필요
            try:
                # 1/R^2 + C 모델 (더 단순한)
                def inverse_square_model(r, a, c):
                    return a / (r ** 2) + c
                
                # 더 안정적인 피팅
                popt, pcov = optimize.curve_fit(
                    inverse_square_model, 
                    valid_distances, 
                    valid_means,
                    p0=[1000, 20],  # 단순한 초기값
                    bounds=([10, 5], [10000, 100]),  # 합리적인 경계
                    maxfev=5000
                )
                
                # 피팅 품질 확인
                fitted_y = inverse_square_model(valid_distances, *popt)
                residuals = valid_means - fitted_y
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((valid_means - np.mean(valid_means)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                print(f"Fitting: a={popt[0]:.1f}, c={popt[1]:.1f}, R²={r_squared:.3f}")
                
                # 결과 저장
                fitted_distances = np.linspace(1, max_dist, 100)
                fitted_intensities = inverse_square_model(fitted_distances, *popt)
                
                # 전체 배열에 NaN을 실제값으로 채움
                final_means = np.array(mean_intensities, dtype=float)
                final_stds = np.array(std_intensities, dtype=float)
                final_p25 = np.full_like(final_means, np.nan)
                final_p75 = np.full_like(final_means, np.nan)
                
                self.analysis_results['distance_intensity_curve'] = {
                    'bin_centers': bin_centers,
                    'mean_intensities': final_means.tolist(),
                    'std_intensities': final_stds.tolist(), 
                    'percentile_25': final_p25.tolist(),
                    'percentile_75': final_p75.tolist(),
                    'fitted_curve': (fitted_distances, fitted_intensities),
                    'fit_params': popt,
                    'fit_equation': f"I = {popt[0]:.1f}/R² + {popt[1]:.1f}",
                    'r_squared': r_squared,
                    'model_type': 'inverse_square'
                }
                
                print(f"✅ Distance-Intensity fitting: {self.analysis_results['distance_intensity_curve']['fit_equation']}")
                print(f"   Quality: R² = {r_squared:.3f}")
                
                return self.analysis_results['distance_intensity_curve']
                
            except Exception as e:
                print(f"❌ Curve fitting failed: {e}")
                # Fallback으로 선형 보간 저장
                if len(valid_distances) >= 2:
                    print("   Using linear interpolation as fallback")
                    self.analysis_results['distance_intensity_curve'] = {
                        'bin_centers': bin_centers,
                        'mean_intensities': mean_intensities,
                        'std_intensities': std_intensities,
                        'valid_distances': valid_distances,
                        'valid_means': valid_means,
                        'model_type': 'interpolation'
                    }
                    return self.analysis_results['distance_intensity_curve']
        else:
            print(f"❌ Too few valid distance bins ({len(valid_distances)}) for analysis")
        
        return None
    
    def analyze_height_intensity_relationship(self, points_list: List[np.ndarray]):
        """높이-intensity 관계 분석 (건설장비의 부위별 특성)"""
        print("Analyzing height-intensity relationship...")
        
        all_heights = []
        all_intensities = []
        all_distances = []
        
        for points in points_list:
            if points is not None and points.shape[1] >= 4:
                heights = points[:, 2]  # z 좌표
                intensities = points[:, 3]
                distances = np.linalg.norm(points[:, :3], axis=1)
                
                # 지면 위 객체만 (z > 0.5m, 거리 제한)
                valid_mask = (heights > 0.5) & (heights < 8.0) & (distances > 2) & (distances < 50) & (intensities > 0)
                
                all_heights.extend(heights[valid_mask])
                all_intensities.extend(intensities[valid_mask])
                all_distances.extend(distances[valid_mask])
        
        all_heights = np.array(all_heights)
        all_intensities = np.array(all_intensities)
        all_distances = np.array(all_distances)
        
        # 높이 구간별 분석
        height_bins = np.linspace(0.5, 8.0, 16)  # 0.5m 간격
        height_centers = (height_bins[:-1] + height_bins[1:]) / 2
        
        height_intensity_stats = []
        
        for i in range(len(height_bins) - 1):
            mask = (all_heights >= height_bins[i]) & (all_heights < height_bins[i+1])
            if np.sum(mask) > 20:
                height_intensities = all_intensities[mask]
                height_distances = all_distances[mask]
                
                # 거리 정규화된 intensity (거리 효과 제거)
                # 가정: I_normalized = I_observed * (distance/reference_distance)^2
                reference_distance = 20.0
                normalized_intensities = height_intensities * (height_distances / reference_distance) ** 2
                
                stats = {
                    'height_center': height_centers[i],
                    'mean_intensity': np.mean(normalized_intensities),
                    'std_intensity': np.std(normalized_intensities),
                    'median_intensity': np.median(normalized_intensities),
                    'count': np.sum(mask)
                }
                height_intensity_stats.append(stats)
        
        self.analysis_results['height_intensity_map'] = height_intensity_stats
        return height_intensity_stats
    
    def analyze_angular_intensity_variation(self, points_list: List[np.ndarray]):
        """각도별 intensity 변화 분석 (sensor 중심에서의 방향별)"""
        print("Analyzing angular intensity variation...")
        
        all_angles = []
        all_intensities = []
        all_distances = []
        
        for points in points_list:
            if points is not None and points.shape[1] >= 4:
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
                intensities = points[:, 3]
                distances = np.linalg.norm(points[:, :3], axis=1)
                
                # 방위각 계산 (0-360도)
                angles = np.arctan2(y, x) * 180 / np.pi
                angles = (angles + 360) % 360  # 0-360 범위로 정규화
                
                # 유효한 데이터만
                valid_mask = (distances > 5) & (distances < 40) & (intensities > 0) & (z > 0)
                
                all_angles.extend(angles[valid_mask])
                all_intensities.extend(intensities[valid_mask])
                all_distances.extend(distances[valid_mask])
        
        all_angles = np.array(all_angles)
        all_intensities = np.array(all_intensities)
        all_distances = np.array(all_distances)
        
        # 각도 구간별 분석 (10도 간격)
        angle_bins = np.arange(0, 370, 10)
        angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        
        angular_stats = []
        
        for i in range(len(angle_bins) - 1):
            mask = (all_angles >= angle_bins[i]) & (all_angles < angle_bins[i+1])
            if np.sum(mask) > 10:
                angle_intensities = all_intensities[mask]
                angle_distances = all_distances[mask]
                
                # 거리 정규화
                reference_distance = 20.0
                normalized_intensities = angle_intensities * (angle_distances / reference_distance) ** 2
                
                stats = {
                    'angle_center': angle_centers[i],
                    'mean_intensity': np.mean(normalized_intensities),
                    'std_intensity': np.std(normalized_intensities),
                    'count': np.sum(mask)
                }
                angular_stats.append(stats)
        
        self.analysis_results['angular_intensity_map'] = angular_stats
        return angular_stats
    
    def analyze_material_signatures(self, points_list: List[np.ndarray]):
        """재질별 intensity 시그니처 분석 (클러스터링 기반)"""
        print("Analyzing material signatures...")
        
        if not SKLEARN_AVAILABLE:
            print("  Sklearn not available, using simplified material analysis...")
            # 간단한 높이 기반 재질 분류
            material_signatures = []
            height_ranges = [(0.5, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 8.0)]
            
            all_heights = []
            all_intensities = []
            
            for points in points_list:
                if points is not None and points.shape[1] >= 4:
                    heights = points[:, 2]
                    intensities = points[:, 3]
                    distances = np.linalg.norm(points[:, :3], axis=1)
                    
                    valid_mask = (distances > 2) & (distances < 60) & (intensities > 0) & (heights > 0)
                    all_heights.extend(heights[valid_mask])
                    all_intensities.extend(intensities[valid_mask])
            
            if all_heights:
                all_heights = np.array(all_heights)
                all_intensities = np.array(all_intensities)
                
                for i, (min_h, max_h) in enumerate(height_ranges):
                    mask = (all_heights >= min_h) & (all_heights < max_h)
                    if np.sum(mask) > 100:
                        signature = {
                            'cluster_id': i,
                            'count': np.sum(mask),
                            'height_range': (min_h, max_h),
                            'height_mean': np.mean(all_heights[mask]),
                            'intensity_mean': np.mean(all_intensities[mask]),
                            'intensity_std': np.std(all_intensities[mask]),
                            'intensity_range': (np.min(all_intensities[mask]), np.max(all_intensities[mask])),
                            'center': np.array([np.mean(all_heights[mask]), np.mean(all_intensities[mask])])
                        }
                        material_signatures.append(signature)
            
            self.analysis_results['material_signatures'] = material_signatures
            return material_signatures
        
        # sklearn이 있는 경우 원래 로직
        all_features = []
        all_intensities = []
        
        for points in points_list:
            if points is not None and points.shape[1] >= 4:
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
                intensities = points[:, 3]
                distances = np.linalg.norm(points[:, :3], axis=1)
                
                # 특징 벡터 생성 [height, distance, normalized_intensity]
                valid_mask = (distances > 2) & (distances < 60) & (intensities > 0) & (z > 0)
                
                heights = z[valid_mask]
                dists = distances[valid_mask]
                intens = intensities[valid_mask]
                
                # Intensity 정규화 (거리 효과 제거)
                normalized_intens = intens * (dists / 20.0) ** 2
                
                features = np.column_stack([heights, dists, normalized_intens])
                all_features.append(features)
                all_intensities.extend(normalized_intens)
        
        if all_features:
            all_features = np.vstack(all_features)
            all_intensities = np.array(all_intensities)
            
            # 샘플링 (너무 많으면 메모리 문제)
            if len(all_features) > 50000:
                indices = np.random.choice(len(all_features), 50000, replace=False)
                all_features = all_features[indices]
                all_intensities = all_intensities[indices]
            
            # K-means 클러스터링으로 재질 그룹 찾기
            n_clusters = 8  # 건설장비의 주요 재질 수
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            # 높이와 정규화된 intensity만 사용
            cluster_features = all_features[:, [0, 2]]  # [height, normalized_intensity]
            cluster_labels = kmeans.fit_predict(cluster_features)
            
            # 클러스터별 특성 분석
            material_signatures = []
            for i in range(n_clusters):
                mask = cluster_labels == i
                if np.sum(mask) > 100:  # 충분한 샘플이 있는 클러스터만
                    cluster_heights = all_features[mask, 0]
                    cluster_intensities = all_intensities[mask]
                    
                    signature = {
                        'cluster_id': i,
                        'count': np.sum(mask),
                        'height_range': (np.min(cluster_heights), np.max(cluster_heights)),
                        'height_mean': np.mean(cluster_heights),
                        'intensity_mean': np.mean(cluster_intensities),
                        'intensity_std': np.std(cluster_intensities),
                        'intensity_range': (np.min(cluster_intensities), np.max(cluster_intensities)),
                        'center': kmeans.cluster_centers_[i]
                    }
                    material_signatures.append(signature)
            
            self.analysis_results['material_signatures'] = material_signatures
            return material_signatures
        
        return None
    
    def analyze_noise_characteristics(self, points_list: List[np.ndarray]):
        """Intensity 노이즈 특성 분석"""
        print("Analyzing noise characteristics...")
        
        # 동일한 거리의 유사한 표면에서 intensity 변동성 분석
        intensity_variations = []
        
        for points in points_list:
            if points is not None and points.shape[1] >= 4:
                distances = np.linalg.norm(points[:, :3], axis=1)
                intensities = points[:, 3]
                heights = points[:, 2]
                
                # 거리별로 그룹화 (2m 간격)
                for dist_center in range(5, 60, 5):
                    mask = (distances >= dist_center - 1) & (distances < dist_center + 1) & \
                           (heights > 0.5) & (heights < 2.0) & (intensities > 0)  # 비슷한 높이의 표면
                    
                    if np.sum(mask) > 20:
                        group_intensities = intensities[mask]
                        
                        # 변동 계수 (CV = std/mean) 계산
                        cv = np.std(group_intensities) / np.mean(group_intensities)
                        
                        intensity_variations.append({
                            'distance': dist_center,
                            'mean_intensity': np.mean(group_intensities),
                            'std_intensity': np.std(group_intensities),
                            'cv': cv,
                            'count': np.sum(mask)
                        })
        
        if intensity_variations:
            # 전체 노이즈 특성
            all_cvs = [v['cv'] for v in intensity_variations if v['count'] > 50]
            noise_characteristics = {
                'variations_by_distance': intensity_variations,
                'average_cv': np.mean(all_cvs) if all_cvs else 0.1,
                'cv_std': np.std(all_cvs) if all_cvs else 0.05
            }
            
            self.analysis_results['noise_characteristics'] = noise_characteristics
            return noise_characteristics
        
        return None
    
    def run_full_analysis(self, max_files: int = 20):
        """전체 분석 실행"""
        print(f"Starting comprehensive analysis of real Velodyne data...")
        print(f"Data path: {self.real_data_path}")
        
        # Real data 파일들 로드 - 모든 확장자 확인
        real_files = (list(self.real_data_path.glob("*.bin")) + 
                     list(self.real_data_path.glob("*.pcd")) + 
                     list(self.real_data_path.glob("*.npy")))
        
        print(f"Found files:")
        print(f"  .bin files: {len(list(self.real_data_path.glob('*.bin')))}")
        print(f"  .pcd files: {len(list(self.real_data_path.glob('*.pcd')))}")
        print(f"  .npy files: {len(list(self.real_data_path.glob('*.npy')))}")
        print(f"  Total: {len(real_files)}")
        
        if not real_files:
            print("No data files found!")
            print("Please check if files exist and have correct extensions (.bin, .pcd, .npy)")
            return None
        
        print(f"Found {len(real_files)} files, using {min(max_files, len(real_files))} for analysis")
        
        # 파일들을 샘플링해서 로드
        selected_files = real_files[:max_files] if len(real_files) > max_files else real_files
        
        points_list = []
        for file_path in selected_files:
            print(f"Loading {file_path.name}...")
            points = self.load_real_pointcloud(file_path)
            if points is not None:
                points_list.append(points)
                print(f"  Loaded: {points.shape} points")
            else:
                print(f"  Failed to load: {file_path.name}")
        
        print(f"Successfully loaded {len(points_list)} point clouds")
        
        if not points_list:
            print("Failed to load any valid point clouds!")
            return None
        
        # 각종 분석 실행
        self.analyze_distance_intensity_relationship(points_list)
        self.analyze_height_intensity_relationship(points_list)
        self.analyze_angular_intensity_variation(points_list)
        self.analyze_material_signatures(points_list)
        self.analyze_noise_characteristics(points_list)
        
        return self.analysis_results
    
    def visualize_analysis_results(self):
        """분석 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distance-Intensity 관계
        if self.analysis_results['distance_intensity_curve']:
            data = self.analysis_results['distance_intensity_curve']
            axes[0,0].plot(data['bin_centers'], data['mean_intensities'], 'bo-', label='Observed')
            if 'fitted_curve' in data:
                axes[0,0].plot(data['fitted_curve'][0], data['fitted_curve'][1], 'r-', 
                              label=data['fit_equation'])
            axes[0,0].set_xlabel('Distance (m)')
            axes[0,0].set_ylabel('Mean Intensity')
            axes[0,0].set_title('Distance vs Intensity')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # 2. Height-Intensity 관계
        if self.analysis_results['height_intensity_map']:
            data = self.analysis_results['height_intensity_map']
            heights = [d['height_center'] for d in data]
            intensities = [d['mean_intensity'] for d in data]
            axes[0,1].plot(heights, intensities, 'go-')
            axes[0,1].set_xlabel('Height (m)')
            axes[0,1].set_ylabel('Mean Normalized Intensity')
            axes[0,1].set_title('Height vs Intensity')
            axes[0,1].grid(True)
        
        # 3. Angular variation
        if self.analysis_results['angular_intensity_map']:
            data = self.analysis_results['angular_intensity_map']
            angles = [d['angle_center'] for d in data]
            intensities = [d['mean_intensity'] for d in data]
            axes[0,2].plot(angles, intensities, 'mo-')
            axes[0,2].set_xlabel('Angle (degrees)')
            axes[0,2].set_ylabel('Mean Normalized Intensity')
            axes[0,2].set_title('Angular Intensity Variation')
            axes[0,2].grid(True)
        
        # 4. Material signatures
        if self.analysis_results['material_signatures']:
            data = self.analysis_results['material_signatures']
            heights = [d['height_mean'] for d in data]
            intensities = [d['intensity_mean'] for d in data]
            counts = [d['count'] for d in data]
            scatter = axes[1,0].scatter(heights, intensities, s=[c/100 for c in counts], 
                                      c=range(len(data)), cmap='tab10', alpha=0.7)
            axes[1,0].set_xlabel('Mean Height (m)')
            axes[1,0].set_ylabel('Mean Intensity')
            axes[1,0].set_title('Material Clusters')
            plt.colorbar(scatter, ax=axes[1,0])
        
        # 5. Noise characteristics
        if self.analysis_results['noise_characteristics']:
            data = self.analysis_results['noise_characteristics']['variations_by_distance']
            distances = [d['distance'] for d in data]
            cvs = [d['cv'] for d in data]
            axes[1,1].plot(distances, cvs, 'co-')
            axes[1,1].set_xlabel('Distance (m)')
            axes[1,1].set_ylabel('Coefficient of Variation')
            axes[1,1].set_title('Noise vs Distance')
            axes[1,1].grid(True)
        
        # 6. Summary statistics
        axes[1,2].axis('off')
        summary_text = "Analysis Summary:\n\n"
        
        if self.analysis_results['distance_intensity_curve']:
            summary_text += f"Distance relationship:\n{self.analysis_results['distance_intensity_curve']['fit_equation']}\n\n"
        
        if self.analysis_results['material_signatures']:
            n_materials = len(self.analysis_results['material_signatures'])
            summary_text += f"Material clusters: {n_materials}\n\n"
        
        if self.analysis_results['noise_characteristics']:
            avg_cv = self.analysis_results['noise_characteristics']['average_cv']
            summary_text += f"Average noise (CV): {avg_cv:.3f}\n\n"
        
        axes[1,2].text(0.1, 0.9, summary_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def save_analysis_model(self, output_path: str):
        """분석 결과를 파일로 저장"""
        with open(output_path, 'wb') as f:
            pickle.dump(self.analysis_results, f)
        print(f"Analysis results saved to {output_path}")
    
    def load_analysis_model(self, input_path: str):
        """저장된 분석 결과 로드"""
        with open(input_path, 'rb') as f:
            self.analysis_results = pickle.load(f)
        print(f"Analysis results loaded from {input_path}")

# 사용 예시
def test_real_data_analysis():
    """테스트 함수"""
    # Real data 경로 설정
    real_data_path = "/home/sungwoo/VoxelNeXt/data/real_data"
    
    # 분석기 초기화
    analyzer = RealDataIntensityAnalyzer(real_data_path)
    
    # 전체 분석 실행
    results = analyzer.run_full_analysis(max_files=10)
    
    if results:
        # 결과 시각화
        analyzer.visualize_analysis_results()
        
        # 모델 저장
        analyzer.save_analysis_model("real_intensity_analysis.pkl")
        
        print("Analysis completed successfully!")
    else:
        print("Analysis failed!")

if __name__ == "__main__":
    test_real_data_analysis()