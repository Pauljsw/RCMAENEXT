import numpy as np
import pickle
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class LearnedIntensityGenerator:
    """
    Real data에서 학습한 패턴을 사용하여 
    Synthetic data에 현실적인 intensity를 생성
    """
    
    def __init__(self, analysis_model_path: str = None):
        self.analysis_results = None
        self.distance_intensity_func = None
        self.height_intensity_func = None
        self.angular_intensity_func = None
        self.material_classifier = None
        
        if analysis_model_path:
            self.load_learned_patterns(analysis_model_path)
    
    def load_learned_patterns(self, model_path: str):
        """학습된 패턴 로드"""
        with open(model_path, 'rb') as f:
            self.analysis_results = pickle.load(f)
        
        print("Loading learned intensity patterns...")
        self._build_interpolation_functions()
        self._build_material_classifier()
        print("Learned patterns loaded successfully!")
    
    def _build_interpolation_functions(self):
        """분석 결과로부터 보간 함수들 생성"""
        
        # 1. Distance-Intensity 함수
        if self.analysis_results['distance_intensity_curve']:
            data = self.analysis_results['distance_intensity_curve']
            
            if 'fit_params' in data and len(data['fit_params']) == 3:
                # Power law 함수 사용
                a, b, c = data['fit_params']
                self.distance_intensity_func = lambda r: a / (r ** b) + c
                print(f"Distance function: I = {a:.1f}/R^{b:.2f} + {c:.1f}")
            else:
                # 선형 보간 사용
                distances = np.array(data['bin_centers'])
                intensities = np.array(data['mean_intensities'])
                valid_mask = intensities > 0
                
                if np.sum(valid_mask) > 2:
                    self.distance_intensity_func = interpolate.interp1d(
                        distances[valid_mask], intensities[valid_mask],
                        kind='cubic', bounds_error=False, fill_value='extrapolate'
                    )
        
        # 2. Height-Intensity 함수
        if self.analysis_results['height_intensity_map']:
            data = self.analysis_results['height_intensity_map']
            heights = np.array([d['height_center'] for d in data])
            intensities = np.array([d['mean_intensity'] for d in data])
            
            if len(heights) > 2:
                self.height_intensity_func = interpolate.interp1d(
                    heights, intensities, kind='linear', 
                    bounds_error=False, fill_value='extrapolate'
                )
        
        # 3. Angular-Intensity 함수
        if self.analysis_results['angular_intensity_map']:
            data = self.analysis_results['angular_intensity_map']
            angles = np.array([d['angle_center'] for d in data])
            intensities = np.array([d['mean_intensity'] for d in data])
            
            if len(angles) > 2:
                # 원형 보간을 위해 양 끝에 데이터 추가
                angles_extended = np.concatenate([angles - 360, angles, angles + 360])
                intensities_extended = np.concatenate([intensities, intensities, intensities])
                
                self.angular_intensity_func = interpolate.interp1d(
                    angles_extended, intensities_extended, kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
    
    def _build_material_classifier(self):
        """재질 분류기 구성"""
        if self.analysis_results['material_signatures']:
            # 각 클러스터의 중심점과 특성을 사용하여 분류기 구성
            signatures = self.analysis_results['material_signatures']
            
            # 클러스터 중심점들 (height, intensity)
            cluster_centers = np.array([sig['center'] for sig in signatures])
            
            # 각 클러스터의 intensity 특성
            self.material_intensity_map = {}
            for i, sig in enumerate(signatures):
                self.material_intensity_map[i] = {
                    'mean_intensity': sig['intensity_mean'],
                    'std_intensity': sig['intensity_std'],
                    'height_range': sig['height_range'],
                    'count': sig['count']
                }
            
            # KNN을 사용한 재질 분류기
            if len(cluster_centers) > 0:
                self.material_classifier = NearestNeighbors(n_neighbors=1)
                self.material_classifier.fit(cluster_centers)
    
    def predict_material_type(self, heights: np.ndarray, base_intensities: np.ndarray) -> np.ndarray:
        """높이와 기본 intensity를 바탕으로 재질 타입 예측"""
        if self.material_classifier is None:
            return np.zeros(len(heights), dtype=int)
        
        # 특징 벡터 구성 [height, normalized_intensity]
        features = np.column_stack([heights, base_intensities])
        
        # 가장 가까운 클러스터 찾기
        distances, indices = self.material_classifier.kneighbors(features)
        
        return indices.flatten()
    
    def apply_distance_effect(self, distances: np.ndarray) -> np.ndarray:
        """거리 효과 적용"""
        if self.distance_intensity_func is None:
            # 기본 거리 감쇠 (1/R^2)
            return 100.0 / (distances ** 2) + 10.0
        
        # 학습된 함수 사용
        try:
            intensities = self.distance_intensity_func(distances)
            
            # 비정상적인 결과 체크 (모든 값이 같거나 너무 작으면)
            if np.std(intensities) < 1.0 or np.mean(intensities) < 5.0:
                print("  Warning: Distance function seems invalid, using fallback")
                return 100.0 / (distances ** 2) + 10.0
            
            # 음수 방지
            return np.maximum(intensities, 1.0)
            
        except Exception as e:
            print(f"  Warning: Distance function failed ({e}), using fallback")
            return 100.0 / (distances ** 2) + 10.0
    
    def apply_height_effect(self, heights: np.ndarray, base_intensities: np.ndarray) -> np.ndarray:
        """높이 효과 적용 (건설장비의 부위별 특성)"""
        if self.height_intensity_func is None:
            # 기본: 높이에 따른 약간의 변화
            height_factor = 1.0 + 0.1 * np.sin(heights * np.pi / 4.0)
            return base_intensities * height_factor
        
        # 학습된 높이별 intensity 특성 적용
        height_intensities = self.height_intensity_func(heights)
        
        # 기본 intensity와 조합 (가중 평균)
        alpha = 0.3  # 높이 효과의 가중치
        combined = (1 - alpha) * base_intensities + alpha * height_intensities
        
        return np.maximum(combined, 1.0)
    
    def apply_angular_effect(self, points: np.ndarray, base_intensities: np.ndarray) -> np.ndarray:
        """각도 효과 적용"""
        if self.angular_intensity_func is None:
            return base_intensities
        
        # 방위각 계산
        angles = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
        angles = (angles + 360) % 360  # 0-360 범위
        
        # 학습된 각도별 intensity 변화 적용
        angular_factors = self.angular_intensity_func(angles)
        
        # 정규화 (평균값을 1로)
        if len(angular_factors) > 0:
            mean_factor = np.mean(angular_factors)
            if mean_factor > 0:
                angular_factors = angular_factors / mean_factor
        
        return base_intensities * angular_factors
    
    def apply_material_effect(self, heights: np.ndarray, 
                            base_intensities: np.ndarray,
                            object_classes: Optional[np.ndarray] = None) -> np.ndarray:
        """재질 효과 적용"""
        if self.material_classifier is None:
            return base_intensities
        
        # 재질 타입 예측
        material_types = self.predict_material_type(heights, base_intensities)
        
        # 각 재질별 intensity 조정
        adjusted_intensities = base_intensities.copy()
        
        for material_id in np.unique(material_types):
            if material_id in self.material_intensity_map:
                mask = material_types == material_id
                material_info = self.material_intensity_map[material_id]
                
                # 해당 재질의 평균 intensity에 맞춰 조정
                current_mean = np.mean(adjusted_intensities[mask])
                target_mean = material_info['mean_intensity']
                
                if current_mean > 0:
                    scale_factor = target_mean / current_mean
                    adjusted_intensities[mask] *= scale_factor
                
                # 재질별 노이즈 추가
                material_noise = np.random.normal(
                    0, material_info['std_intensity'] * 0.2, np.sum(mask)
                )
                adjusted_intensities[mask] += material_noise
        
        return np.maximum(adjusted_intensities, 1.0)
    
    def apply_noise_effect(self, intensities: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """학습된 노이즈 특성 적용"""
        if self.analysis_results['noise_characteristics'] is None:
            # 기본 노이즈 (안전한 버전)
            noise_level = 0.03
            noise_std = np.maximum(noise_level * intensities, 1.0)  # 최소 1.0 보장
            noise = np.random.normal(0, noise_std)
            return np.maximum(intensities + noise, 1.0)
        
        # 거리별 노이즈 특성 적용
        noise_data = self.analysis_results['noise_characteristics']['variations_by_distance']
        
        noisy_intensities = intensities.copy()
        
        for noise_info in noise_data:
            dist_center = noise_info['distance']
            cv = max(noise_info['cv'], 0.01)  # 최소값 보장
            
            # 해당 거리 범위의 점들
            mask = (distances >= dist_center - 2.5) & (distances < dist_center + 2.5)
            
            if np.sum(mask) > 0:
                # CV 기반 노이즈 추가 (안전한 버전)
                base_intensities = noisy_intensities[mask]
                noise_std = np.maximum(cv * base_intensities, 0.5)  # 최소 0.5 보장
                noise = np.random.normal(0, noise_std)
                noisy_intensities[mask] += noise
        
        return np.maximum(noisy_intensities, 1.0)
    
    def generate_realistic_intensity(self, 
                                   synthetic_points: np.ndarray,
                                   object_classes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        메인 함수: Synthetic points에 realistic intensity 생성
        
        Args:
            synthetic_points: (N, 3) [x, y, z] coordinates
            object_classes: (N,) object class labels (optional)
            
        Returns:
            intensities: (N,) realistic intensity values
        """
        n_points = synthetic_points.shape[0]
        
        # 기본 계산
        distances = np.linalg.norm(synthetic_points, axis=1)
        heights = synthetic_points[:, 2]
        
        print(f"Generating intensity for {n_points} synthetic points...")
        
        # 1. 거리 기반 기본 intensity
        base_intensities = self.apply_distance_effect(distances)
        print(f"Applied distance effect. Range: {base_intensities.min():.1f} - {base_intensities.max():.1f}")
        
        # 2. 높이 효과 적용
        height_adjusted = self.apply_height_effect(heights, base_intensities)
        print(f"Applied height effect. Range: {height_adjusted.min():.1f} - {height_adjusted.max():.1f}")
        
        # 3. 각도 효과 적용
        angular_adjusted = self.apply_angular_effect(synthetic_points, height_adjusted)
        print(f"Applied angular effect. Range: {angular_adjusted.min():.1f} - {angular_adjusted.max():.1f}")
        
        # 4. 재질 효과 적용
        material_adjusted = self.apply_material_effect(heights, angular_adjusted, object_classes)
        print(f"Applied material effect. Range: {material_adjusted.min():.1f} - {material_adjusted.max():.1f}")
        
        # 5. 노이즈 효과 적용
        final_intensities = self.apply_noise_effect(material_adjusted, distances)
        print(f"Applied noise effect. Range: {final_intensities.min():.1f} - {final_intensities.max():.1f}")
        
        # 6. 최종 범위 조정 (0-255)
        final_intensities = np.clip(final_intensities, 0, 255)
        
        return final_intensities.astype(np.uint8)
    
    def validate_generated_intensity(self, 
                                   synthetic_points: np.ndarray,
                                   generated_intensities: np.ndarray) -> Dict:
        """생성된 intensity의 품질 검증"""
        
        distances = np.linalg.norm(synthetic_points, axis=1)
        heights = synthetic_points[:, 2]
        
        validation_results = {}
        
        # 1. 거리-intensity 관계 검증
        if self.analysis_results['distance_intensity_curve']:
            real_curve = self.analysis_results['distance_intensity_curve']
            
            # 거리별 평균 intensity 계산
            distance_bins = np.linspace(5, 60, 12)
            synthetic_means = []
            real_means = []
            
            for i in range(len(distance_bins) - 1):
                mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
                if np.sum(mask) > 10:
                    synthetic_means.append(np.mean(generated_intensities[mask]))
                    
                    # Real data의 해당 거리 intensity
                    bin_center = (distance_bins[i] + distance_bins[i+1]) / 2
                    if self.distance_intensity_func:
                        real_means.append(self.distance_intensity_func(bin_center))
                    else:
                        real_means.append(50)  # 기본값
            
            if synthetic_means and real_means:
                correlation = np.corrcoef(synthetic_means, real_means)[0, 1]
                validation_results['distance_correlation'] = correlation
        
        # 2. 전체 통계 비교
        validation_results['intensity_stats'] = {
            'mean': np.mean(generated_intensities),
            'std': np.std(generated_intensities),
            'min': np.min(generated_intensities),
            'max': np.max(generated_intensities),
            'range': np.max(generated_intensities) - np.min(generated_intensities)
        }
        
        # 3. 높이별 분포 비교
        height_bins = np.linspace(0.5, 6, 6)
        height_intensity_correlation = []
        
        for i in range(len(height_bins) - 1):
            mask = (heights >= height_bins[i]) & (heights < height_bins[i+1])
            if np.sum(mask) > 5:
                height_intensity_correlation.append(np.mean(generated_intensities[mask]))
        
        validation_results['height_intensity_pattern'] = height_intensity_correlation
        
        return validation_results
    
    def plot_validation_results(self, 
                              synthetic_points: np.ndarray,
                              generated_intensities: np.ndarray):
        """검증 결과 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        distances = np.linalg.norm(synthetic_points, axis=1)
        heights = synthetic_points[:, 2]
        
        # 1. Distance vs Intensity (Real vs Synthetic 비교)
        if self.analysis_results['distance_intensity_curve']:
            real_data = self.analysis_results['distance_intensity_curve']
            
            # Real data 플롯
            axes[0,0].plot(real_data['bin_centers'], real_data['mean_intensities'], 
                          'bo-', label='Real Data', alpha=0.7)
            
            # Synthetic data 플롯
            distance_bins = np.linspace(5, 60, 20)
            synthetic_means = []
            bin_centers = []
            
            for i in range(len(distance_bins) - 1):
                mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
                if np.sum(mask) > 5:
                    synthetic_means.append(np.mean(generated_intensities[mask]))
                    bin_centers.append((distance_bins[i] + distance_bins[i+1]) / 2)
            
            axes[0,0].plot(bin_centers, synthetic_means, 'ro-', label='Synthetic Data', alpha=0.7)
            axes[0,0].set_xlabel('Distance (m)')
            axes[0,0].set_ylabel('Mean Intensity')
            axes[0,0].set_title('Distance vs Intensity Comparison')
            axes[0,0].legend()
            axes[0,0].grid(True)
        
        # 2. Height vs Intensity
        axes[0,1].scatter(heights, generated_intensities, alpha=0.3, s=1)
        axes[0,1].set_xlabel('Height (m)')
        axes[0,1].set_ylabel('Intensity')
        axes[0,1].set_title('Height vs Intensity (Synthetic)')
        axes[0,1].grid(True)
        
        # 3. Intensity 분포
        axes[1,0].hist(generated_intensities, bins=50, alpha=0.7, density=True)
        axes[1,0].set_xlabel('Intensity')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Intensity Distribution (Synthetic)')
        axes[1,0].grid(True)
        
        # 4. 3D 산점도 (거리, 높이, intensity)
        scatter = axes[1,1].scatter(distances, heights, c=generated_intensities, 
                                   alpha=0.5, s=1, cmap='viridis')
        axes[1,1].set_xlabel('Distance (m)')
        axes[1,1].set_ylabel('Height (m)')
        axes[1,1].set_title('Distance-Height-Intensity')
        plt.colorbar(scatter, ax=axes[1,1], label='Intensity')
        
        plt.tight_layout()
        plt.show()

# 통합 클래스: 전체 프로세스 관리
class SyntheticIntensityPipeline:
    """Real data 분석부터 Synthetic data 적용까지의 전체 파이프라인"""
    
    def __init__(self, real_data_path: str):
        self.real_data_path = real_data_path
        self.analyzer = None
        self.generator = None
    
    def run_analysis_phase(self, max_files: int = 15):
        """Phase 1: Real data 분석"""
        print("="*50)
        print("PHASE 1: ANALYZING REAL VELODYNE DATA")
        print("="*50)
        
        from real_data_intensity_analyzer import RealDataIntensityAnalyzer
        
        self.analyzer = RealDataIntensityAnalyzer(self.real_data_path)
        results = self.analyzer.run_full_analysis(max_files=max_files)
        
        if results:
            self.analyzer.visualize_analysis_results()
            
            # 모델 저장
            model_path = "learned_intensity_model.pkl"
            self.analyzer.save_analysis_model(model_path)
            print(f"\nAnalysis model saved to: {model_path}")
            
            return model_path
        else:
            print("Failed to analyze real data!")
            return None
    
    def run_generation_phase(self, model_path: str, 
                           synthetic_points: np.ndarray,
                           object_classes: Optional[np.ndarray] = None):
        """Phase 2: Synthetic data에 intensity 생성"""
        print("="*50)
        print("PHASE 2: GENERATING SYNTHETIC INTENSITY")
        print("="*50)
        
        self.generator = LearnedIntensityGenerator(model_path)
        
        # Intensity 생성
        generated_intensities = self.generator.generate_realistic_intensity(
            synthetic_points, object_classes
        )
        
        # 검증
        validation_results = self.generator.validate_generated_intensity(
            synthetic_points, generated_intensities
        )
        
        print("\nValidation Results:")
        for key, value in validation_results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.3f}")
            else:
                print(f"{key}: {value:.3f}")
        
        # 시각화
        self.generator.plot_validation_results(synthetic_points, generated_intensities)
        
        return generated_intensities
    
    def process_synthetic_dataset(self, 
                                synthetic_data_path: str,
                                output_path: str,
                                model_path: str = None):
        """전체 synthetic dataset 처리"""
        
        if model_path is None:
            model_path = self.run_analysis_phase()
            if model_path is None:
                return False
        
        print("="*50)
        print("PHASE 3: PROCESSING SYNTHETIC DATASET")
        print("="*50)
        
        synthetic_files = list(Path(synthetic_data_path).glob("*.bin"))
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        self.generator = LearnedIntensityGenerator(model_path)
        
        for i, file_path in enumerate(synthetic_files):
            print(f"Processing {file_path.name} ({i+1}/{len(synthetic_files)})")
            
            # Synthetic data 로드
            try:
                points = np.fromfile(file_path, dtype=np.float32)
                if len(points) % 3 == 0:
                    points = points.reshape(-1, 3)
                else:
                    print(f"Skipping {file_path.name}: unexpected format")
                    continue
                
                # Intensity 생성
                intensities = self.generator.generate_realistic_intensity(points)
                
                # x,y,z,intensity 형태로 결합
                enhanced_points = np.column_stack([points, intensities])
                
                # 저장
                output_file = output_dir / file_path.name
                enhanced_points.astype(np.float32).tofile(output_file)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        
        print(f"Dataset processing completed! Output: {output_path}")
        return True

# 사용 예시
def main():
    # 파이프라인 초기화
    pipeline = SyntheticIntensityPipeline("/home/sungwoo/VoxelNeXt/data/real_data")
    
    # 전체 프로세스 실행
    success = pipeline.process_synthetic_dataset(
        synthetic_data_path="/home/sungwoo/VoxelNeXt/data/synthetic_gazebo",
        output_path="/home/sungwoo/VoxelNeXt/data/enhanced_synthetic",
        model_path=None  # None이면 자동으로 분석부터 시작
    )
    
    if success:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed!")

if __name__ == "__main__":
    main()