import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SimpleRealisticIntensityGenerator:
    """
    Real data의 통계적 특성을 파악해서 
    Synthetic data에 합리적인 intensity를 생성
    """
    
    def __init__(self, real_data_path: str = None):
        self.real_data_path = Path(real_data_path) if real_data_path else None
        self.real_stats = None
        
    def analyze_real_data_statistics(self, max_files: int = 20):
        """Real data의 기본 통계만 파악"""
        if not self.real_data_path or not self.real_data_path.exists():
            print("⚠️ Real data path not found, using default statistics")
            return self._get_default_stats()
        
        print(f"📊 Analyzing real data statistics from {self.real_data_path}")
        
        # Real files 찾기
        real_files = (list(self.real_data_path.glob("*.bin")) + 
                     list(self.real_data_path.glob("*.pcd")) + 
                     list(self.real_data_path.glob("*.npy")))
        
        if not real_files:
            print("⚠️ No real data files found, using default statistics")
            return self._get_default_stats()
        
        print(f"Found {len(real_files)} files, analyzing {min(max_files, len(real_files))} files")
        
        all_intensities = []
        all_distances = []
        all_heights = []
        
        for i, file_path in enumerate(real_files[:max_files]):
            try:
                # 파일 로드
                if file_path.suffix == '.npy':
                    points = np.load(file_path)
                elif file_path.suffix == '.bin':
                    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
                else:
                    continue
                
                if points.shape[1] < 4:
                    continue
                
                # 기본 필터링
                distances = np.linalg.norm(points[:, :3], axis=1)
                intensities = points[:, 3]
                heights = points[:, 2]
                
                valid_mask = (distances > 1.0) & (distances < 80.0) & (intensities > 0)
                
                all_intensities.extend(intensities[valid_mask])
                all_distances.extend(distances[valid_mask])
                all_heights.extend(heights[valid_mask])
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed {i+1}/{min(max_files, len(real_files))} files")
                    
            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
        
        if not all_intensities:
            print("❌ Failed to load any valid data, using default statistics")
            return self._get_default_stats()
        
        # 통계 계산
        all_intensities = np.array(all_intensities)
        all_distances = np.array(all_distances)
        all_heights = np.array(all_heights)
        
        # 기본 통계
        intensity_stats = {
            'mean': np.mean(all_intensities),
            'std': np.std(all_intensities),
            'min': np.min(all_intensities),
            'max': np.max(all_intensities),
            'median': np.median(all_intensities),
            'p25': np.percentile(all_intensities, 25),
            'p75': np.percentile(all_intensities, 75)
        }
        
        # 거리별 intensity (대략적)
        distance_groups = []
        for dist_range in [(1, 10), (10, 20), (20, 40), (40, 80)]:
            mask = (all_distances >= dist_range[0]) & (all_distances < dist_range[1])
            if np.sum(mask) > 100:
                group_intensities = all_intensities[mask]
                distance_groups.append({
                    'distance_range': dist_range,
                    'mean_intensity': np.mean(group_intensities),
                    'std_intensity': np.std(group_intensities),
                    'count': np.sum(mask)
                })
        
        # 높이별 intensity (대략적)
        height_groups = []
        for height_range in [(0, 1), (1, 2), (2, 4), (4, 10)]:
            mask = (all_heights >= height_range[0]) & (all_heights < height_range[1])
            if np.sum(mask) > 100:
                group_intensities = all_intensities[mask]
                height_groups.append({
                    'height_range': height_range,
                    'mean_intensity': np.mean(group_intensities),
                    'std_intensity': np.std(group_intensities),
                    'count': np.sum(mask)
                })
        
        self.real_stats = {
            'total_points': len(all_intensities),
            'intensity': intensity_stats,
            'distance_groups': distance_groups,
            'height_groups': height_groups,
            'distance_range': (np.min(all_distances), np.max(all_distances)),
            'height_range': (np.min(all_heights), np.max(all_heights))
        }
        
        print(f"✅ Real data analysis completed:")
        print(f"   📊 Intensity: mean={intensity_stats['mean']:.1f}, std={intensity_stats['std']:.1f}")
        print(f"   📏 Distance groups: {len(distance_groups)}")
        print(f"   📐 Height groups: {len(height_groups)}")
        
        return self.real_stats
    
    def _get_default_stats(self):
        """Real data가 없을 때 기본 통계"""
        return {
            'intensity': {
                'mean': 35.0, 'std': 25.0, 'min': 1.0, 'max': 200.0,
                'median': 30.0, 'p25': 15.0, 'p75': 50.0
            },
            'distance_groups': [
                {'distance_range': (1, 10), 'mean_intensity': 45.0, 'std_intensity': 20.0},
                {'distance_range': (10, 20), 'mean_intensity': 35.0, 'std_intensity': 18.0},
                {'distance_range': (20, 40), 'mean_intensity': 28.0, 'std_intensity': 15.0},
                {'distance_range': (40, 80), 'mean_intensity': 22.0, 'std_intensity': 12.0}
            ],
            'height_groups': [
                {'height_range': (0, 1), 'mean_intensity': 25.0, 'std_intensity': 15.0},    # 바퀴/하부
                {'height_range': (1, 2), 'mean_intensity': 40.0, 'std_intensity': 20.0},   # 차체
                {'height_range': (2, 4), 'mean_intensity': 35.0, 'std_intensity': 18.0},   # 상부
                {'height_range': (4, 10), 'mean_intensity': 30.0, 'std_intensity': 16.0}   # 캐빈 등
            ]
        }
    
    def generate_synthetic_intensity(self, synthetic_points: np.ndarray) -> np.ndarray:
        """Real data 특성을 기반으로 synthetic intensity 생성"""
        
        if self.real_stats is None:
            print("⚠️ No real data stats available, using defaults")
            self.real_stats = self._get_default_stats()
        
        n_points = synthetic_points.shape[0]
        distances = np.linalg.norm(synthetic_points, axis=1)
        heights = synthetic_points[:, 2]
        
        print(f"🎯 Generating intensity for {n_points} points based on real data patterns")
        
        # 1. 거리별 기본 intensity
        base_intensities = np.zeros(n_points)
        
        for group in self.real_stats['distance_groups']:
            dist_min, dist_max = group['distance_range']
            mask = (distances >= dist_min) & (distances < dist_max)
            
            if np.sum(mask) > 0:
                # 해당 거리 범위의 평균값 기반으로 생성
                group_mean = group['mean_intensity']
                group_std = group.get('std_intensity', 15.0)
                
                # 약간의 랜덤성 추가
                base_intensities[mask] = np.random.normal(
                    group_mean, group_std * 0.3, np.sum(mask)
                )
        
        # 거리 그룹에 해당하지 않는 점들 처리
        unassigned_mask = base_intensities == 0
        if np.sum(unassigned_mask) > 0:
            # 전체 평균값 사용
            overall_mean = self.real_stats['intensity']['mean']
            overall_std = self.real_stats['intensity']['std']
            base_intensities[unassigned_mask] = np.random.normal(
                overall_mean, overall_std * 0.5, np.sum(unassigned_mask)
            )
        
        print(f"   Distance-based intensity: {base_intensities.min():.1f} - {base_intensities.max():.1f}")
        
        # 2. 높이별 조정
        height_adjusted = base_intensities.copy()
        
        for group in self.real_stats['height_groups']:
            height_min, height_max = group['height_range']
            mask = (heights >= height_min) & (heights < height_max)
            
            if np.sum(mask) > 0:
                # 높이별 특성을 비율로 적용
                current_mean = np.mean(base_intensities[mask])
                target_mean = group['mean_intensity']
                
                if current_mean > 0:
                    adjustment_factor = target_mean / current_mean
                    adjustment_factor = np.clip(adjustment_factor, 0.7, 1.4)  # 극단적 변화 방지
                    height_adjusted[mask] *= adjustment_factor
        
        print(f"   Height-adjusted intensity: {height_adjusted.min():.1f} - {height_adjusted.max():.1f}")
        
        # 3. 약간의 공간적 변화 (현실적)
        x, y = synthetic_points[:, 0], synthetic_points[:, 1]
        angles = np.arctan2(y, x)
        
        # 방향별 약간의 변화 (센서 특성 모사)
        angular_factor = 1.0 + 0.15 * np.sin(angles * 2)  # ±15% 변화
        spatial_adjusted = height_adjusted * angular_factor
        
        print(f"   Spatially-adjusted intensity: {spatial_adjusted.min():.1f} - {spatial_adjusted.max():.1f}")
        
        # 4. Real data와 유사한 노이즈 특성
        real_std_ratio = self.real_stats['intensity']['std'] / self.real_stats['intensity']['mean']
        noise_std = spatial_adjusted * real_std_ratio * 0.3  # 30% 적용
        noise_std = np.clip(noise_std, 1.0, 10.0)  # 합리적 범위
        
        noise = np.random.normal(0, noise_std)
        final_intensities = spatial_adjusted + noise
        
        # 5. Real data 범위에 맞춰 조정
        real_min = max(1.0, self.real_stats['intensity']['min'])
        real_max = min(255.0, self.real_stats['intensity']['max'])
        
        final_clipped = np.clip(final_intensities, real_min, real_max)
        
        print(f"   Final intensity: {final_clipped.min():.1f} - {final_clipped.max():.1f}, mean: {final_clipped.mean():.1f}")
        
        # 6. Real data와 통계 비교
        real_mean = self.real_stats['intensity']['mean']
        synthetic_mean = np.mean(final_clipped)
        print(f"   📊 Real vs Synthetic: {real_mean:.1f} vs {synthetic_mean:.1f} (diff: {abs(real_mean-synthetic_mean):.1f})")
        
        return final_clipped.astype(np.uint8)
    
    def save_stats(self, output_path: str):
        """분석된 통계 저장"""
        if self.real_stats:
            with open(output_path, 'wb') as f:
                pickle.dump(self.real_stats, f)
            print(f"💾 Real data statistics saved to {output_path}")
    
    def load_stats(self, input_path: str):
        """저장된 통계 로드"""
        with open(input_path, 'rb') as f:
            self.real_stats = pickle.load(f)
        print(f"📂 Real data statistics loaded from {input_path}")

# 통합 프로세서
class IntensityProcessor:
    """전체 프로세스 관리"""
    
    def __init__(self, real_data_path: str):
        self.generator = SimpleRealisticIntensityGenerator(real_data_path)
    
    def process_synthetic_dataset(self, 
                                synthetic_data_path: str,
                                output_path: str):
        """전체 synthetic dataset 처리"""
        
        print("🚀 Starting Intensity Processing Pipeline")
        print("="*50)
        
        # 1. Real data 분석
        print("📊 STEP 1: Analyzing Real Data")
        stats = self.generator.analyze_real_data_statistics(max_files=1246)
        
        if not stats:
            print("❌ Failed to analyze real data")
            return False
        
        # 2. Synthetic files 찾기
        print(f"\n🔧 STEP 2: Processing Synthetic Data")
        synthetic_path = Path(synthetic_data_path)
        synthetic_files = (list(synthetic_path.glob("*.bin")) + 
                          list(synthetic_path.glob("*.npy")) + 
                          list(synthetic_path.glob("*.pcd")))
        
        if not synthetic_files:
            print(f"❌ No synthetic files found in {synthetic_data_path}")
            return False
        
        print(f"Found {len(synthetic_files)} synthetic files")
        
        # 3. 출력 디렉토리 생성
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # 4. 각 파일 처리
        success_count = 0
        for i, file_path in enumerate(synthetic_files):
            try:
                print(f"Processing ({i+1}/{len(synthetic_files)}): {file_path.name}")
                
                # 파일 로드
                if file_path.suffix == '.npy':
                    points = np.load(file_path)
                elif file_path.suffix == '.bin':
                    points = np.fromfile(file_path, dtype=np.float32)
                    if len(points) % 3 == 0:
                        points = points.reshape(-1, 3)
                    else:
                        print(f"  ⚠️ Unexpected format: {file_path.name}")
                        continue
                else:
                    continue
                
                if points.shape[1] != 3:
                    print(f"  ⚠️ Expected 3 columns, got {points.shape[1]}: {file_path.name}")
                    continue
                
                # Intensity 생성
                intensities = self.generator.generate_synthetic_intensity(points)
                
                # x,y,z,intensity 결합
                enhanced_points = np.column_stack([points, intensities])
                
                # NPY로 저장
                output_file = output_dir / f"{file_path.stem}.npy"
                np.save(output_file, enhanced_points.astype(np.float32))
                
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path.name}: {e}")
        
        print(f"\n✅ Processing completed!")
        print(f"   Successfully processed: {success_count}/{len(synthetic_files)} files")
        print(f"   Output directory: {output_path}")
        
        # 통계 저장
        self.generator.save_stats(output_dir / "real_data_stats.pkl")
        
        return success_count > 0

# 메인 실행 함수
def main():
    """간단한 실행 함수"""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python script.py <real_data_path> <synthetic_data_path> <output_path>")
        print("Example: python script.py /data/real /data/synthetic /data/enhanced")
        return
    
    real_data_path = sys.argv[1]
    synthetic_data_path = sys.argv[2] 
    output_path = sys.argv[3]
    
    processor = IntensityProcessor(real_data_path)
    success = processor.process_synthetic_dataset(synthetic_data_path, output_path)
    
    if success:
        print("🎉 All done! Enhanced synthetic data is ready for R-MAE training.")
    else:
        print("❌ Processing failed!")

if __name__ == "__main__":
    main()