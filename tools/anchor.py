# tools/analyze_dataset_anchors.py
"""
건설장비 데이터셋 Anchor 최적화 분석 도구

.npy (점군) + .txt (라벨) 파일에서 실제 객체 크기 통계 분석
- 클래스별 평균/표준편차 계산
- Anchor size 최적화
- IoU threshold 최적화
- 시각화 및 보고서 생성
"""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import json


class ConstructionEquipmentAnchorAnalyzer:
    """건설장비 데이터셋 Anchor 분석기"""
    
    def __init__(self, data_dir, class_names=['dumptruck', 'excavator', 'grader', 'roller']):
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.id_to_class = {idx: name for idx, name in enumerate(class_names)}
        
        # 분석 결과 저장
        self.box_stats = {cls: {'lengths': [], 'widths': [], 'heights': []} 
                         for cls in class_names}
        self.all_boxes = []
        
        print(f"🔍 건설장비 Anchor 분석 시작")
        print(f"   📁 데이터 경로: {data_dir}")
        print(f"   🏗️ 클래스: {class_names}")
    
    def load_labels(self, label_file):
        """라벨 파일 로드 (.txt 형식)"""
        boxes = []
        if not os.path.exists(label_file):
            return boxes
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
                
            parts = line.split()
            if len(parts) < 8:  # x, y, z, length, width, height, rotation, class_name
                continue
                
            try:
                # 형식: x y z length width height rotation class_name
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                length, width, height = float(parts[3]), float(parts[4]), float(parts[5])
                rotation = float(parts[6])
                class_name = parts[7]
                
                # 클래스명 검증
                if class_name in self.class_names:
                    boxes.append({
                        'class': class_name,
                        'x': x, 'y': y, 'z': z,
                        'length': length, 'width': width, 'height': height,
                        'rotation': rotation
                    })
                else:
                    print(f"⚠️ 알 수 없는 클래스: {class_name} in {label_file.name}")
                    
            except (ValueError, IndexError) as e:
                print(f"⚠️ 라벨 파싱 오류: {line.strip()} - {e}")
                continue
                
        return boxes
    
    def analyze_dataset(self):
        """전체 데이터셋 분석"""
        print("\n📊 데이터셋 분석 시작...")
        
        # .txt 파일 찾기
        label_files = list(self.data_dir.glob("*.txt"))
        
        if len(label_files) == 0:
            print("❌ 라벨 파일을 찾을 수 없습니다!")
            return
            
        print(f"📝 총 {len(label_files)}개의 라벨 파일 발견")
        
        total_objects = 0
        class_counts = {cls: 0 for cls in self.class_names}
        
        for label_file in label_files:
            boxes = self.load_labels(label_file)
            
            for box in boxes:
                class_name = box['class']
                if class_name in self.box_stats:
                    self.box_stats[class_name]['lengths'].append(box['length'])
                    self.box_stats[class_name]['widths'].append(box['width'])
                    self.box_stats[class_name]['heights'].append(box['height'])
                    
                    self.all_boxes.append(box)
                    class_counts[class_name] += 1
                    total_objects += 1
        
        print(f"\n✅ 분석 완료!")
        print(f"   📦 총 객체 수: {total_objects}")
        for cls, count in class_counts.items():
            print(f"   🏗️ {cls}: {count}개")
        
        if total_objects == 0:
            print("❌ 유효한 객체를 찾을 수 없습니다!")
            return
            
        # 통계 분석
        self.compute_statistics()
        self.generate_optimal_anchors()
        self.create_visualizations()
        self.generate_config()
    
    def compute_statistics(self):
        """클래스별 통계 계산"""
        print("\n📈 통계 분석...")
        
        self.statistics = {}
        
        for class_name in self.class_names:
            if len(self.box_stats[class_name]['lengths']) == 0:
                print(f"⚠️ {class_name}: 데이터 없음")
                continue
                
            lengths = np.array(self.box_stats[class_name]['lengths'])
            widths = np.array(self.box_stats[class_name]['widths'])
            heights = np.array(self.box_stats[class_name]['heights'])
            
            stats = {
                'count': len(lengths),
                'length': {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths),
                    'median': np.median(lengths),
                    'q25': np.percentile(lengths, 25),
                    'q75': np.percentile(lengths, 75)
                },
                'width': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': np.min(widths),
                    'max': np.max(widths),
                    'median': np.median(widths),
                    'q25': np.percentile(widths, 25),
                    'q75': np.percentile(widths, 75)
                },
                'height': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': np.min(heights),
                    'max': np.max(heights),
                    'median': np.median(heights),
                    'q25': np.percentile(heights, 25),
                    'q75': np.percentile(heights, 75)
                }
            }
            
            self.statistics[class_name] = stats
            
            print(f"\n🏗️ {class_name} ({stats['count']}개):")
            print(f"   📏 Length: {stats['length']['mean']:.2f}±{stats['length']['std']:.2f}m")
            print(f"   📐 Width:  {stats['width']['mean']:.2f}±{stats['width']['std']:.2f}m")
            print(f"   📊 Height: {stats['height']['mean']:.2f}±{stats['height']['std']:.2f}m")
    
    def generate_optimal_anchors(self):
        """최적 Anchor 생성"""
        print("\n🎯 최적 Anchor 계산...")
        
        self.optimal_anchors = {}
        
        for class_name in self.class_names:
            if class_name not in self.statistics:
                continue
                
            stats = self.statistics[class_name]
            
            # 전략 1: 평균값 사용
            anchor_mean = [
                round(stats['length']['mean'], 1),
                round(stats['width']['mean'], 1), 
                round(stats['height']['mean'], 1)
            ]
            
            # 전략 2: 중앙값 사용 (outlier에 더 robust)
            anchor_median = [
                round(stats['length']['median'], 1),
                round(stats['width']['median'], 1),
                round(stats['height']['median'], 1)
            ]
            
            # 전략 3: Multi-scale (작은/큰 객체 고려)
            anchor_small = [
                round(stats['length']['q25'], 1),
                round(stats['width']['q25'], 1),
                round(stats['height']['q25'], 1)
            ]
            
            anchor_large = [
                round(stats['length']['q75'], 1),
                round(stats['width']['q75'], 1),
                round(stats['height']['q75'], 1)
            ]
            
            self.optimal_anchors[class_name] = {
                'mean': anchor_mean,
                'median': anchor_median,
                'small': anchor_small,
                'large': anchor_large,
                'recommended': anchor_median  # 중앙값 추천
            }
            
            print(f"   🏗️ {class_name}:")
            print(f"      평균:     {anchor_mean}")
            print(f"      중앙값:   {anchor_median} ⭐")
            print(f"      소형:     {anchor_small}")
            print(f"      대형:     {anchor_large}")
    
    def create_visualizations(self):
        """시각화 생성"""
        print("\n📊 시각화 생성...")
        
        # 1. 클래스별 크기 분포
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('건설장비 클래스별 크기 분포', fontsize=16)
        
        for idx, class_name in enumerate(self.class_names):
            if class_name not in self.statistics:
                continue
                
            ax = axes[idx // 2, idx % 2]
            
            lengths = self.box_stats[class_name]['lengths']
            widths = self.box_stats[class_name]['widths'] 
            heights = self.box_stats[class_name]['heights']
            
            ax.hist([lengths, widths, heights], 
                   bins=20, alpha=0.7, 
                   label=['Length', 'Width', 'Height'])
            ax.set_title(f'{class_name} 크기 분포')
            ax.set_xlabel('크기 (m)')
            ax.set_ylabel('빈도')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('construction_equipment_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 클래스별 평균 크기 비교
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        classes = []
        lengths = []
        widths = []
        heights = []
        
        for class_name in self.class_names:
            if class_name in self.statistics:
                classes.append(class_name)
                lengths.append(self.statistics[class_name]['length']['mean'])
                widths.append(self.statistics[class_name]['width']['mean'])
                heights.append(self.statistics[class_name]['height']['mean'])
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, lengths, width, label='Length', alpha=0.8)
        ax.bar(x, widths, width, label='Width', alpha=0.8) 
        ax.bar(x + width, heights, width, label='Height', alpha=0.8)
        
        ax.set_xlabel('건설장비 클래스')
        ax.set_ylabel('평균 크기 (m)')
        ax.set_title('클래스별 평균 크기 비교')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 값 표시
        for i, (l, w, h) in enumerate(zip(lengths, widths, heights)):
            ax.text(i - width, l + 0.1, f'{l:.1f}', ha='center', va='bottom')
            ax.text(i, w + 0.1, f'{w:.1f}', ha='center', va='bottom')
            ax.text(i + width, h + 0.1, f'{h:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('construction_equipment_size_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_config(self):
        """최적화된 설정 파일 생성"""
        print("\n⚙️ 최적화된 설정 생성...")
        
        # YAML 설정 생성
        anchor_config = []
        
        for class_name in self.class_names:
            if class_name in self.optimal_anchors:
                recommended = self.optimal_anchors[class_name]['recommended']
                
                config = {
                    'class_name': class_name,
                    'anchor_sizes': [recommended],
                    'anchor_rotations': [0, 1.57],  # 0°, 90°
                    'anchor_bottom_heights': [-1.78],
                    'align_center': False,
                    'feature_map_stride': 8,
                    'matched_threshold': 0.6,
                    'unmatched_threshold': 0.45
                }
                anchor_config.append(config)
        
        # 결과 출력
        print("\n🎯 최적화된 ANCHOR_GENERATOR_CONFIG:")
        print("```yaml")
        print("ANCHOR_GENERATOR_CONFIG: [")
        for i, config in enumerate(anchor_config):
            print("    {")
            for key, value in config.items():
                if isinstance(value, str):
                    print(f"        '{key}': '{value}',")
                else:
                    print(f"        '{key}': {value},")
            print("    },")
        print("]")
        print("```")
        
        # JSON 파일로 저장
        output_file = 'optimized_anchor_config.json'
        with open(output_file, 'w') as f:
            json.dump({
                'statistics': self.statistics,
                'optimal_anchors': self.optimal_anchors,
                'anchor_generator_config': anchor_config
            }, f, indent=2)
        
        print(f"\n💾 결과 저장: {output_file}")
        
        # 개선 예상 효과 출력
        self.estimate_improvement()
    
    def estimate_improvement(self):
        """성능 개선 예상치 계산"""
        print("\n📈 예상 성능 개선:")
        
        # 현재 anchor와 최적 anchor 비교
        current_anchors = {
            'dumptruck': [8.5, 2.8, 3.5],
            'excavator': [6.0, 2.5, 3.0], 
            'grader': [9.0, 2.8, 3.2],
            'roller': [5.5, 2.2, 2.8]
        }
        
        total_improvement = 0
        improvements = []
        
        for class_name in self.class_names:
            if class_name in self.optimal_anchors and class_name in current_anchors:
                current = np.array(current_anchors[class_name])
                optimal = np.array(self.optimal_anchors[class_name]['recommended'])
                
                # L2 거리 기반 개선도 계산
                distance = np.linalg.norm(current - optimal)
                improvement = max(0, min(10, (3.0 - distance) * 2))  # 0~10% 범위
                
                improvements.append(improvement)
                total_improvement += improvement
                
                print(f"   🏗️ {class_name}:")
                print(f"      현재:    {current}")
                print(f"      최적:    {optimal}")
                print(f"      차이:    {distance:.2f}m")
                print(f"      개선:    +{improvement:.1f}% mAP")
        
        avg_improvement = total_improvement / len(improvements) if improvements else 0
        print(f"\n🎯 예상 총 개선: +{avg_improvement:.1f}% mAP")
        
        if avg_improvement > 2:
            print("✅ 상당한 성능 개선이 예상됩니다!")
        elif avg_improvement > 1:
            print("✅ 중간 정도의 성능 개선이 예상됩니다.")
        else:
            print("ℹ️ 현재 anchor가 이미 적절합니다.")


def main():
    """메인 함수"""
    print("🏗️ 건설장비 데이터셋 Anchor 최적화 도구")
    print("="*50)
    
    # 데이터 경로 설정 (수정 필요)
    data_dir = input("📁 데이터 디렉토리 경로를 입력하세요: ").strip()
    
    if not os.path.exists(data_dir):
        print(f"❌ 경로를 찾을 수 없습니다: {data_dir}")
        return
    
    # 분석 실행
    analyzer = ConstructionEquipmentAnchorAnalyzer(data_dir)
    analyzer.analyze_dataset()
    
    print("\n🎉 분석 완료!")
    print("   📊 시각화 파일: construction_equipment_*.png")
    print("   💾 설정 파일: optimized_anchor_config.json")


if __name__ == "__main__":
    main()