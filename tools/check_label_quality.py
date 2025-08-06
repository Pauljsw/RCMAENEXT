# tools/check_label_quality.py
"""
라벨 품질 검증 및 부분 취득 감지 도구

부분 취득된 객체를 감지하고 anchor 전략을 개선
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


class LabelQualityChecker:
    """라벨 품질 검증기"""
    
    def __init__(self, optimized_config_path):
        # 기존 분석 결과 로드
        with open(optimized_config_path, 'r') as f:
            self.config = json.load(f)
        
        self.statistics = self.config['statistics']
        print("🔍 라벨 품질 검증 시작")
    
    def detect_partial_objects(self):
        """부분 취득 객체 감지"""
        print("\n📊 부분 취득 객체 분석...")
        
        partial_detection = {}
        
        for class_name, stats in self.statistics.items():
            if stats['count'] == 0:
                continue
                
            # IQR 기반 이상값 감지
            length_data = np.array([])  # 실제 데이터 필요
            
            # 통계 기반 분석
            mean_length = stats['length']['mean']
            std_length = stats['length']['std']
            q25_length = stats['length']['q25']
            q75_length = stats['length']['q75']
            
            # IQR 기반 이상값 경계
            iqr = q75_length - q25_length
            lower_bound = q25_length - 1.5 * iqr
            upper_bound = q75_length + 1.5 * iqr
            
            # 부분 취득 의심 비율 계산
            normal_size = mean_length
            partial_threshold = normal_size * 0.6  # 60% 미만이면 부분 취득 의심
            
            partial_ratio = max(0, (partial_threshold - stats['length']['min']) / 
                              (stats['length']['max'] - stats['length']['min']))
            
            partial_detection[class_name] = {
                'normal_size': normal_size,
                'partial_threshold': partial_threshold,
                'min_size': stats['length']['min'],
                'max_size': stats['length']['max'],
                'suspected_partial_ratio': partial_ratio,
                'iqr_lower': lower_bound,
                'iqr_upper': upper_bound
            }
            
            print(f"\n🏗️ {class_name}:")
            print(f"   📏 정상 크기: {normal_size:.1f}m")
            print(f"   ⚠️ 부분 취득 의심 기준: <{partial_threshold:.1f}m")
            print(f"   📊 크기 범위: {stats['length']['min']:.1f}~{stats['length']['max']:.1f}m")
            print(f"   🔍 부분 취득 의심 비율: {partial_ratio*100:.1f}%")
        
        return partial_detection
    
    def generate_multiscale_anchors(self, partial_detection):
        """다중 스케일 Anchor 생성"""
        print("\n🎯 Multi-scale Anchor 생성...")
        
        multiscale_config = []
        
        for class_name in ['dumptruck', 'excavator', 'grader', 'roller']:
            if class_name not in self.statistics or class_name not in partial_detection:
                continue
                
            stats = self.statistics[class_name]
            partial_info = partial_detection[class_name]
            
            # 원본 최적 크기
            optimal_size = [
                round(stats['length']['median'], 1),
                round(stats['width']['median'], 1),
                round(stats['height']['median'], 1)
            ]
            
            # 부분 취득 고려 크기 (70% 크기)
            partial_size = [
                round(optimal_size[0] * 0.7, 1),
                round(optimal_size[1] * 0.8, 1),  # 폭은 덜 변함
                round(optimal_size[2] * 0.9, 1)   # 높이는 거의 변하지 않음
            ]
            
            # 대형 크기 (120% 크기)
            large_size = [
                round(optimal_size[0] * 1.2, 1),
                round(optimal_size[1] * 1.1, 1),
                round(optimal_size[2] * 1.1, 1)
            ]
            
            # 단일 크기 vs 다중 크기 선택
            if partial_info['suspected_partial_ratio'] > 0.2:  # 20% 이상 부분 취득
                anchor_sizes = [partial_size, optimal_size, large_size]
                strategy = "Multi-scale (부분 취득 고려)"
            else:
                anchor_sizes = [optimal_size]
                strategy = "Single-scale (품질 양호)"
            
            config = {
                'class_name': class_name,
                'anchor_sizes': anchor_sizes,
                'anchor_rotations': [0, 1.57],
                'anchor_bottom_heights': [-1.78],
                'align_center': False,
                'feature_map_stride': 8,
                'matched_threshold': 0.6,
                'unmatched_threshold': 0.45
            }
            
            multiscale_config.append(config)
            
            print(f"   🏗️ {class_name}: {strategy}")
            print(f"      Anchor sizes: {anchor_sizes}")
        
        return multiscale_config
    
    def generate_improved_config(self):
        """개선된 설정 생성"""
        partial_detection = self.detect_partial_objects()
        multiscale_config = self.generate_multiscale_anchors(partial_detection)
        
        print("\n⚙️ 개선된 ANCHOR_GENERATOR_CONFIG:")
        print("```yaml")
        print("ANCHOR_GENERATOR_CONFIG: [")
        for config in multiscale_config:
            print("    {")
            for key, value in config.items():
                if isinstance(value, str):
                    print(f"        '{key}': '{value}',")
                else:
                    print(f"        '{key}': {value},")
            print("    },")
        print("]")
        print("```")
        
        # 개선된 설정 저장
        improved_config = {
            'partial_detection_analysis': partial_detection,
            'multiscale_anchor_config': multiscale_config,
            'recommendations': self._generate_recommendations(partial_detection)
        }
        
        with open('improved_anchor_config.json', 'w') as f:
            json.dump(improved_config, f, indent=2)
        
        print(f"\n💾 개선된 설정 저장: improved_anchor_config.json")
        return improved_config
    
    def _generate_recommendations(self, partial_detection):
        """권장사항 생성"""
        recommendations = []
        
        for class_name, info in partial_detection.items():
            if info['suspected_partial_ratio'] > 0.3:
                recommendations.append({
                    'class': class_name,
                    'issue': 'High partial acquisition suspected',
                    'action': 'Use multi-scale anchors',
                    'priority': 'High'
                })
            elif info['suspected_partial_ratio'] > 0.1:
                recommendations.append({
                    'class': class_name,
                    'issue': 'Moderate partial acquisition',
                    'action': 'Consider data cleaning',
                    'priority': 'Medium'
                })
        
        return recommendations


def main():
    """메인 함수"""
    print("🔍 라벨 품질 검증 및 개선 도구")
    print("="*50)
    
    config_path = 'optimized_anchor_config.json'
    
    if not Path(config_path).exists():
        print(f"❌ {config_path}를 찾을 수 없습니다!")
        print("   먼저 analyze_dataset_anchors.py를 실행하세요.")
        return
    
    checker = LabelQualityChecker(config_path)
    improved_config = checker.generate_improved_config()
    
    print("\n🎯 권장사항:")
    for rec in improved_config['recommendations']:
        print(f"   🏗️ {rec['class']}: {rec['action']} ({rec['priority']} priority)")
    
    print("\n💡 다음 단계:")
    print("   1. improved_anchor_config.json 확인")
    print("   2. Multi-scale anchor 적용 고려")
    print("   3. 부분 취득 데이터 검토 및 정제")


if __name__ == "__main__":
    main()