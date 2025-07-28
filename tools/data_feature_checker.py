import numpy as np
import os
from pathlib import Path
from collections import defaultdict

def check_all_point_cloud_features():
    """모든 point cloud 파일의 feature 차원을 확인"""
    
    data_path = Path("/home/sungwoo/VoxelNeXt/data/0528_real_synthetic/points")
    
    if not data_path.exists():
        print(f"ERROR: 데이터 경로가 존재하지 않습니다: {data_path}")
        return
    
    # .npy 파일 목록 가져오기
    npy_files = list(data_path.glob("*.npy"))
    print(f"총 {len(npy_files)}개의 .npy 파일을 찾았습니다.")
    
    if len(npy_files) == 0:
        print("ERROR: .npy 파일을 찾을 수 없습니다.")
        return
    
    # 차원별 통계
    dimension_stats = defaultdict(list)
    error_files = []
    
    print("\n=== 전체 파일 검사 시작 ===")
    
    for i, file_path in enumerate(npy_files):
        try:
            # Point cloud 로드
            points = np.load(file_path)
            
            # 차원 정보 저장
            shape = points.shape
            dimension_stats[shape[1]].append(file_path.name)
            
            # 진행 상황 출력 (100개마다)
            if (i + 1) % 100 == 0:
                print(f"진행률: {i+1}/{len(npy_files)} ({(i+1)/len(npy_files)*100:.1f}%)")
                
        except Exception as e:
            error_files.append((file_path.name, str(e)))
            print(f"ERROR loading {file_path.name}: {e}")
    
    print("\n=== 검사 결과 ===")
    
    # 차원별 통계 출력
    print(f"\n📊 차원별 파일 개수:")
    for dim, files in sorted(dimension_stats.items()):
        print(f"  {dim}차원: {len(files)}개 파일")
        
        # 처음 5개 파일명 예시
        if len(files) <= 5:
            print(f"    파일들: {files}")
        else:
            print(f"    예시 파일들: {files[:5]} ...")
    
    # 4차원이 아닌 파일들 상세 출력
    print(f"\n🔍 4차원이 아닌 파일들 상세:")
    for dim, files in dimension_stats.items():
        if dim != 4:
            print(f"\n  === {dim}차원 파일들 ({len(files)}개) ===")
            for file_name in files[:10]:  # 처음 10개만 출력
                file_path = data_path / file_name
                try:
                    points = np.load(file_path)
                    print(f"    {file_name}: shape={points.shape}, dtype={points.dtype}")
                    
                    # 첫 번째 파일의 샘플 데이터 출력
                    if file_name == files[0]:
                        print(f"      샘플 데이터 (첫 3개 포인트):")
                        for j in range(min(3, points.shape[0])):
                            print(f"        {points[j]}")
                            
                except Exception as e:
                    print(f"    {file_name}: ERROR - {e}")
            
            if len(files) > 10:
                print(f"    ... 그 외 {len(files)-10}개 파일 생략")
    
    # 에러 파일들
    if error_files:
        print(f"\n❌ 로딩 실패한 파일들 ({len(error_files)}개):")
        for file_name, error in error_files:
            print(f"  {file_name}: {error}")
    
    # 요약
    print(f"\n📋 요약:")
    print(f"  - 총 파일 수: {len(npy_files)}")
    print(f"  - 4차원 파일 수: {len(dimension_stats.get(4, []))}")
    print(f"  - 4차원이 아닌 파일 수: {sum(len(files) for dim, files in dimension_stats.items() if dim != 4)}")
    print(f"  - 에러 파일 수: {len(error_files)}")
    
    # 4차원이 아닌 파일이 있는 경우 경고
    non_4d_count = sum(len(files) for dim, files in dimension_stats.items() if dim != 4)
    if non_4d_count > 0:
        print(f"\n⚠️  경고: {non_4d_count}개 파일이 4차원이 아닙니다!")
        print("  YAML 설정에서 src_feature_list를 실제 데이터 차원에 맞게 수정해야 합니다.")
    else:
        print(f"\n✅ 모든 파일이 4차원입니다!")

def check_sample_files():
    """몇 개 샘플 파일의 상세 정보 확인"""
    
    data_path = Path("/home/sungwoo/VoxelNeXt/data/0528_real_synthetic/points")
    npy_files = list(data_path.glob("*.npy"))
    
    if len(npy_files) == 0:
        print("샘플 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n=== 샘플 파일 상세 정보 ===")
    
    # 처음 5개 파일 상세 검사
    for i, file_path in enumerate(npy_files[:5]):
        print(f"\n파일 {i+1}: {file_path.name}")
        try:
            points = np.load(file_path)
            print(f"  Shape: {points.shape}")
            print(f"  Dtype: {points.dtype}")
            print(f"  포인트 수: {points.shape[0]}")
            print(f"  Feature 수: {points.shape[1]}")
            
            if points.shape[1] >= 4:
                print(f"  X 범위: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
                print(f"  Y 범위: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
                print(f"  Z 범위: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
                print(f"  Intensity 범위: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
            else:
                print(f"  각 차원 범위:")
                for j in range(points.shape[1]):
                    print(f"    차원 {j}: [{points[:, j].min():.2f}, {points[:, j].max():.2f}]")
            
            # 첫 3개 포인트 출력
            print(f"  첫 3개 포인트:")
            for j in range(min(3, points.shape[0])):
                print(f"    {points[j]}")
                
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    print("🔍 Point Cloud 데이터 Feature 차원 검사")
    print("=" * 50)
    
    # 전체 파일 검사
    check_all_point_cloud_features()
    
    # 샘플 파일 상세 정보
    check_sample_files()