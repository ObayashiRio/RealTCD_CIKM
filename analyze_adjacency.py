#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2024/12/19  analyze_adjacency.py
Rio Obayashi
隣接行列の詳細分析を行うスクリプト
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_adjacency_matrix(file_path):
    """
    隣接行列の詳細分析を行う
    
    Args:
        file_path (str): .npyファイルのパス
    """
    print(f"=== 分析対象: {file_path} ===")
    
    # ファイルを読み込み
    data = np.load(file_path)
    
    # 基本情報
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")
    print(f"Mean value: {np.mean(data):.4f}")
    print(f"Non-zero elements: {np.count_nonzero(data)}")
    print(f"Density: {np.count_nonzero(data) / data.size * 100:.2f}%")
    
    # 非ゼロ要素の分析
    nonzero_indices = np.nonzero(data)
    print(f"\n=== 非ゼロ要素の詳細 ===")
    print(f"非ゼロ要素の数: {len(nonzero_indices[0])}")
    
    if len(nonzero_indices[0]) > 0:
        # 最初の10個の非ゼロ要素を表示
        print("最初の10個の非ゼロ要素:")
        for i in range(min(10, len(nonzero_indices[0]))):
            row, col = nonzero_indices[0][i], nonzero_indices[1][i]
            value = data[row, col]
            print(f"  Position ({row:2d}, {col:2d}): {value}")
    
    # 行・列ごとの非ゼロ要素数
    row_sums = np.sum(data != 0, axis=1)
    col_sums = np.sum(data != 0, axis=0)
    print(f"\n=== 行・列の分析 ===")
    print(f"最大行非ゼロ要素数: {np.max(row_sums)}")
    print(f"最小行非ゼロ要素数: {np.min(row_sums)}")
    print(f"平均行非ゼロ要素数: {np.mean(row_sums):.2f}")
    print(f"最大列非ゼロ要素数: {np.max(col_sums)}")
    print(f"最小列非ゼロ要素数: {np.min(col_sums)}")
    print(f"平均列非ゼロ要素数: {np.mean(col_sums):.2f}")
    
    return data

def visualize_adjacency_matrix(data, title="Adjacency Matrix"):
    """
    隣接行列を可視化する
    
    Args:
        data (np.ndarray): 隣接行列
        title (str): グラフのタイトル
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='Blues', cbar=True, square=True)
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.show()

def main():
    """メイン関数"""
    # 分析対象ファイル
    files_to_analyze = [
        'llm_res/res_matrix_yuan.npy',
        'llm_res/downstream_gpt4_basep_nodata_res_matrix.npy',
        'llm_res/gpt4noimp_res_matrix.npy'
    ]
    
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            data = analyze_adjacency_matrix(file_path)
            print("\n" + "="*50 + "\n")
        else:
            print(f"ファイルが見つかりません: {file_path}")

if __name__ == "__main__":
    main() 