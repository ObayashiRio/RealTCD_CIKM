#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2024/12/19  quick_view.py
Rio Obayashi
隣接行列を簡単に確認するスクリプト
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
plt.rcParams['axes.unicode_minus'] = False

def quick_view_matrix(file_path, show_plot=True):
    """
    隣接行列を簡単に確認する
    
    Args:
        file_path (str): .npyファイルのパス
        show_plot (bool): プロットを表示するかどうか
    """
    print(f"=== {file_path} の分析 ===")
    
    # ファイルを読み込み
    data = np.load(file_path)
    
    # 基本情報
    print(f"Shape: {data.shape}")
    print(f"Non-zero elements: {np.count_nonzero(data)}")
    print(f"Density: {np.count_nonzero(data)/data.size*100:.2f}%")
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")
    
    # 非ゼロ要素の位置を表示
    nonzero_indices = np.nonzero(data)
    print(f"\n非ゼロ要素の位置 (最初の20個):")
    for i in range(min(20, len(nonzero_indices[0]))):
        row, col = nonzero_indices[0][i], nonzero_indices[1][i]
        value = data[row, col]
        print(f"  ({row:2d}, {col:2d}): {value}")
    
    if show_plot:
        # 可視化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 全体の隣接行列
        sns.heatmap(data, cmap='Blues', cbar=True, square=True, ax=ax1)
        ax1.set_title(f'隣接行列 (全体)\n{np.count_nonzero(data)}個の非ゼロ要素')
        ax1.set_xlabel('列')
        ax1.set_ylabel('行')
        
        # 非ゼロ要素のみを強調表示
        mask = data == 0
        sns.heatmap(data, cmap='Reds', cbar=True, square=True, 
                    mask=mask, ax=ax2, vmin=0, vmax=1)
        ax2.set_title(f'非ゼロ要素のみ\n密度: {np.count_nonzero(data)/data.size*100:.1f}%')
        ax2.set_xlabel('列')
        ax2.set_ylabel('行')
        
        plt.tight_layout()
        plt.show()
    
    return data

def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # デフォルトファイル
        file_path = 'llm_res/res_matrix_yuan.npy'
    
    if not file_path.endswith('.npy'):
        file_path += '.npy'
    
    try:
        data = quick_view_matrix(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません")
        print("利用可能なファイル:")
        import os
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.npy'):
                    print(f"  {os.path.join(root, file)}")
    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main() 