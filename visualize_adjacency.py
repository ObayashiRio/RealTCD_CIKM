#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2024/12/19  visualize_adjacency.py
Rio Obayashi
隣接行列の可視化を行うスクリプト
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定
plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
plt.rcParams['axes.unicode_minus'] = False

def visualize_adjacency_matrix(data, title="隣接行列", save_path=None):
    """
    隣接行列を可視化する
    
    Args:
        data (np.ndarray): 隣接行列
        title (str): グラフのタイトル
        save_path (str): 保存パス（Noneの場合は表示のみ）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 全体の隣接行列
    sns.heatmap(data, cmap='Blues', cbar=True, square=True, ax=ax1)
    ax1.set_title(f'{title} (全体)')
    ax1.set_xlabel('列')
    ax1.set_ylabel('行')
    
    # 非ゼロ要素のみを強調表示
    mask = data == 0
    sns.heatmap(data, cmap='Reds', cbar=True, square=True, 
                mask=mask, ax=ax2, vmin=0, vmax=1)
    ax2.set_title(f'{title} (非ゼロ要素のみ)')
    ax2.set_xlabel('列')
    ax2.set_ylabel('行')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"画像を保存しました: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_network_structure(data, title="ネットワーク構造", save_path=None):
    """
    ネットワーク構造を可視化する（非ゼロ要素の分布）
    
    Args:
        data (np.ndarray): 隣接行列
        title (str): グラフのタイトル
        save_path (str): 保存パス
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 行ごとの非ゼロ要素数
    row_sums = np.sum(data != 0, axis=1)
    ax1.bar(range(len(row_sums)), row_sums, alpha=0.7, color='skyblue')
    ax1.set_title('行ごとの非ゼロ要素数')
    ax1.set_xlabel('行番号')
    ax1.set_ylabel('非ゼロ要素数')
    ax1.grid(True, alpha=0.3)
    
    # 列ごとの非ゼロ要素数
    col_sums = np.sum(data != 0, axis=0)
    ax2.bar(range(len(col_sums)), col_sums, alpha=0.7, color='lightcoral')
    ax2.set_title('列ごとの非ゼロ要素数')
    ax2.set_xlabel('列番号')
    ax2.set_ylabel('非ゼロ要素数')
    ax2.grid(True, alpha=0.3)
    
    # 非ゼロ要素の位置を散布図で表示
    nonzero_indices = np.nonzero(data)
    ax3.scatter(nonzero_indices[1], nonzero_indices[0], 
                alpha=0.6, s=20, color='red')
    ax3.set_title('非ゼロ要素の位置')
    ax3.set_xlabel('列番号')
    ax3.set_ylabel('行番号')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 密度の分布
    ax4.hist(data[data != 0], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.set_title('非ゼロ要素の値の分布')
    ax4.set_xlabel('値')
    ax4.set_ylabel('頻度')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"画像を保存しました: {save_path}")
    else:
        plt.show()
    
    plt.close()

def compare_matrices(file_paths, save_dir="visualization_results"):
    """
    複数の隣接行列を比較して可視化する
    
    Args:
        file_paths (list): 比較するファイルパスのリスト
        save_dir (str): 保存ディレクトリ
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 各ファイルを読み込み
    matrices = {}
    for file_path in file_paths:
        if os.path.exists(file_path):
            name = os.path.basename(file_path).replace('.npy', '')
            matrices[name] = np.load(file_path)
            print(f"読み込み完了: {name} - 非ゼロ要素数: {np.count_nonzero(matrices[name])}")
    
    # 比較用のサブプロットを作成
    n_matrices = len(matrices)
    fig, axes = plt.subplots(2, n_matrices, figsize=(5*n_matrices, 10))
    
    if n_matrices == 1:
        axes = axes.reshape(2, 1)
    
    for i, (name, data) in enumerate(matrices.items()):
        # 全体の隣接行列
        sns.heatmap(data, cmap='Blues', cbar=True, square=True, 
                    ax=axes[0, i], xticklabels=False, yticklabels=False)
        axes[0, i].set_title(f'{name}\n(非ゼロ要素: {np.count_nonzero(data)}個)')
        
        # 非ゼロ要素のみ
        mask = data == 0
        sns.heatmap(data, cmap='Reds', cbar=True, square=True, 
                    mask=mask, ax=axes[1, i], vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False)
        axes[1, i].set_title(f'{name}\n(密度: {np.count_nonzero(data)/data.size*100:.1f}%)')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "matrix_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"比較画像を保存しました: {save_path}")
    plt.show()
    plt.close()

def main():
    """メイン関数"""
    # 分析対象ファイル
    files_to_visualize = [
        'llm_res/res_matrix_yuan.npy',
        'llm_res/downstream_gpt4_basep_nodata_res_matrix.npy',
        'llm_res/gpt4noimp_res_matrix.npy'
    ]
    
    # 保存ディレクトリ
    save_dir = "visualization_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("=== 隣接行列の可視化を開始します ===")
    
    for file_path in files_to_visualize:
        if os.path.exists(file_path):
            name = os.path.basename(file_path).replace('.npy', '')
            data = np.load(file_path)
            
            print(f"\n--- {name} の可視化 ---")
            print(f"非ゼロ要素数: {np.count_nonzero(data)}")
            print(f"密度: {np.count_nonzero(data)/data.size*100:.2f}%")
            
            # 隣接行列の可視化
            save_path = os.path.join(save_dir, f"{name}_adjacency.png")
            visualize_adjacency_matrix(data, f"{name} 隣接行列", save_path)
            
            # ネットワーク構造の可視化
            save_path = os.path.join(save_dir, f"{name}_network.png")
            visualize_network_structure(data, f"{name} ネットワーク構造", save_path)
            
        else:
            print(f"ファイルが見つかりません: {file_path}")
    
    # 比較可視化
    print("\n--- 行列の比較可視化 ---")
    compare_matrices(files_to_visualize, save_dir)
    
    print(f"\n=== 可視化完了 ===")
    print(f"結果は '{save_dir}' ディレクトリに保存されました")

if __name__ == "__main__":
    main() 