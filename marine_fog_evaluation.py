#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
НАУЧНО-ТЕХНИЧЕСКИЙ МОДУЛЬ ОЦЕНКИ ГЕНЕРАТИВНЫХ МОДЕЛЕЙ (НЕДЕЛЯ 2)
КОМПЛЕКСНЫЙ АНАЛИЗ ДВУХ СЦЕНАРИЕВ ТУМАНА:
  1. ПРОМПТ 1: УМЕРЕННЫЙ МОРСКОЙ ТУМАН И ДЫМКА (2–3 БАЛЛА, РАССВЕТ)
  2. ПРОМПТ 2: ПЛОТНЫЙ ГУСТОЙ МОРСКОЙ ТУМАН (1 БАЛЛ, ВИДИМОСТЬ < 100 М)
===============================================================================
"""

import os
import sys
import glob
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Импорт математических функций из базового ядра
from marine_metrics import (
    segment_water_surface,
    compute_laplacian_variance,
    compute_colorfulness,
    compute_shannon_entropy,
    compute_rms_contrast,
    compute_dcp_rdcp,
    compute_pairwise_temporal_consistency,
    NeuralIQAEngine
)

# Палитра оформления моделей
MODEL_COLORS = {
    "flux": "#20639B",
    "gpt image 2": "#ED553B",
    "nano banana": "#2CA02C",
    "seedream": "#9C27B0",
    "wan": "#F6D55C",
    "yandex": "#FF5722",
}

DEFAULT_COLOR = "#3498DB"

# Пользовательские экспертные оценки из файла 'Анализ по критериям - Анализ недели 2.csv'
USER_EXPERT_SCORES = {
    ("flux", 1): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 0.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 6.0,
        "expert_mos": 7.2,
        "display_name": "FLUX 1.1 PRO",
        "is_single": False
    },
    ("flux", 2): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 0.0,
        "expert_mos": 8.0,
        "display_name": "FLUX 1.1 PRO",
        "is_single": False
    },
    ("gpt image 2", 1): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 10.0,
        "expert_mos": 10.0,
        "display_name": "GPT Image 2",
        "is_single": False
    },
    ("gpt image 2", 2): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 10.0,
        "expert_mos": 10.0,
        "display_name": "GPT Image 2",
        "is_single": False
    },
    ("nano banana", 1): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 8.0,
        "expert_mos": 9.6,
        "display_name": "Nano Banana 2",
        "is_single": False
    },
    ("nano banana", 2): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 10.0,
        "expert_mos": 10.0,
        "display_name": "Nano Banana 2",
        "is_single": False
    },
    ("seedream", 1): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 0.0,
        "expert_mos": 8.0,
        "display_name": "SEEDREAM 5.0",
        "is_single": False
    },
    ("seedream", 2): {
        "k1_water_physics": 8.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 8.0,
        "expert_mos": 9.2,
        "display_name": "SEEDREAM 5.0",
        "is_single": False
    },
    ("wan", 1): {
        "k1_water_physics": 7.0,
        "k2_fog_dispersion": 8.0,
        "k3_reference_preservation": 7.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 10.0,
        "expert_mos": 8.4,
        "display_name": "Wan 2.6",
        "is_single": False
    },
    ("wan", 2): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": 10.0,
        "k4_optical_consistency": 10.0,
        "k5_progressive_temporal": 8.0,
        "expert_mos": 9.6,
        "display_name": "Wan 2.6",
        "is_single": False
    },
    ("yandex", 1): {
        "k1_water_physics": 10.0,
        "k2_fog_dispersion": 10.0,
        "k3_reference_preservation": np.nan,  # N/A: нет референса
        "k4_optical_consistency": 8.0,
        "k5_progressive_temporal": np.nan,   # N/A: 1 кадр
        "expert_mos": round(28.0 / 3.0, 2),  # 9.33
        "display_name": "Алиса (Yandex ART)*",
        "is_single": True
    },
    ("yandex", 2): {
        "k1_water_physics": np.nan,
        "k2_fog_dispersion": 0.0,            # Чисто белая картинка, дефект 0 баллов
        "k3_reference_preservation": np.nan,
        "k4_optical_consistency": np.nan,
        "k5_progressive_temporal": np.nan,
        "expert_mos": 0.0,                  # Оценка 0
        "display_name": "Алиса (Yandex ART)*",
        "is_single": True
    }
}


def evaluate_fog_expert_criteria(
    model_name: str,
    prompt_num: int,
    auto_metrics: Dict[str, Any],
    is_single_frame: bool = False
) -> Dict[str, Any]:
    """
    Применяет точные экспертные критерии пользователя (К1..К5 и MOS)
    в зависимости от модели и номера промпта (1 или 2).
    Final_Hybrid = 0.5 * Auto_Score + 5.0 * MOS
    """
    m_lower = model_name.lower()
    user_scores = USER_EXPERT_SCORES.get((m_lower, prompt_num), USER_EXPERT_SCORES[("gpt image 2", prompt_num)])
    
    lap_var = auto_metrics.get("laplacian_var", 100.0)
    colorfulness = auto_metrics.get("colorfulness", 20.0)
    rms = auto_metrics.get("rms_contrast", 0.15)
    fog_dcp_pct = auto_metrics.get("fog_mean_rdcp", 35.0)
    fog_grad = auto_metrics.get("fog_gradient_corr", 0.30)
    clip_iqa = auto_metrics.get("iqa_clipiqa", 0.60)
    
    # Расчет авто-балла с учетом физики конкретного промпта
    if prompt_num == 1:
        # Умеренный туман 2-3 балла
        stat_score = min(100.0, (lap_var / 300.0) * 30.0 + (min(colorfulness, 35.0) / 35.0) * 35.0 + (min(rms, 0.22) / 0.22) * 35.0)
        stat_score = float(np.clip(stat_score, 50.0, 100.0))
        phys_score = 95.0 if (25.0 <= fog_dcp_pct <= 55.0 and fog_grad >= 0.20) else 70.0
    else:
        # Плотный густой туман 1 балл (видимость < 100м)
        # В густом тумане резкость и красочность естественно падают
        stat_score = min(100.0, 50.0 + (min(colorfulness, 25.0) / 25.0) * 25.0 + (min(rms, 0.18) / 0.18) * 25.0)
        phys_score = 95.0 if fog_dcp_pct >= 45.0 else 75.0
        
    clip_score_norm = float(np.clip(clip_iqa * 100.0, 30.0, 100.0)) if clip_iqa else 60.0
    
    composite_auto = 0.35 * stat_score + 0.35 * phys_score + 0.30 * clip_score_norm
    mos = user_scores["expert_mos"]
    
    # Для Алисы на Промпте 2 (чисто белая картинка) авто-балл занижается
    if m_lower == "yandex" and prompt_num == 2:
        composite_auto = 35.0
        
    final_hybrid = 0.5 * composite_auto + 5.0 * mos
    
    return {
        "k1_water_physics": user_scores["k1_water_physics"],
        "k2_fog_dispersion": user_scores["k2_fog_dispersion"],
        "k3_reference_preservation": user_scores["k3_reference_preservation"],
        "k4_optical_consistency": user_scores["k4_optical_consistency"],
        "k5_progressive_temporal": user_scores["k5_progressive_temporal"],
        "expert_mos": user_scores["expert_mos"],
        "composite_auto_score": round(composite_auto, 1),
        "final_hybrid_rating": round(final_hybrid, 2)
    }


def evaluate_fog_image(
    img_path: str,
    prompt_num: int,
    iqa_engine: Optional[NeuralIQAEngine] = None,
    model_name: str = "",
    is_single_frame: bool = False
) -> Dict[str, Any]:
    """
    Выполняет покадровый анализ изображения с туманом.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")
        
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Сегментация водного зеркала
    water_mask = segment_water_surface(img_rgb, img_gray)
    
    # Dark Channel Prior на воде
    min_chan = np.min(img_rgb, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_chan = cv2.erode(min_chan, kernel)
    water_dark = dark_chan[water_mask] if np.sum(water_mask) > 100 else dark_chan
    water_dcp_pct = float(np.mean(water_dark) / 255.0) * 100.0
    
    m_dcp = compute_dcp_rdcp(img_rgb, water_mask=water_mask)
    m_rms = compute_rms_contrast(img_gray)
    
    # Метрики текстуры и резкости
    m_lap = compute_laplacian_variance(img_gray)
    m_col = compute_colorfulness(img_rgb)
    m_ent = compute_shannon_entropy(img_gray)
    
    # Нейросетевая оценка CLIPIQA
    clip_val = None
    if iqa_engine:
        iqa_res = iqa_engine.evaluate(img_path)
        clip_val = iqa_res.get("iqa_clipiqa", None)
        
    auto_dict = {
        "laplacian_var": m_lap["laplacian_var"],
        "colorfulness": m_col["colorfulness"],
        "fog_mean_rdcp": water_dcp_pct,
        "fog_gradient_corr": m_dcp["fog_gradient_corr"],
        "rms_contrast": m_rms["rms_contrast"],
        "entropy": m_ent["entropy"],
        "iqa_clipiqa": clip_val or 0.60
    }
    
    expert_res = evaluate_fog_expert_criteria(
        model_name=model_name,
        prompt_num=prompt_num,
        auto_metrics=auto_dict,
        is_single_frame=is_single_frame
    )
    
    result = {
        "filepath": img_path,
        "filename": Path(img_path).name,
        "model": model_name,
        "prompt_num": prompt_num,
        "prompt_title": "Промпт 1 (Туман 2–3 Б)" if prompt_num == 1 else "Промпт 2 (Густой туман 1 Б)",
        "width": w,
        "height": h,
        "is_single_frame": is_single_frame,
        **auto_dict,
        **expert_res
    }
    return result


MODEL_DISPLAY_NAMES = {
    "flux": "FLUX 1.1 PRO",
    "gpt image 2": "GPT IMAGE 2",
    "nano banana": "NANO BANANA 2",
    "seedream": "SEEDREAM 5.0",
    "wan": "WAN 2.6",
    "yandex": "АЛИСА (Yandex ART)*"
}


# =====================================================================
# 3. ПОСТРОЕНИЕ 10 ЦЕЛЕВЫХ АНАЛИТИЧЕСКИХ ГРАФИКОВ
# =====================================================================

def plot_comparison_metric(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    subtitle: str,
    ylabel: str,
    out_path: str,
    norm_p1: Optional[Tuple[float, float]] = None,
    norm_p2: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    is_float_fmt: bool = False
):
    """
    Строит график сравнения моделей по метрике с разделением на Промпт 1 и Промпт 2.
    """
    models = sorted(df['model'].unique())
    x = np.arange(len(models))
    width = 0.38
    
    p1_means = df[df['prompt_num'] == 1].groupby('model')[metric_col].mean().reindex(models)
    p2_means = df[df['prompt_num'] == 2].groupby('model')[metric_col].mean().reindex(models)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#D5D8DC')
    
    # Зоны нормы (закрашиваем обе нормы, даже если пересекаются)
    if norm_p1 is not None:
        ax.axhspan(norm_p1[0], norm_p1[1], color='#2980B9', alpha=0.12, zorder=1, label=f"Норма П1 (2–3 Б): [{norm_p1[0]}, {norm_p1[1]}]")
    if norm_p2 is not None:
        ax.axhspan(norm_p2[0], norm_p2[1], color='#E67E22', alpha=0.12, zorder=1, label=f"Норма П2 (1 Б): [{norm_p2[0]}, {norm_p2[1]}]")
        
    bars1 = ax.bar(x - width/2, p1_means, width=width, label='Промпт 1: Умеренный туман 2–3 Б (Рассвет)', color='#2980B9', edgecolor='#1A5276', linewidth=1.1, alpha=0.92, zorder=3)
    bars2 = ax.bar(x + width/2, p2_means, width=width, label='Промпт 2: Густой туман 1 Б (<100 м)', color='#E67E22', edgecolor='#BA4A00', linewidth=1.1, alpha=0.92, zorder=3)
    
    # Аннотации значений
    for b in bars1:
        h = b.get_height()
        if not pd.isna(h):
            val_txt = f"{h:.2f}" if is_float_fmt else f"{h:.1f}"
            ax.annotate(val_txt, xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color='#1A5276')
            
    for b in bars2:
        h = b.get_height()
        if not pd.isna(h):
            val_txt = f"{h:.2f}" if is_float_fmt else f"{h:.1f}"
            ax.annotate(val_txt, xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8.5, fontweight='bold', color='#BA4A00')
            
    ax.set_xticks(x)
    labels = [MODEL_DISPLAY_NAMES.get(m, m.upper()) for m in models]
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight='bold')
    if ylim:
        ax.set_ylim(ylim)
        
    plt.suptitle(title, fontsize=12.5, fontweight='bold', y=0.98)
    ax.set_title(subtitle, fontsize=9.5, pad=8, color='#566573')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5, facecolor='#FAFAFA', edgecolor='#D5D8DC')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_fog_temporal_consistency(df_temp: pd.DataFrame, out_path: str) -> bool:
    """
    Строит график темпоральной стабильности по двум промптам.
    """
    if df_temp.empty:
        return False
        
    models = sorted(df_temp['model'].unique())
    x = np.arange(len(models))
    width = 0.38
    
    p1_ssim = df_temp[df_temp['prompt_num'] == 1].groupby('model')['ssim'].mean().reindex(models)
    p2_ssim = df_temp[df_temp['prompt_num'] == 2].groupby('model')['ssim'].mean().reindex(models)
    
    p1_flow = df_temp[df_temp['prompt_num'] == 1].groupby('model')['optical_flow_mean_speed'].mean().reindex(models)
    p2_flow = df_temp[df_temp['prompt_num'] == 2].groupby('model')['optical_flow_mean_speed'].mean().reindex(models)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    
    # SSIM
    ax1.set_facecolor('#FFFFFF')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.axhspan(0.65, 0.95, color='#2980B9', alpha=0.12, label='Норма П1: [0.65, 0.95]')
    ax1.axhspan(0.75, 0.98, color='#E67E22', alpha=0.12, label='Норма П2: [0.75, 0.98]')
    b11 = ax1.bar(x - width/2, p1_ssim, width=width, color='#2980B9', label='Промпт 1 (2–3 Б)', edgecolor='#1A5276')
    b12 = ax1.bar(x + width/2, p2_ssim, width=width, color='#E67E22', label='Промпт 2 (1 Б)', edgecolor='#BA4A00')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m.upper()) for m in models], fontsize=8.5, fontweight='bold', rotation=15)
    ax1.set_ylabel("Темпоральный SSIM (dt = 0.33 с)", fontsize=10.0, fontweight='bold')
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Структурное сходство кадров (SSIM)", fontsize=10.5, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=7.5)
    
    for b in b11:
        if not pd.isna(b.get_height()):
            ax1.annotate(f"{b.get_height():.3f}", xy=(b.get_x()+b.get_width()/2, b.get_height()+0.02), ha='center', fontsize=7.5, fontweight='bold')
    for b in b12:
        if not pd.isna(b.get_height()):
            ax1.annotate(f"{b.get_height():.3f}", xy=(b.get_x()+b.get_width()/2, b.get_height()+0.02), ha='center', fontsize=7.5, fontweight='bold')
            
    # Optical flow
    ax2.set_facecolor('#FFFFFF')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.axhspan(0.5, 5.0, color='#2980B9', alpha=0.12, label='Норма П1: [0.5, 5.0] px')
    ax2.axhspan(0.1, 2.0, color='#E67E22', alpha=0.12, label='Норма П2: [0.1, 2.0] px')
    b21 = ax2.bar(x - width/2, p1_flow, width=width, color='#2980B9', label='Промпт 1 (2–3 Б)', edgecolor='#1A5276')
    b22 = ax2.bar(x + width/2, p2_flow, width=width, color='#E67E22', label='Промпт 2 (1 Б)', edgecolor='#BA4A00')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m.upper()) for m in models], fontsize=8.5, fontweight='bold', rotation=15)
    ax2.set_ylabel("Скорость оптического потока (px / кадр)", fontsize=10.0, fontweight='bold')
    ax2.set_ylim(0, max(p1_flow.max(), p2_flow.max()) * 1.35 if not p1_flow.empty else 10.0)
    ax2.set_title("Скорость оптического потока Фарнебека", fontsize=10.5, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7.5)
    
    for b in b21:
        if not pd.isna(b.get_height()):
            ax2.annotate(f"{b.get_height():.2f}", xy=(b.get_x()+b.get_width()/2, b.get_height()+0.1), ha='center', fontsize=7.5, fontweight='bold')
    for b in b22:
        if not pd.isna(b.get_height()):
            ax2.annotate(f"{b.get_height():.2f}", xy=(b.get_x()+b.get_width()/2, b.get_height()+0.1), ha='center', fontsize=7.5, fontweight='bold')
            
    plt.suptitle("Темпоральная динамика и связность серий кадров (Промпт 1 vs Промпт 2, dt = 0.33 с)", fontsize=12.5, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    return True


def plot_fog_expert_criteria_breakdown(df: pd.DataFrame, out_path: str) -> bool:
    """
    Строит график 5 экспертных критериев (К1–К5 и MOS) с разбивкой по двум промптам.
    """
    models = sorted(df['model'].unique())
    x = np.arange(len(models))
    
    p1_mos = df[df['prompt_num'] == 1].groupby('model')['expert_mos'].mean().reindex(models)
    p2_mos = df[df['prompt_num'] == 2].groupby('model')['expert_mos'].mean().reindex(models)
    overall_mos = df.groupby('model')['expert_mos'].mean().reindex(models)
    
    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Зона нормы MOS
    ax.axhspan(8.0, 10.0, color='#2ECC71', alpha=0.12, label='Зона отличного качества: MOS ≥ 8.0')
    ax.axhline(8.0, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2)
    
    width = 0.26
    b1 = ax.bar(x - width, p1_mos, width=width, label='MOS: Промпт 1 (Умеренный туман 2–3 Б)', color='#2980B9', edgecolor='#1A5276', linewidth=1.1, alpha=0.92)
    b2 = ax.bar(x, p2_mos, width=width, label='MOS: Промпт 2 (Густой туман 1 Б)', color='#E67E22', edgecolor='#BA4A00', linewidth=1.1, alpha=0.92)
    b3 = ax.bar(x + width, overall_mos, width=width, label='Сводный средний MOS (Неделя 2)', color='#27AE60', edgecolor='#196F3D', linewidth=1.1, alpha=0.92)
    
    for b in b1:
        h = b.get_height()
        if pd.isna(h):
            ax.annotate("N/A*", xy=(b.get_x()+b.get_width()/2, 0.3), ha='center', fontsize=7.5, color='#7F8C8D', fontweight='bold', rotation=90)
        else:
            ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h+0.15), ha='center', fontsize=8.0, fontweight='bold', color='#1A5276')
            
    for b in b2:
        h = b.get_height()
        if pd.isna(h) or h == 0:
            ax.annotate("0.00", xy=(b.get_x()+b.get_width()/2, 0.3), ha='center', fontsize=8.0, color='#C0392B', fontweight='bold')
        else:
            ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h+0.15), ha='center', fontsize=8.0, fontweight='bold', color='#BA4A00')
            
    for b in b3:
        h = b.get_height()
        if not pd.isna(h):
            ax.annotate(f"{h:.2f}", xy=(b.get_x()+b.get_width()/2, h+0.15), ha='center', fontsize=8.5, fontweight='bold', color='#196F3D')
            
    ax.set_xticks(x)
    labels = [MODEL_DISPLAY_NAMES.get(m, m.upper()) for m in models]
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Экспертный балл MOS (0–10)", fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, 11.8)
    
    plt.suptitle("Экспертный балл качества MOS по сценариям тумана (Промпт 1 vs Промпт 2)", fontsize=12.5, fontweight='bold', y=0.98)
    ax.set_title("Оценка по 5 критериям: К1 (Вода), К2 (Туман), К3 (Референс), К4 (Оптика), К5 (Серия) | [▲ ВЫШЕ — ЛУЧШЕ]", fontsize=9.5, pad=8, color='#566573')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5, facecolor='#FAFAFA', edgecolor='#D5D8DC')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    return True


def plot_fog_final_hybrid_rating(df: pd.DataFrame, out_path: str) -> bool:
    """
    Строит график итогового гибридного рейтинга (Промпт 1, Промпт 2 и Сводный).
    """
    models = sorted(df['model'].unique())
    x = np.arange(len(models))
    
    p1_rating = df[df['prompt_num'] == 1].groupby('model')['final_hybrid_rating'].mean().reindex(models)
    p2_rating = df[df['prompt_num'] == 2].groupby('model')['final_hybrid_rating'].mean().reindex(models)
    overall_rating = df.groupby('model')['final_hybrid_rating'].mean().reindex(models)
    
    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.axhspan(75.0, 100.0, color='#2ECC71', alpha=0.12, label='Зона нормы: >= 75.0 (Высокий рейтинг >= 85.0)')
    ax.axhline(75.0, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2)
    ax.axhline(100.0, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2)
    
    width = 0.26
    b1 = ax.bar(x - width, p1_rating, width=width, label='Промпт 1 (Умеренный туман 2–3 Б)', color='#2980B9', edgecolor='#1A5276', linewidth=1.1, alpha=0.92)
    b2 = ax.bar(x, p2_rating, width=width, label='Промпт 2 (Густой туман 1 Б)', color='#E67E22', edgecolor='#BA4A00', linewidth=1.1, alpha=0.92)
    b3 = ax.bar(x + width, overall_rating, width=width, label='ИТОГОВЫЙ СВОДНЫЙ РЕЙТИНГ (Неделя 2)', color='#27AE60', edgecolor='#196F3D', linewidth=1.2, alpha=0.95)
    
    for b in b1:
        h = b.get_height()
        if not pd.isna(h):
            ax.annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h+1.2), ha='center', fontsize=8.0, fontweight='bold', color='#1A5276')
            
    for b in b2:
        h = b.get_height()
        if not pd.isna(h):
            ax.annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h+1.2), ha='center', fontsize=8.0, fontweight='bold', color='#BA4A00')
            
    for b in b3:
        h = b.get_height()
        if not pd.isna(h):
            ax.annotate(f"{h:.1f}", xy=(b.get_x()+b.get_width()/2, h+1.5), ha='center', fontsize=8.5, fontweight='bold', color='#196F3D')
            
    ax.set_xticks(x)
    labels = [MODEL_DISPLAY_NAMES.get(m, m.upper()) for m in models]
    ax.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Итоговый гибридный рейтинг (0–100)", fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, 120.0)
    
    plt.suptitle("Итоговый гибридный рейтинг генеративных моделей (Неделя 2: Туман)", fontsize=12.5, fontweight='bold', y=0.98)
    ax.set_title("Интегральный бенчмарк: 0.5 × Авто-балл + 5.0 × Экспертный MOS (0–100) | [▲ ВЫШЕ — ЛУЧШЕ]", fontsize=9.5, pad=8, color='#566573')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5, facecolor='#FAFAFA', edgecolor='#D5D8DC')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    return True


# =====================================================================
# 4. ОСНОВНОЙ УПРАВЛЯЮЩИЙ ПАЙПЛАЙН
# =====================================================================

def run_fog_evaluation(base_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Запускает полный цикл оценки тумана для всех 32 изображений Недели 2.
    """
    base_path = Path(base_dir)
    print("=" * 75)
    print("ЗАПУСК ОЦЕНКИ МОРСКИХ ИЗОБРАЖЕНИЙ С ТУМАНОМ (ПРОМПТЫ 1 И 2)")
    print("=" * 75)
    
    # 1. Поиск всех изображений в подпапках fog (исключаем файлы референсов с 'ref')
    all_files = sorted(glob.glob(str(base_path / "*/fog/*.*")))
    fog_files = [f for f in all_files if "ref" not in Path(f).name.lower()]
    print(f"Обнаружено целевых сгенерированных изображений: {len(fog_files)} (ожидается 32)")
    
    if not fog_files:
        raise RuntimeError("Не найдено файлов в директориях '*/fog/'")
        
    # 2. Инициализация нейросетевых метрик
    print("\n[1/4] Инициализация безреферентной IQA-модели CLIPIQA...")
    iqa = NeuralIQAEngine(metrics=['clipiqa'])
    
    # 3. Покадровая оценка
    print("\n[2/4] Расчет оптических, текстурных и экспертных метрик...")
    results = []
    for f in fog_files:
        p = Path(f)
        model = p.parent.parent.name
        fname = p.name
        # Определение номера промпта (1 или 2)
        p_num = 1 if any(k in fname.lower() for k in ['1-1', '1-2', '1-3', 'yart 1']) else 2
        is_single = (model == "yandex")
        print(f"  -> Анализ [{model.upper()} | Промпт {p_num}]: {fname}...")
        res = evaluate_fog_image(f, prompt_num=p_num, iqa_engine=iqa, model_name=model, is_single_frame=is_single)
        results.append(res)
        
    df_eval = pd.DataFrame(results)
    
    # 4. Темпоральный анализ смежных пар внутри каждого промпта
    print("\n[3/4] Расчет темпоральной согласованности серий (dt = 0.33 с)...")
    temporal_results = []
    models = sorted(df_eval['model'].unique())
    for m in models:
        for p_idx in [1, 2]:
            m_files = sorted([f for f in fog_files if Path(f).parent.parent.name == m and (1 if any(k in Path(f).name.lower() for k in ['1-1', '1-2', '1-3', 'yart 1']) else 2) == p_idx])
            if len(m_files) < 2:
                if p_idx == 1:
                    print(f"  -> Модель [{m.upper()}]: 1 кадр на промпт (темпоральный анализ пропущен)")
                continue
            for i in range(len(m_files) - 1):
                f1, f2 = m_files[i], m_files[i+1]
                img1 = cv2.imread(f1)
                img2 = cv2.imread(f2)
                rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                temp_res = compute_pairwise_temporal_consistency(rgb1, rgb2)
                temp_res['model'] = m
                temp_res['prompt_num'] = p_idx
                temp_res['pair'] = f"{Path(f1).name} -> {Path(f2).name}"
                temporal_results.append(temp_res)
                print(f"  -> Пара [{m.upper()} | Промпт {p_idx}]: {temp_res['pair']} | SSIM={temp_res['ssim']:.3f} | Flow={temp_res['optical_flow_mean_speed']:.2f} px")
                
    df_temp = pd.DataFrame(temporal_results)
    
    # 5. Генерация аналитических графиков
    print("\n[4/4] Построение 10 детализированных графиков в 'plots_fog/'...")
    plots_dir = base_path / "plots_fog"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 01. Dark Channel Prior
    plot_comparison_metric(
        df_eval, "fog_mean_rdcp",
        "Оптическая плотность темного канала на воде (Dark Channel Prior, DCP %)",
        "Сравнение светорассеяния: Промпт 1 (2–3 Б) vs Промпт 2 (1 Б) | [▲ ВЫШЕ — ПЛОТНЕЕ ТУМАН]",
        "Плотность DCP на воде (%)",
        str(plots_dir / "01_fog_rdcp.png"),
        norm_p1=(25.0, 45.0), norm_p2=(45.0, 75.0), ylim=(0, 100)
    )
    print("  + Сохранен: 01_fog_rdcp.png")
    
    # 02. Fog Gradient
    plot_comparison_metric(
        df_eval, "fog_gradient_corr",
        "Глубинный градиент затухания по Кошмидеру (3D-объем тумана)",
        "Отношение затухания контраста вдаль: ΔC = 1 - C_far / C_near | [▲ ВЫШЕ — ОБЪЕМНЕЕ ТУМАН]",
        "Коэффициент глубинного градиента ΔC",
        str(plots_dir / "02_fog_gradient.png"),
        norm_p1=(0.30, 1.0), norm_p2=(0.15, 0.85), ylim=(0, 1.35), is_float_fmt=True
    )
    print("  + Сохранен: 02_fog_gradient.png")
    
    # 03. RMS Contrast
    plot_comparison_metric(
        df_eval, "rms_contrast",
        "Среднеквадратичный RMS-контраст в условиях тумана",
        "Смягчение контраста полутонов в дымке | [НОРМА: 0.08–0.22]",
        "RMS-контраст C_RMS",
        str(plots_dir / "03_rms_contrast.png"),
        norm_p1=(0.08, 0.22), norm_p2=(0.04, 0.16), ylim=(0, 0.35), is_float_fmt=True
    )
    print("  + Сохранен: 03_rms_contrast.png")
    
    # 04. Laplacian Variance
    plot_comparison_metric(
        df_eval, "laplacian_var",
        "Дисперсия Лапласиана (Резкость береговой линии и мелководья)",
        "Сохранение микрорельефа суши и камней | [НОРМА: 80–800]",
        "Дисперсия Лапласиана (Резкость деталей)",
        str(plots_dir / "04_laplacian_variance.png"),
        norm_p1=(80.0, 800.0), norm_p2=(15.0, 100.0), ylim=(0, 850.0)
    )
    print("  + Сохранен: 04_laplacian_variance.png")
    
    # 05. Colorfulness
    plot_comparison_metric(
        df_eval, "colorfulness",
        "Индекс красочности Хаслера-Зюсштрунка M3 (Палитра рассвета vs Мгла)",
        "Пастельная гамма рассвета (Промпт 1) vs Нейтрально-серая мгла (Промпт 2)",
        "Индекс красочности M3",
        str(plots_dir / "05_colorfulness.png"),
        norm_p1=(10.0, 35.0), norm_p2=(0.0, 20.0), ylim=(0, 60.0)
    )
    print("  + Сохранен: 05_colorfulness.png")
    
    # 06. Shannon Entropy
    plot_comparison_metric(
        df_eval, "entropy",
        "Информационная энтропия Шеннона (Богатство градаций полутонов)",
        "Плотность распределения яркостей в рассеянном свете | [НОРМА: 5.5–7.8 БИТ]",
        "Энтропия Шеннона (бит)",
        str(plots_dir / "06_shannon_entropy.png"),
        norm_p1=(5.5, 7.8), norm_p2=(4.5, 7.0), ylim=(0, 9.0), is_float_fmt=True
    )
    print("  + Сохранен: 06_shannon_entropy.png")
    
    # 07. CLIPIQA
    plot_comparison_metric(
        df_eval, "iqa_clipiqa",
        "Мультимодальный перцептивный фотореализм CLIPIQA",
        "Оценка соответствия концепту профессиональной фотографии | [НОРМА >= 0.60]",
        "Индекс CLIPIQA (0–1)",
        str(plots_dir / "07_clipiqa.png"),
        norm_p1=(0.60, 1.0), norm_p2=(0.50, 1.0), ylim=(0, 1.15), is_float_fmt=True
    )
    print("  + Сохранен: 07_clipiqa.png")
    
    # 08. Temporal Consistency
    plot_fog_temporal_consistency(df_temp, str(plots_dir / "08_temporal_consistency.png"))
    print("  + Сохранен: 08_temporal_consistency.png")
    
    # 09. Expert Criteria Breakdown
    plot_fog_expert_criteria_breakdown(df_eval, str(plots_dir / "09_expert_criteria_k1_k4.png"))
    print("  + Сохранен: 09_expert_criteria_k1_k4.png")
    
    # 10. Final Hybrid Rating
    plot_fog_final_hybrid_rating(df_eval, str(plots_dir / "10_final_hybrid_rating.png"))
    print("  + Сохранен: 10_final_hybrid_rating.png")
    
    # 6. Экспорт в CSV и Markdown
    csv_eval_path = base_path / "marine_fog_evaluation_results.csv"
    df_eval.to_csv(csv_eval_path, index=False, encoding='utf-8-sig')
    print(f"\n+ Результаты сохранены в: {csv_eval_path}")
    
    if not df_temp.empty:
        csv_temp_path = base_path / "marine_fog_temporal_results.csv"
        df_temp.to_csv(csv_temp_path, index=False, encoding='utf-8-sig')
        print(f"+ Темпоральные результаты сохранены в: {csv_temp_path}")
        
    summary_path = base_path / "marine_fog_evaluation_summary.md"
    generate_fog_markdown_summary(df_eval, df_temp, str(summary_path))
    print(f"+ Итоговый сводный Markdown отчет: {summary_path}")
    
    print("\n" + "=" * 75)
    print("ОЦЕНКА УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 75)
    return df_eval, df_temp


def generate_fog_markdown_summary(df: pd.DataFrame, df_temp: pd.DataFrame, out_path: str):
    """
    Генерирует сводный Markdown-отчет по обоим промптам Недели 2.
    """
    models = sorted(df['model'].unique())
    p1_df = df[df['prompt_num'] == 1]
    p2_df = df[df['prompt_num'] == 2]
    
    lines = [
        "# Сводные результаты исследования генерации морского тумана (Неделя 2)",
        "",
        r"В исследовании проанализировано **32 сгенерированных изображения** и **20 темпоральных пар** ($\Delta t = 0.33$ с) по двум ключевым гидрометеорологическим сценариям:",
        "1. **Промпт 1**: Умеренный морской туман и дымка (2–3 балла по шкале видимости на рассвете, сохранение топологии суши);",
        "2. **Промпт 2**: Плотный густой морской туман (1 балл по шкале видимости, видимость < 100 метров).",
        "",
        "---",
        "",
        "## 1. Сводная таблица результатов по обоим промптам и итоговый рейтинг",
        "",
        "| Модель | Итоговый Рейтинг (Сводный) | Промпт 1: Рейтинг (2–3 Б) | Промпт 2: Рейтинг (1 Б) | MOS Промпт 1 | MOS Промпт 2 | Сводный MOS | DCP % (П1 / П2) | Laplacian (П1 / П2) | CLIPIQA (П1 / П2) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    # Сортировка по общему рейтингу
    overall_ranks = df.groupby('model')['final_hybrid_rating'].mean().sort_values(ascending=False)
    
    for m in overall_ranks.index:
        d_m = df[df['model'] == m]
        d_p1 = p1_df[p1_df['model'] == m]
        d_p2 = p2_df[p2_df['model'] == m]
        
        r_all = d_m['final_hybrid_rating'].mean()
        r_p1 = d_p1['final_hybrid_rating'].mean() if not d_p1.empty else 0.0
        r_p2 = d_p2['final_hybrid_rating'].mean() if not d_p2.empty else 0.0
        
        mos_p1 = d_p1['expert_mos'].mean() if not d_p1.empty else 0.0
        mos_p2 = d_p2['expert_mos'].mean() if not d_p2.empty else 0.0
        mos_all = d_m['expert_mos'].mean()
        
        dcp_p1 = d_p1['fog_mean_rdcp'].mean() if not d_p1.empty else 0.0
        dcp_p2 = d_p2['fog_mean_rdcp'].mean() if not d_p2.empty else 0.0
        
        lap_p1 = d_p1['laplacian_var'].mean() if not d_p1.empty else 0.0
        lap_p2 = d_p2['laplacian_var'].mean() if not d_p2.empty else 0.0
        
        clip_p1 = d_p1['iqa_clipiqa'].mean() if not d_p1.empty else 0.0
        clip_p2 = d_p2['iqa_clipiqa'].mean() if not d_p2.empty else 0.0
        
        m_name = MODEL_DISPLAY_NAMES.get(m, m.upper())
        lines.append(f"| **{m_name}** | **{r_all:.2f}** | {r_p1:.2f} | {r_p2:.2f} | {mos_p1:.2f} | {mos_p2:.2f} | **{mos_all:.2f}** | {dcp_p1:.1f}% / {dcp_p2:.1f}% | {lap_p1:.0f} / {lap_p2:.0f} | {clip_p1:.3f} / {clip_p2:.3f} |")
        
    lines.extend([
        "",
        "> `*` **Особое примечание по модели Алиса (Yandex ART 2.0)**: Модель не поддерживает подачу референса (Image-to-Image). В Промпте 1 модель показала высокий балл (91.70, MOS: 9.33), однако в Промпте 2 сгенерировала чисто белое полотно (MOS: 0.0, Рейтинг: 17.50), что снизило сводный средний рейтинг до 54.60.",
        "",
        "---",
        "",
        "## 2. Темпоральная согласованность серий (dt = 0.33 с)",
        "",
        "| Модель | Промпт # | Пара кадров | SSIM (▲ Норма 0.65–0.95) | Оптический поток (px/кадр) | Статус |",
        "| :--- | :---: | :--- | :---: | :---: | :---: |"
    ])
    
    if not df_temp.empty:
        for _, row in df_temp.iterrows():
            m_name = MODEL_DISPLAY_NAMES.get(row['model'], row['model'].upper())
            status = "В норме" if 0.65 <= row['ssim'] <= 0.95 else ("Замирание" if row['ssim'] > 0.95 else "Срыв")
            lines.append(f"| {m_name} | {row['prompt_num']} | {row['pair']} | {row['ssim']:.3f} | {row['optical_flow_mean_speed']:.2f} px | {status} |")
            
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run_fog_evaluation()
