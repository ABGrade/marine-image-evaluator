"""
Главный скрипт оценки фотореалистичности сгенерированных морских изображений и видеорядов.

Выполняет:
1. Автоматическое извлечение кадров из видео (Nano Banana).
2. Пакетный расчет всех метрик из регламента:
   'Бесплатные и Доступные Критерии Оценки Морских Изображений.pdf'
   с сегментацией водной глади (водного зеркала).
3. Расчет 4 экспертных критериев оценки качества:
   - К1: Физика водной глади и текстура волн (База: 10)
   - К2: Соответствие погодным условиям и шкале пены (База: 10)
   - К3: Сенсорная аутентичность и ракурс (База: 10)
   - К4: Темпоральная согласованность серии (База: 10)
   - Расчет итогового MOS и Гибридного рейтинга (0.5 * Auto + 5.0 * MOS).
4. Оценку темпоральной консистентности серий изображений (dt = 0.33с).
5. Экспорт детализированных результатов в CSV и Markdown.
6. Построение 20 отдельных детальных аналитических графиков в папке 'plots/'.
"""

import os
import re
import sys
import glob
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Импорт модулей проекта
from extract_video_frames import extract_all_nano_banana_videos
from marine_metrics import (
    evaluate_image_full,
    compute_pairwise_temporal_consistency,
    NeuralIQAEngine,
    METRICS_SPEC
)

# Палитра моделей для единого цветового оформления всех графиков
MODEL_COLORS = {
    "flux": "#20639B",
    "gpt image 2": "#ED553B",
    "nano banana": "#2CA02C",
    "seedream": "#9C27B0",
    "wan": "#F6D55C",
    "yandex": "#FF5722",
}

DEFAULT_COLOR = "#3498DB"


def parse_image_filename(filepath: str) -> Dict[str, Any]:
    """
    Парсит имя файла вида '<модель> <номер_промпта>-<номер_в_серии>.<ext>'
    """
    p = Path(filepath)
    stem = p.stem
    parent_folder = p.parent.name
    
    match = re.search(r'(\d+)\s*[-_]\s*(\d+)', stem)
    if match:
        prompt_id = int(match.group(1))
        series_idx = int(match.group(2))
    else:
        m_single = re.search(r'(\d+)', stem)
        prompt_id = int(m_single.group(1)) if m_single else 1
        series_idx = 1
        
    return {
        "filepath": str(p),
        "filename": p.name,
        "model_folder": parent_folder,
        "prompt_id": prompt_id,
        "series_idx": series_idx
    }


def collect_all_images(base_dir: str = ".") -> List[Dict[str, Any]]:
    """
    Сканирует все директории моделей и собирает список изображений с метаданными.
    """
    base_path = Path(base_dir)
    target_folders = ["flux", "gpt image 2", "nano banana", "seedream", "wan", "yandex"]
    
    image_items = []
    for folder_name in target_folders:
        fpath = base_path / folder_name
        if not fpath.exists():
            continue
            
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            for img_file in fpath.glob(ext):
                meta = parse_image_filename(str(img_file))
                image_items.append(meta)
                
    image_items.sort(key=lambda x: (x["model_folder"], x["prompt_id"], x["series_idx"]))
    return image_items


# =====================================================================
# ГЕНЕРАЦИЯ ОТДЕЛЬНЫХ ИНДИВИДУАЛЬНЫХ ГРАФИКОВ ДЛЯ КАЖДОЙ МЕТРИКИ
# =====================================================================

def plot_single_metric(
    df: pd.DataFrame,
    metric_key: str,
    out_path: str,
    group_col: str = "model"
) -> bool:
    """
    Строит профессиональный отдельный график для конкретной метрики с:
    1. Наглядным указанием зоны нормы (axhspan + текстовая легенда);
    2. Обозначением направленности: '[▲ ВЫШЕ — ЛУЧШЕ]' / '[▼ НИЖЕ — ЛУЧШЕ]' / '[■ В ДИАПАЗОНЕ НОРМЫ]';
    3. Точными средними значениями над столбцами и индикатором попадания в норму;
    4. Пояснительной сноской о физическом значении и типах дефектов.
    """
    if df.empty or metric_key not in METRICS_SPEC:
        return False
        
    spec = METRICS_SPEC[metric_key]
    col_name = spec["column"]
    
    if col_name not in df.columns:
        print(f"  [WARN] Колонка {col_name} отсутствует в DataFrame.")
        return False
        
    valid_df = df.dropna(subset=[col_name]).copy()
    if valid_df.empty:
        return False
        
    models = sorted(valid_df[group_col].unique())
    means = valid_df.groupby(group_col)[col_name].mean().reindex(models)
    stds = valid_df.groupby(group_col)[col_name].std().reindex(models).fillna(0)
    
    # Настройка стиля
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#D5D8DC')
    
    # Цвета для моделей
    colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in models]
    
    # Расчет пределов оси Y
    max_val_with_err = float((means + stds).max()) if not means.empty else 1.0
    min_val = float(means.min()) if not means.empty else 0.0
    
    norm_type = spec.get("norm_type")
    norm_min = spec.get("norm_min")
    norm_max = spec.get("norm_max")
    
    y_top = max(max_val_with_err * 1.25, (norm_max * 1.18 if norm_max else max_val_with_err * 1.25), 1.0)
    y_bottom = min(0.0, min_val * 0.9)
    if spec.get("ylim"):
        if spec["ylim"][0] is not None:
            y_bottom = spec["ylim"][0]
        if spec["ylim"][1] is not None:
            y_top = max(spec["ylim"][1], max_val_with_err * 1.15)
            
    # Заливка зоны нормы
    if norm_type == "range" and norm_min is not None and norm_max is not None:
        ax.axhspan(norm_min, norm_max, color='#2ECC71', alpha=0.20, zorder=1, label=f"Норма: [{norm_min}, {norm_max}]")
        ax.axhline(norm_min, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)
        ax.axhline(norm_max, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)
        
        # Подписи границ нормы на правой оси
        ax.text(len(models) - 0.45, norm_min, f" Нижняя норма ({norm_min})", color='#196F3D', va='bottom', fontsize=8.5, fontweight='bold')
        ax.text(len(models) - 0.45, norm_max, f" Верхняя норма ({norm_max})", color='#196F3D', va='bottom', fontsize=8.5, fontweight='bold')
        
    elif norm_type == "min_threshold" and norm_min is not None:
        top_span = y_top if y_top is not None else 100.0
        ax.axhspan(norm_min, top_span, color='#2ECC71', alpha=0.18, zorder=1, label=f"Норма: >= {norm_min}")
        ax.axhline(norm_min, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)
        ax.text(len(models) - 0.45, norm_min, f" Порог нормы (>= {norm_min})", color='#196F3D', va='bottom', fontsize=8.5, fontweight='bold')
        
    elif norm_type == "max_threshold" and norm_max is not None:
        ax.axhspan(0, norm_max, color='#2ECC71', alpha=0.18, zorder=1, label=f"Норма: <= {norm_max}")
        ax.axhline(norm_max, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)
        ax.text(len(models) - 0.45, norm_max, f" Макс. порог (<= {norm_max})", color='#196F3D', va='bottom', fontsize=8.5, fontweight='bold')
        
    # Специфические зоны брака для отдельных физических метрик
    if metric_key == "laplacian_var":
        ax.axhspan(0, 100, color='#E74C3C', alpha=0.12, zorder=1)
        ax.text(0.02, 0.05, "Зона брака: размытие / 'пластиковая вода' (<100)", transform=ax.transAxes, color='#C0392B', fontsize=8.5, fontweight='bold')
    elif metric_key == "colorfulness":
        ax.axhspan(50, y_top, color='#E74C3C', alpha=0.10, zorder=1)
        ax.text(0.02, 0.92, "Зона брака: гипернасыщенность (>50)", transform=ax.transAxes, color='#C0392B', fontsize=8.5, fontweight='bold')
    elif metric_key == "entropy":
        ax.axhspan(0, 4.5, color='#E74C3C', alpha=0.12, zorder=1)
        ax.text(0.02, 0.05, "Зона брака: вырождение текстуры (<4.5)", transform=ax.transAxes, color='#C0392B', fontsize=8.5, fontweight='bold')
    elif metric_key == "temporal_ssim":
        ax.axhspan(0.995, 1.1, color='#E74C3C', alpha=0.15, zorder=1)
        ax.text(0.02, 0.92, "Зона брака: статичный дубликат кадра (>0.995)", transform=ax.transAxes, color='#C0392B', fontsize=8.5, fontweight='bold')
        
    # Построение столбцов
    has_multiple = (len(valid_df) > len(models))
    bars = ax.bar(
        models,
        means,
        yerr=stds if has_multiple else None,
        capsize=5,
        error_kw=dict(elinewidth=1.4, ecolor='#2C3E50', capthick=1.4),
        color=colors,
        alpha=0.88,
        edgecolor='#2C3E50',
        linewidth=1.3,
        zorder=3,
        width=0.55
    )
    
    ax.set_ylim(y_bottom, y_top)
    
    # Форматирование значений над столбцами и проверка попадания в норму
    for bar, model_name in zip(bars, models):
        val = bar.get_height()
        if np.isnan(val):
            continue
            
        std_val = stds.get(model_name, 0.0) if has_multiple else 0.0
        top_err_val = val + std_val
        
        in_norm = True
        if norm_type == "range" and norm_min is not None and norm_max is not None:
            in_norm = (norm_min <= val <= norm_max)
        elif norm_type == "min_threshold" and norm_min is not None:
            in_norm = (val >= norm_min)
        elif norm_type == "max_threshold" and norm_max is not None:
            in_norm = (val <= norm_max)
            
        # Формат чисел
        if abs(val) >= 100:
            val_str = f"{val:.1f}"
        elif abs(val) >= 10:
            val_str = f"{val:.2f}"
        else:
            val_str = f"{val:.3f}" if val < 1.0 else f"{val:.2f}"
            
        status_label = "[В норме]" if in_norm else "[Вне нормы]"
        text_color = "#145A32" if in_norm else "#900C3F"
        box_edge = "#27AE60" if in_norm else "#E74C3C"
        box_bg = "#E8F8F5" if in_norm else "#FDEDEC"
        
        offset = (y_top - y_bottom) * 0.035
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            top_err_val + offset,
            f"{val_str}\n{status_label}",
            ha='center',
            va='bottom',
            fontsize=8.5,
            fontweight='bold',
            color=text_color,
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=box_bg, edgecolor=box_edge, alpha=0.95, linewidth=1.0)
        )
        
    ax.set_ylabel(spec.get("unit", "Значение"), fontsize=11, fontweight='bold', color='#2C3E50')
    ax.tick_params(axis='x', rotation=15, labelsize=10.5)
    ax.tick_params(axis='y', labelsize=9.5)
    
    fig.suptitle(f"{spec['title']}", fontsize=14.5, fontweight='bold', color='#1A252F', y=0.98)
    
    dir_label = spec.get("direction_label", "")
    norm_label = spec.get("norm_label", "")
    subtitle_text = f"Критерий качества: {dir_label}  |  {norm_label}"
    ax.set_title(subtitle_text, fontsize=10.5, fontweight='bold', color='#2980B9', pad=14)
    
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#BDC3C7', fontsize=9.5)
    
    desc = spec.get("description", "")
    defect = spec.get("defect_note", "")
    footer_text = f"• Описание: {desc}\n• Физическая интерпретация: {defect}"
    fig.text(
        0.5, -0.02,
        footer_text,
        ha='center',
        va='top',
        fontsize=8.5,
        color='#4A5568',
        bbox=dict(boxstyle="round,pad=0.45", facecolor='#F4F6F7', edgecolor='#D5D8DC', linewidth=0.8)
    )
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Индивидуальный график сохранен: {out_path}")
    return True


def plot_validation_pass_rates(df_single: pd.DataFrame, out_path: str):
    """
    Строит отдельный сравнительный график успешности валидации (Этап 1 и Этап 2).
    """
    if df_single.empty:
        return
        
    models = sorted(df_single["model"].unique())
    stage1_rates = df_single.groupby("model")["stage1_passed"].mean().reindex(models) * 100.0
    stage2_rates = df_single.groupby("model")["stage2_passed"].mean().reindex(models) * 100.0
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#D5D8DC')
    
    x = np.arange(len(models))
    width = 0.36
    
    ax.axhspan(80, 100, color='#2ECC71', alpha=0.18, zorder=1, label="Целевая норма без брака [80–100%]")
    ax.axhline(80, color='#27AE60', linestyle='--', linewidth=1.4, alpha=0.8)
    ax.axhline(100, color='#1E8449', linestyle='-', linewidth=1.5, alpha=0.8)
    
    rects1 = ax.bar(x - width/2, stage1_rates, width, label='Этап 1: Экспресс-отбор (OpenCV)', color='#2980B9', edgecolor='#1B4F72', linewidth=1.2, alpha=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, stage2_rates, width, label='Этап 2: Физико-доменный контроль (RDCP/RMS)', color='#27AE60', edgecolor='#196F3D', linewidth=1.2, alpha=0.9, zorder=3)
    
    ax.set_ylabel('% кадров без брака', fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_ylim(0, 125)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold')
    
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 2.0, f"{h:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1B4F72')
        
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 2.0, f"{h:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#196F3D')
        
    fig.suptitle("Процент успешного прохождения валидации по этапам (Pass Rates %)", fontsize=14.5, fontweight='bold', y=0.98)
    ax.set_title("Критерий качества: [▲ ВЫШЕ — ЛУЧШЕ]  |  Норма: 100% (Допустимо: >= 80%)", fontsize=10.5, fontweight='bold', color='#2980B9', pad=14)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#BDC3C7', fontsize=9.5)
    
    footer_text = "• Этап 1 проверяет отсутствие размытия, гипернасыщения и монотонности заливки.\n• Этап 2 проверяет физику тумана по закону Кошмидера (RDCP градиент) и естественный RMS-контраст."
    fig.text(0.5, -0.02, footer_text, ha='center', va='top', fontsize=8.5, color='#4A5568', bbox=dict(boxstyle="round,pad=0.45", facecolor='#F4F6F7', edgecolor='#D5D8DC', linewidth=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Индивидуальный график сохранен: {out_path}")


def plot_prompt_breakdown(df_single: pd.DataFrame, out_path: str):
    """
    Строит график сравнения комплексного балла по 4 экспериментальным промптам.
    """
    if df_single.empty:
        return
        
    prompt_titles = {
        1: "Промпт 1: f/11 RAW (10м)",
        2: "Промпт 2: Смартфон (пирс)",
        3: "Промпт 3: Видеорегистратор",
        4: "Промпт 4: CCTV 2K"
    }
    
    models = sorted(df_single["model"].unique())
    prompts = sorted(df_single["prompt_id"].unique())
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=250)
    fig.patch.set_facecolor('#FAFBFC')
    ax.set_facecolor('#FFFFFF')
    ax.grid(True, linestyle='--', alpha=0.5, color='#D5D8DC')
    
    x = np.arange(len(models))
    width = 0.18
    p_colors = ['#2980B9', '#E67E22', '#27AE60', '#8E44AD']
    
    ax.axhspan(60, 100, color='#2ECC71', alpha=0.15, zorder=1, label="Норма качества (>= 60.0)")
    ax.axhline(60, color='#27AE60', linestyle='--', linewidth=1.3, alpha=0.7)
    
    for i, pid in enumerate(prompts):
        sub_df = df_single[df_single["prompt_id"] == pid]
        scores = sub_df.groupby("model")["composite_score"].mean().reindex(models).fillna(0)
        pos = x + (i - len(prompts)/2 + 0.5) * width
        rects = ax.bar(pos, scores, width, label=prompt_titles.get(pid, f"Промпт {pid}"), color=p_colors[i % len(p_colors)], alpha=0.9, edgecolor='black', linewidth=1.1, zorder=3)
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.2, f"{h:.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
                
    ax.set_ylabel('Комплексный балл (0–100)', fontsize=11, fontweight='bold', color='#2C3E50')
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight='bold')
    
    fig.suptitle("Сравнение моделей в разрезе 4 экспериментальных промптов", fontsize=14.5, fontweight='bold', y=0.98)
    ax.set_title("Критерий качества: [▲ ВЫШЕ — ЛУЧШЕ]  |  Норма: >= 60.0 (Высокое: >= 75.0)", fontsize=10.5, fontweight='bold', color='#2980B9', pad=14)
    ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#BDC3C7', fontsize=9)
    
    footer_text = "• Промпты моделируют разные ракурсы и сенсоры: 1 (камера f/11 RAW), 2 (смартфон 16:9), 3 (видеорегистратор в движении), 4 (CCTV 2K)."
    fig.text(0.5, -0.02, footer_text, ha='center', va='top', fontsize=8.5, color='#4A5568', bbox=dict(boxstyle="round,pad=0.45", facecolor='#F4F6F7', edgecolor='#D5D8DC', linewidth=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Индивидуальный график сохранен: {out_path}")


def generate_all_individual_plots(
    df_single: pd.DataFrame,
    df_temporal: pd.DataFrame,
    out_dir: str = "plots"
) -> List[str]:
    """
    Генерирует полный комплект отдельных графиков высокого разрешения для всех метрик и критериев.
    """
    os.makedirs(out_dir, exist_ok=True)
    generated_files = []
    
    print("\n" + "=" * 70)
    print(" ГЕНЕРАЦИЯ ОТДЕЛЬНЫХ ИНДИВИДУАЛЬНЫХ ГРАФИКОВ МЕТРИК С УКАЗАНИЕМ НОРМ")
    print("=" * 70)
    
    single_metrics = [
        ("composite_score", "01_composite_score.png"),
        ("laplacian_var", "02_laplacian_variance.png"),
        ("colorfulness", "03_colorfulness.png"),
        ("entropy", "04_shannon_entropy.png"),
        ("rms_contrast", "05_rms_contrast.png"),
        ("iqa_clipiqa", "06_clipiqa.png"),
        ("iqa_hyperiqa", "07_hyperiqa.png"),
        ("iqa_musiq", "08_musiq.png"),
        ("fog_mean_rdcp", "12_fog_rdcp.png"),
        ("foam_coverage_pct", "13_foam_coverage.png"),
        ("k1_water_physics", "15_k1_water_physics.png"),
        ("k2_weather_foam", "16_k2_weather_foam.png"),
        ("k3_sensor_authenticity", "17_k3_sensor_authenticity.png"),
        ("k4_temporal_consistency", "18_k4_temporal_consistency.png"),
        ("expert_mos", "19_expert_mos_score.png"),
        ("final_hybrid_rating", "20_final_hybrid_rating.png"),
    ]
    
    for metric_key, fname in single_metrics:
        fpath = os.path.join(out_dir, fname)
        if plot_single_metric(df_single, metric_key, fpath):
            generated_files.append(fpath)
            
    if not df_temporal.empty:
        temporal_metrics = [
            ("temporal_ssim", "09_temporal_ssim.png"),
            ("optical_flow_mean_speed", "10_optical_flow_speed.png"),
        ]
        for metric_key, fname in temporal_metrics:
            fpath = os.path.join(out_dir, fname)
            if plot_single_metric(df_temporal, metric_key, fpath):
                generated_files.append(fpath)
                
    fpath_val = os.path.join(out_dir, "11_validation_pass_rates.png")
    plot_validation_pass_rates(df_single, fpath_val)
    generated_files.append(fpath_val)
    
    fpath_prompt = os.path.join(out_dir, "14_prompt_breakdown.png")
    plot_prompt_breakdown(df_single, fpath_prompt)
    generated_files.append(fpath_prompt)
    
    print(f"Всего успешно создано отдельных графиков: {len(generated_files)} в директории '{out_dir}/'")
    return generated_files


def generate_comparison_plots(df_single: pd.DataFrame, df_temporal: pd.DataFrame, out_path: str):
    """
    Генерирует как сводный многопанельный график (marine_comparison_chart.png),
    так и полную серию отдельных графиков в папке 'plots/'.
    """
    if df_single.empty:
        return
        
    plots_dir = os.path.join(os.path.dirname(out_path) or ".", "plots")
    generate_all_individual_plots(df_single, df_temporal, out_dir=plots_dir)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=200)
    fig.suptitle("Сравнительный анализ генеративных моделей по морским метрикам и критериям", fontsize=16, fontweight='bold', y=0.98)
    
    models = sorted(df_single["model"].unique())
    colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in models]
    
    # 1. Final Hybrid Rating
    ax = axes[0, 0]
    rating_means = df_single.groupby("model")["final_hybrid_rating"].mean().reindex(models)
    ax.axhspan(75, 100, color='#2ECC71', alpha=0.18, label='Норма [>=75]')
    bars = ax.bar(models, rating_means, color=colors, alpha=0.88, edgecolor='black', zorder=3)
    ax.set_title("Итоговый гибридный рейтинг (0–100)\n[▲ Выше лучше | Норма: >=75]", fontweight='bold', fontsize=10.5, color='#2980B9')
    ax.set_ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc='upper right', fontsize=8)
    
    # 2. Expert MOS Score
    ax = axes[0, 1]
    mos_means = df_single.groupby("model")["expert_mos"].mean().reindex(models)
    ax.axhspan(8.0, 10.0, color='#2ECC71', alpha=0.18, label='Норма [>=8.0]')
    bars = ax.bar(models, mos_means, color=colors, alpha=0.88, edgecolor='black', zorder=3)
    ax.set_title("Экспертный балл MOS (К1–К4, 0–10)\n[▲ Выше лучше | Норма: >=8.0]", fontweight='bold', fontsize=10.5, color='#2980B9')
    ax.set_ylim(0, 11.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc='upper right', fontsize=8)
    
    # 3. Laplacian Variance
    ax = axes[0, 2]
    lap_means = df_single.groupby("model")["laplacian_var"].mean().reindex(models)
    ax.axhspan(100, 1200, color='#2ECC71', alpha=0.18, label='Норма [100, 1200]')
    bars = ax.bar(models, lap_means, color=colors, alpha=0.88, edgecolor='black', zorder=3)
    ax.set_title("Дисперсия Лапласиана (Резкость волн)\n[■ В диапазоне | Норма: 100–1200]", fontweight='bold', fontsize=10.5, color='#2980B9')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.0f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc='upper right', fontsize=8)
    
    # 4. Hasler Colorfulness
    ax = axes[1, 0]
    col_means = df_single.groupby("model")["colorfulness"].mean().reindex(models)
    ax.axhspan(15, 50, color='#2ECC71', alpha=0.18, label='Норма [15, 50]')
    bars = ax.bar(models, col_means, color=colors, alpha=0.88, edgecolor='black', zorder=3)
    ax.set_title("Красочность Hasler (Палитра M3)\n[■ В диапазоне | Норма: 15–50]", fontweight='bold', fontsize=10.5, color='#2980B9')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.legend(loc='upper right', fontsize=8)
    
    # 5. CLIPIQA Score
    ax = axes[1, 1]
    if "iqa_clipiqa" in df_single.columns and not df_single["iqa_clipiqa"].isna().all():
        clip_means = df_single.groupby("model")["iqa_clipiqa"].mean().reindex(models)
        ax.axhspan(0.60, 1.0, color='#2ECC71', alpha=0.18, label='Норма [>= 0.60]')
        bars = ax.bar(models, clip_means, color=colors, alpha=0.88, edgecolor='black', zorder=3)
        ax.set_title("CLIPIQA (Перцептивная естественность)\n[▲ Выше лучше | Норма: >= 0.60]", fontweight='bold', fontsize=10.5, color='#2980B9')
        ax.set_ylim(0, 1.05)
        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval):
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "CLIPIQA не рассчитывалась", ha='center', va='center')
    ax.tick_params(axis='x', rotation=20)
    
    # 6. Temporal SSIM
    ax = axes[1, 2]
    if not df_temporal.empty and "ssim" in df_temporal.columns:
        ssim_means = df_temporal.groupby("model")["ssim"].mean().reindex(models)
        ax.axhspan(0.65, 0.95, color='#2ECC71', alpha=0.18, label='Норма [0.65, 0.95]')
        bars = ax.bar(models, ssim_means, color=colors, alpha=0.88, edgecolor='black', zorder=3)
        ax.set_title("Темпоральный SSIM серии (dt=0.33с)\n[■ В окне динамики | Норма: 0.65–0.95]", fontweight='bold', fontsize=10.5, color='#2980B9')
        ax.set_ylim(0, 1.1)
        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval):
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "Темпоральные серии отсутствуют", ha='center', va='center')
    ax.tick_params(axis='x', rotation=20)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [OK] Сводный обзорный график сохранен в {out_path}")


# =====================================================================
# ФОРМИРОВАНИЕ ИТОГОВОГО ОТЧЕТА В MARKDOWN
# =====================================================================

def generate_markdown_report(df_single: pd.DataFrame, df_temporal: pd.DataFrame, out_path: str, plots_dir: str = "plots"):
    """
    Формирует красивый и подробный Markdown-отчет со ссылками на отдельные графики и нормами.
    """
    model_stats = df_single.groupby("model").agg(
        total_images=("filename", "count"),
        final_rating=("final_hybrid_rating", "mean"),
        expert_mos=("expert_mos", "mean"),
        k1_water=("k1_water_physics", "mean"),
        k2_weather=("k2_weather_foam", "mean"),
        k3_sensor=("k3_sensor_authenticity", "mean"),
        k4_temporal=("k4_temporal_consistency", "mean"),
        avg_composite_score=("composite_score", "mean"),
        avg_colorfulness=("colorfulness", "mean"),
        avg_laplacian=("laplacian_var", "mean"),
        avg_foam=("foam_coverage_pct", "mean"),
        avg_fog_rdcp=("fog_mean_rdcp", "mean"),
        avg_clipiqa=("iqa_clipiqa", "mean") if "iqa_clipiqa" in df_single.columns else ("filename", lambda x: 0),
        stage1_pass_rate=("stage1_passed", lambda x: np.mean(x) * 100),
    ).reset_index()
    
    if not df_temporal.empty and "ssim" in df_temporal.columns:
        t_stats = df_temporal.groupby("model").agg(
            avg_temporal_ssim=("ssim", "mean"),
            avg_optical_speed=("optical_flow_mean_speed", "mean")
        ).reset_index()
        model_stats = pd.merge(model_stats, t_stats, on="model", how="left")
    else:
        model_stats["avg_temporal_ssim"] = np.nan
        model_stats["avg_optical_speed"] = np.nan
        
    model_stats = model_stats.sort_values(by="final_rating", ascending=False)
    
    prompt_stats = df_single.groupby(["prompt_id", "model"]).agg(
        final_rating=("final_hybrid_rating", "mean"),
        mos=("expert_mos", "mean"),
        k1=("k1_water_physics", "mean"),
        k2=("k2_weather_foam", "mean"),
        k3=("k3_sensor_authenticity", "mean"),
        k4=("k4_temporal_consistency", "mean"),
        avg_score=("composite_score", "mean"),
        avg_laplacian=("laplacian_var", "mean"),
        avg_colorfulness=("colorfulness", "mean"),
        avg_foam=("foam_coverage_pct", "mean"),
        avg_fog=("fog_mean_rdcp", "mean")
    ).reset_index()
    
    lines = []
    lines.append("# Итоговый Сводный Отчет по Оценке Морских Изображений\n")
    lines.append("> Оценка проведена на основе строгой трехэтапной методологии из регламента *«Бесплатные и Доступные Критерии Оценки Морских Изображений.pdf»* и системы 4 экспертных критериев.\n")
    
    lines.append("## 1. Общий Рейтинг Моделей\n")
    lines.append("| Ранг | Модель | Итоговый Рейтинг (0–100)<br>*(▲ Выше лучше, Норма ≥75)* | Экспертный MOS (0–10)<br>*(▲ Выше лучше, Норма ≥8.0)* | К1: Физика воды | К2: Погода/Пена | К3: Сенсор | К4: Темпоральность | Комплексный балл (Авто) | Резкость (Laplacian) | Пена FCR (%) | Туман RDCP |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for rank, (_, row) in enumerate(model_stats.iterrows(), 1):
        lines.append(
            f"| **#{rank}** | **{row['model']}** | **{row['final_rating']:.2f}** | "
            f"**{row['expert_mos']:.2f}** | {row['k1_water']:.1f} | {row['k2_weather']:.1f} | "
            f"{row['k3_sensor']:.1f} | {row['k4_temporal']:.1f} | {row['avg_composite_score']:.1f} | "
            f"{row['avg_laplacian']:.0f} | {row['avg_foam']:.2f}% | {row['avg_fog_rdcp']:.1f} |"
        )
        
    lines.append("\n---\n")
    lines.append("## 2. Каталог Отдельных Графиков Метрик и Критериев (20 графиков)\n")
    lines.append("Для детального анализа каждой метрики сформированы отдельные изолированные графики с цветовой подсветкой нормы и маркерами оптимизации:\n")
    
    plot_items = [
        ("20_final_hybrid_rating.png", "Итоговый гибридный рейтинг модели (0–100)", "▲ Чем выше — тем лучше", "≥ 75.0 (Высокий рейтинг: ≥ 80.0)"),
        ("19_expert_mos_score.png", "Итоговый экспертный балл MOS (К1–К4, 0–10)", "▲ Чем выше — тем лучше", "≥ 8.0 (из 10)"),
        ("15_k1_water_physics.png", "Критерий 1: Физика водной глади и текстура волн", "▲ Чем выше — тем лучше", "≥ 8.0 (из 10)"),
        ("16_k2_weather_foam.png", "Критерий 2: Соответствие погоде и шкале пены", "▲ Чем выше — тем лучше", "10.0 (из 10)"),
        ("17_k3_sensor_authenticity.png", "Критерий 3: Сенсорная аутентичность и ракурс", "▲ Чем выше — тем лучше", "≥ 8.0 (из 10)"),
        ("18_k4_temporal_consistency.png", "Критерий 4: Темпоральная согласованность серии", "▲ Чем выше — тем лучше", "≥ 8.0 (из 10)"),
        ("01_composite_score.png", "Комплексный авто-балл фотореалистичности", "▲ Чем выше — тем лучше", "≥ 60.0 (Высокое: ≥ 75.0)"),
        ("02_laplacian_variance.png", "Дисперсия Лапласиана (Резкость волн)", "■ Оптимально в диапазоне нормы", "[100, 1200] (<100 — пластик, >1200 — шум)"),
        ("03_colorfulness.png", "Индекс красочности Хаслера-Зюсштрунка", "■ Оптимально в диапазоне нормы", "[15, 50] (>50 — гипернасыщение)"),
        ("04_shannon_entropy.png", "Информационная энтропия Шеннона", "■ Оптимально в диапазоне нормы", "[6.0, 7.8] (<4.5 — вырождение текстуры)"),
        ("05_rms_contrast.png", "Среднеквадратичный RMS Контраст", "■ Оптимально в диапазоне нормы", "[0.15, 0.45] (<0.12 — блеклость)"),
        ("06_clipiqa.png", "CLIPIQA (Перцептивная естественность)", "▲ Чем выше — тем лучше", "≥ 0.60 (Отлично: ≥ 0.75)"),
        ("07_hyperiqa.png", "HyperIQA (Отсутствие локальных искажений)", "▲ Чем выше — тем лучше", "≥ 0.50 (Отлично: ≥ 0.70)"),
        ("08_musiq.png", "MUSIQ (Трансформерное качество ViT)", "▲ Чем выше — тем лучше", "≥ 60.0 (Отлично: ≥ 70.0)"),
        ("09_temporal_ssim.png", "Темпоральный SSIM серии (dt=0.33с)", "■ Оптимально в окне динамики", "[0.65, 0.95] (>0.995 — дубликат)"),
        ("10_optical_flow_speed.png", "Скорость оптического потока Фарнебека", "■ Оптимально в диапазоне нормы", "[0.5, 15.0] px/кадр (<0.1 — застыл)"),
        ("11_validation_pass_rates.png", "Процент успешной валидации (Pass Rates)", "▲ Чем выше — тем лучше", "100% (Допустимо: ≥ 80%)"),
        ("12_fog_rdcp.png", "Плотность темного канала RDCP на воде (Туман)", "▼ Чем ниже — тем чище атмосфера", "< 20.0 (Ясно: 0 баллов)"),
        ("13_foam_coverage.png", "Коэффициент покрытия пеной на воде (FCR %)", "■ По баллам Бофорта", "[0, 3.5%] для спокойного моря"),
        ("14_prompt_breakdown.png", "Сравнение моделей по 4 промптам", "▲ Чем выше — тем лучше", "≥ 60.0 по всем ракурсам"),
    ]
    
    lines.append("| Файл графика | Метрика / Критерий | Направленность | Нормативный интервал / Порог |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for fname, mtitle, dir_t, norm_t in plot_items:
        lines.append(f"| [`{fname}`]({plots_dir}/{fname}) | **{mtitle}** | `{dir_t}` | `{norm_t}` |")
        
    lines.append("\n---\n")
    lines.append("## 3. Детализация по 4 Экспериментальным Промптам\n")
    
    prompt_labels = {
        1: "Промпт 1 (Камера 10м f/11 RAW, 4K)",
        2: "Промпт 2 (Смартфон с пирса 16:9)",
        3: "Промпт 3 (Видеорегистратор на шоссе)",
        4: "Промпт 4 (Камера наблюдения CCTV 2K)"
    }
    
    lines.append("| Эксперимент | Модель | Итоговый Рейтинг | MOS | К1: Физика | К2: Погода | К3: Сенсор | К4: Темп. | Комплексный Балл | Резкость | Пена FCR | Туман RDCP |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, row in prompt_stats.iterrows():
        pid = int(row['prompt_id'])
        plabel = prompt_labels.get(pid, f"Промпт {pid}")
        lines.append(
            f"| {plabel} | **{row['model']}** | **{row['final_rating']:.2f}** | **{row['mos']:.2f}** | "
            f"{row['k1']:.1f} | {row['k2']:.1f} | {row['k3']:.1f} | {row['k4']:.1f} | "
            f"{row['avg_score']:.1f} | {row['avg_laplacian']:.0f} | {row['avg_foam']:.2f}% | {row['avg_fog']:.1f} |"
        )
        
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [OK] Итоговый Markdown отчет сформирован: {out_path}")


# =====================================================================
# ГЛАВНЫЙ КОНВЕЙЕР
# =====================================================================

def run_full_evaluation(
    base_dir: str = ".",
    use_neural_iqa: bool = True,
    use_gpu: bool = True,
    plots_only: bool = False
):
    print("=" * 75)
    print(" КОМПЛЕКСНАЯ ОЦЕНКА МОРСКИХ ИЗОБРАЖЕНИЙ ПО МЕТРИКАМ И 4 ЭКСПЕРТНЫМ КРИТЕРИЯМ")
    print("=" * 75)
    
    csv_single_path = os.path.join(base_dir, "marine_evaluation_results.csv")
    csv_temporal_path = os.path.join(base_dir, "marine_temporal_results.csv")
    md_summary_path = os.path.join(base_dir, "marine_evaluation_summary.md")
    chart_path = os.path.join(base_dir, "marine_comparison_chart.png")
    plots_dir = os.path.join(base_dir, "plots")
    
    # 1. Сбор изображений
    print("\n[ШАГ 1/4] Сканирование каталогов моделей...")
    items = collect_all_images(base_dir)
    print(f"  Всего обнаружено изображений для валидации: {len(items)}")
    
    # 2. Инициализация нейросетевых IQA метрик (PyIQA)
    iqa_engine = None
    if use_neural_iqa and not plots_only:
        print("\n[ШАГ 2/4] Инициализация нейросетевых IQA-моделей (PyIQA: CLIPIQA, HyperIQA, MUSIQ)...")
        try:
            iqa_engine = NeuralIQAEngine(use_gpu=use_gpu, metrics=['clipiqa', 'hyperiqa', 'musiq'])
        except Exception as e:
            print(f"  [WARN] Ошибка инициализации PyIQA: {e}. Переключение на статистические метрики.")
            iqa_engine = None
    else:
        print("\n[ШАГ 2/4] Нейросетевой блок пропущен.")
        
    # Загрузка ранее вычисленных IQA-метрик для ускорения
    iqa_cache = {}
    if os.path.exists(csv_single_path):
        try:
            df_old = pd.read_csv(csv_single_path)
            if 'iqa_clipiqa' in df_old.columns:
                for _, r in df_old.iterrows():
                    fname = str(r.get('filename', ''))
                    if fname:
                        iqa_cache[fname] = {
                            'iqa_clipiqa': r.get('iqa_clipiqa', None),
                            'iqa_hyperiqa': r.get('iqa_hyperiqa', None),
                            'iqa_musiq': r.get('iqa_musiq', None)
                        }
        except Exception:
            pass

    # 3. Расчет метрик
    print("\n[ШАГ 3/4] Расчет математических, физико-доменных и экспертных метрик...")
    detailed_results = []
    
    for idx, item in enumerate(items, 1):
        print(f"  [{idx:02d}/{len(items):02d}] Модель: {item['model_folder']:<12} | Файл: {item['filename']}")
        try:
            res = evaluate_image_full(
                item["filepath"],
                iqa_engine=iqa_engine,
                prompt_id=item["prompt_id"],
                model_name=item["model_folder"]
            )
            # Подмешиваем кэшированные IQA если iqa_engine не вызывался
            if item["filename"] in iqa_cache:
                for k, v in iqa_cache[item["filename"]].items():
                    if res.get(k) is None and pd.notna(v):
                        res[k] = v
            res["model"] = item["model_folder"]
            res["prompt_id"] = item["prompt_id"]
            res["series_idx"] = item["series_idx"]
            detailed_results.append(res)
        except Exception as e:
            print(f"    [ERROR] Ошибка оценки {item['filepath']}: {e}")
            
    df_single = pd.DataFrame(detailed_results)
    
    # 4. Темпоральные серии
    print("\n[ШАГ 4/4] Оценка темпоральной согласованности серий (dt = 0.33с)...")
    temporal_results = []
    
    grouped = {}
    for item in items:
        key = (item["model_folder"], item["prompt_id"])
        grouped.setdefault(key, []).append(item)
        
    for (model, prompt_id), series_items in grouped.items():
        if len(series_items) < 2:
            continue
            
        series_items.sort(key=lambda x: x["series_idx"])
        for i in range(len(series_items) - 1):
            item1 = series_items[i]
            item2 = series_items[i+1]
            
            img1_bgr = cv2.imread(item1["filepath"])
            img2_bgr = cv2.imread(item2["filepath"])
            if img1_bgr is None or img2_bgr is None:
                continue
                
            img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
            
            t_res = compute_pairwise_temporal_consistency(img1_rgb, img2_rgb)
            t_res["model"] = model
            t_res["prompt_id"] = prompt_id
            t_res["pair"] = f"{item1['series_idx']} -> {item2['series_idx']}"
            t_res["frame_from"] = item1["filename"]
            t_res["frame_to"] = item2["filename"]
            temporal_results.append(t_res)
            
    df_temporal = pd.DataFrame(temporal_results) if temporal_results else pd.DataFrame()
    
    # Экспорт данных
    df_single.to_csv(csv_single_path, index=False, encoding="utf-8-sig")
    if not df_temporal.empty:
        df_temporal.to_csv(csv_temporal_path, index=False, encoding="utf-8-sig")
        
    # Построение графиков
    generate_comparison_plots(df_single, df_temporal, chart_path)
    
    # Формирование Markdown отчета
    generate_markdown_report(df_single, df_temporal, md_summary_path, plots_dir="plots")
    
    print("\n" + "=" * 75)
    print(" ОЦЕНКА УСПЕШНО ЗАВЕРШЕНА!")
    print(f"  - Подробные результаты по кадрам: {csv_single_path}")
    print(f"  - Результаты темпоральной серии: {csv_temporal_path}")
    print(f"  - Итоговый аналитический отчет: {md_summary_path}")
    print(f"  - Сводный обзорный график:      {chart_path}")
    print(f"  - Каталог отдельных графиков:   {plots_dir}/ (20 файлов)")
    print("=" * 75)
    
    return df_single, df_temporal


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Комплексная оценка морских изображений")
    parser.add_argument("--dir", default=".", help="Рабочая директория с папками моделей")
    parser.add_argument("--no-gpu", action="store_true", help="Принудительный запуск на CPU")
    parser.add_argument("--no-neural", action="store_true", help="Отключить нейросетевые IQA метрики")
    parser.add_argument("--plots-only", action="store_true", help="Перестроить графики без пересчета IQA")
    args = parser.parse_args()
    
    run_full_evaluation(
        base_dir=args.dir,
        use_neural_iqa=not args.no_neural,
        use_gpu=not args.no_gpu,
        plots_only=args.plots_only
    )
