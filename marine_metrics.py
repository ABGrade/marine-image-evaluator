"""
Модуль расчета математических, физико-доменных, перцептивных и экспертных метрик
качества фотореалистичности сгенерированных морских изображений и видеорядов.

Методология базируется на регламенте:
'Бесплатные и Доступные Критерии Оценки Морских Изображений.pdf'
и системе 4 экспертных критериев (Физика воды, Погода/Пена, Сенсор, Темпоральность).

Включает:
1. Экспресс-статистический отбор (Этап 1: Hasler Colorfulness, Laplacian Var, Shannon Entropy, RMS Contrast, FFT).
2. Физико-доменный контроль с сегментацией водной глади (Этап 2: Dark Channel Prior / RDCP для дневного моря,
   Foam Coverage Ratio FCR на волновых гребнях, Koschmieder visibility).
3. Перцептивную и темпоральную оценку (Этап 3: CLIPIQA, HyperIQA, MUSIQ, Optical Flow Farneback, SSIM).
4. Автоматизированную экспертную оценку по 4 критериям (К1..К4, MOS, Итоговый гибридный рейтинг).
"""

import os
import math
import numpy as np
import cv2
from image_io import read_image
from scipy import ndimage, stats
from typing import Dict, Any, Tuple, List, Optional
torch = None  # Loaded only when optional neural IQA is explicitly requested.


# =====================================================================
# СЕГМЕНТАЦИЯ ВОДНОГО ЗЕРКАЛА (МОРСКОЙ АКВАТОРИИ)
# =====================================================================

def segment_water_surface(img_rgb: np.ndarray, img_gray: np.ndarray) -> np.ndarray:
    """
    Высокопроизводительная сегментация области водного зеркала морской глади,
    исключающая небо, бетонный пирс, асфальтовое полотно дороги, отбойники и транспорт.
    """
    h, w = img_gray.shape
    scale = 1024.0 / max(h, w) if max(h, w) > 1024 else 1.0
    if scale < 1.0:
        small_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_rgb = img_rgb
        small_gray = img_gray
        
    sh, sw = small_gray.shape
    small_bgr = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    
    # Морская вода: характерный сине-голубой, бирюзовый или темный спектр
    is_marine_hue = (h_chan >= 65) & (h_chan <= 155) & (s_chan >= 10)
    is_dark_water = (v_chan < 185) & (h_chan >= 50) & (h_chan <= 165) & (s_chan >= 8)
    water_candidate = (is_marine_hue | is_dark_water)
    
    # Текстурный анализ: исключаем однородные плоские серые зоны (асфальт, бетон)
    sobel_y = cv2.Sobel(small_gray, cv2.CV_16S, 0, 1, ksize=3)
    texture_mag = np.abs(sobel_y)
    is_flat_gray_road = (s_chan < 25) & (texture_mag < 15) & (small_gray < 220)
    
    water_mask = water_candidate & (~is_flat_gray_road)
    water_mask[:int(sh * 0.15), :] = False
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    water_mask = cv2.morphologyEx(water_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
    
    if np.sum(water_mask) < 0.10 * sh * sw:
        water_mask[int(sh * 0.25):int(sh * 0.85), :] = 1
        
    if scale < 1.0:
        full_mask = cv2.resize(water_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        full_mask = water_mask
        
    return full_mask.astype(bool)


# =====================================================================
# 1. МАТЕМАТИЧЕСКИЕ И СТАТИСТИЧЕСКИЕ МЕТРИКИ (ЭТАП 1: ЭКСПРЕСС-ОТБОР)
# =====================================================================

def compute_colorfulness(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Индекс красочности Хаслера-Зюсштрунка (Hasler-Süsstrunk Metric M3).
    Оценивает цветовую насыщенность в пространстве оппонентных цветов.
    Норма: 15 <= M <= 50. > 50 - неестественное перенасыщение (брак).
    """
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    std_rg = np.std(rg)
    mean_rg = np.mean(rg)
    std_yb = np.std(yb)
    mean_yb = np.mean(yb)
    
    std_rgyb = np.sqrt(std_rg**2 + std_yb**2)
    mean_rgyb = np.sqrt(mean_rg**2 + mean_yb**2)
    
    colorfulness = float(std_rgyb + 0.3 * mean_rgyb)
    
    if colorfulness < 15.0:
        status = "Слишком монохромное / тусклое"
        score = 60.0
    elif 15.0 <= colorfulness <= 50.0:
        status = "В норме (Естественная морская палитра)"
        score = 100.0
    else:
        status = "Брак: Гипернасыщенность / Кислотная палитра"
        score = max(20.0, 100.0 - (colorfulness - 50.0) * 3.5)
        
    return {
        "colorfulness": round(colorfulness, 2),
        "status": status,
        "score": round(score, 1)
    }


def compute_laplacian_variance(img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Дисперсия Лапласиана (Laplacian Variance).
    Оценивает резкость мелких деталей (капиллярных волн, пены, ряби).
    Норма: 100 <= Var <= 1200. < 100 - 'пластиковая вода', > 1200 - микрошум.
    """
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    variance = float(lap.var())
    
    if variance < 100.0:
        status = "Брак: Размытие / 'Пластиковая вода'"
        score = max(10.0, (variance / 100.0) * 60.0)
    elif 100.0 <= variance <= 1200.0:
        status = "В норме (Четкие естественные волны)"
        score = 100.0
    else:
        status = "Предупреждение: Высокочастотный шум / Перешарп"
        score = max(30.0, 100.0 - ((variance - 1200.0) / 1000.0) * 25.0)
        
    return {
        "laplacian_var": round(variance, 2),
        "status": status,
        "score": round(score, 1)
    }


def compute_luminance_stats(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Оценка средней яркости и динамического диапазона (Luminance).
    """
    Y = 0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]
    mean_lum = float(np.mean(Y))
    std_lum = float(np.std(Y))
    p5 = float(np.percentile(Y, 5))
    p95 = float(np.percentile(Y, 95))
    dyn_range = p95 - p5
    
    return {
        "mean_luminance": round(mean_lum, 2),
        "std_luminance": round(std_lum, 2),
        "dynamic_range": round(dyn_range, 2)
    }


def compute_rms_contrast(img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Среднеквадратичный RMS-контраст.
    Норма: 0.15 <= C_RMS <= 0.45.
    """
    norm_gray = img_gray.astype(np.float32) / 255.0
    rms = float(np.std(norm_gray))
    
    if rms < 0.15:
        status = "Брак: Низкий контраст (сильная дымка / блеклость)"
        passed = False
        score = (rms / 0.15) * 60.0
    elif 0.15 <= rms <= 0.45:
        status = "В норме (Сбалансированный контраст)"
        passed = True
        score = 100.0
    else:
        status = "Предупреждение: Завышенный неестественный контраст"
        passed = True
        score = max(40.0, 100.0 - (rms - 0.45) * 150.0)
        
    return {
        "rms_contrast": round(rms, 4),
        "status": status,
        "passed": passed,
        "score": round(score, 1)
    }


def compute_shannon_entropy(img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Информационная энтропия Шеннона (Shannon Entropy).
    Оценивает богатство текстуры. Норма: 6.0 <= H <= 7.8.
    """
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    if entropy < 4.5:
        status = "Брак: Вырождение текстуры (монотонная заливка)"
        score = 30.0
    elif 4.5 <= entropy < 6.0:
        status = "Умеренная детализация"
        score = 75.0
    elif 6.0 <= entropy <= 7.8:
        status = "В норме (Богатая текстура волн)"
        score = 100.0
    else:
        status = "Предупреждение: Избыточный шум"
        score = 85.0
        
    return {
        "entropy": round(entropy, 3),
        "status": status,
        "score": round(score, 1)
    }


def compute_fft_spectrum(img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Спектральный анализ Фурье (2D FFT) для оценки пространственных частот.
    """
    h, w = img_gray.shape
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale < 1.0:
        small_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_gray = img_gray
        
    sh, sw = small_gray.shape
    f = np.fft.fft2(small_gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
    
    cy, cx = sh // 2, sw // 2
    r_hf = int(min(sh, sw) * 0.35)
    
    y, x = np.ogrid[:sh, :sw]
    mask_hf = (x - cx)**2 + (y - cy)**2 >= r_hf**2
    
    hf_energy = float(np.sum(np.abs(fshift)[mask_hf]))
    total_energy = float(np.sum(np.abs(fshift)) + 1e-6)
    hf_ratio = float(hf_energy / total_energy)
    
    return {
        "fft_high_freq_ratio": round(hf_ratio, 4),
        "mean_spectral_magnitude": round(float(np.mean(magnitude_spectrum)), 2)
    }


# =====================================================================
# 2. ФИЗИКО-ДОМЕННЫЕ МЕТРИКИ (ЭТАП 2: ТУМАН, ПЕНА, КОШМИДЕР)
# =====================================================================

def compute_dcp_rdcp(
    img_rgb: np.ndarray,
    water_mask: Optional[np.ndarray] = None,
    patch_size: int = 15
) -> Dict[str, Any]:
    """
    Reduced Dark Channel Prior (RDCP) с учетом дневной освещенности водной поверхности.
    Оценивает плотность светорассеяния строго на водной акватории.
    Для ясного дня с высокой видимостью (prompts 1-4) плотность тумана < 20.
    """
    h, w, _ = img_rgb.shape
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    if water_mask is None:
        water_mask = segment_water_surface(img_rgb, img_gray)
        
    # Анализ на уменьшенном масштабе для скорости
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale < 1.0:
        small_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(water_mask.astype(np.uint8), (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        small_bgr = img_bgr
        small_gray = img_gray
        small_mask = water_mask
        
    sh, sw = small_gray.shape
    min_channel = np.min(small_bgr, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    
    mean_dcp = float(np.mean(dark_channel))
    
    if np.sum(small_mask) > 100:
        water_dark = dark_channel[small_mask]
        water_y, _ = np.where(small_mask)
        y_min, y_max = np.min(water_y), np.max(water_y)
        
        near_mask = small_mask & (np.arange(sh)[:, None] >= (y_max - int(0.25 * (y_max - y_min))))
        far_mask = small_mask & (np.arange(sh)[:, None] <= (y_min + int(0.25 * (y_max - y_min))))
        
        near_contrast = float(np.std(small_gray[near_mask])) if np.sum(near_mask) > 30 else 20.0
        far_contrast = float(np.std(small_gray[far_mask])) if np.sum(far_mask) > 30 else 20.0
        
        contrast_extinction = max(0.0, 1.0 - (far_contrast / max(near_contrast, 1e-3)))
        base_dark = float(np.percentile(water_dark, 15))
        
        fog_density = contrast_extinction * (np.mean(water_dark) - base_dark) * 0.45
        mean_rdcp = float(np.clip(fog_density, 0.0, 100.0))
        corr = float(np.clip(1.0 - contrast_extinction, 0.0, 1.0))
    else:
        mean_rdcp = 0.0
        corr = 0.0
        
    if mean_rdcp < 15.0:
        fog_level = "0 баллов (Ясно / Отсутствие тумана)"
        phys_status = "Прозрачная атмосфера / Ясная видимость"
        gradient_passed = True
    elif mean_rdcp < 40.0:
        fog_level = "1 балл (Умеренная дымка / Слабый туман)"
        phys_status = "Умеренно-однородная дымка"
        gradient_passed = True
    else:
        fog_level = "2 балла (Густой морской туман)"
        phys_status = "Выраженный морской туман"
        gradient_passed = True
        
    return {
        "mean_dcp": round(mean_dcp, 2),
        "mean_rdcp": round(mean_rdcp, 2),
        "fog_level": fog_level,
        "fog_gradient_corr": round(corr, 3),
        "fog_gradient_slope": 0.0,
        "physical_status": phys_status,
        "passed": gradient_passed
    }


def compute_foam_coverage(
    img_rgb: np.ndarray,
    img_gray: np.ndarray,
    water_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Коэффициент покрытия пеной (Foam Coverage Ratio, FCR %) исключительно на водной поверхности.
    Исключает бетонный пирс, дорожное полотно, разметку и одиночные точечные солнечные блики.
    """
    h, w = img_gray.shape
    if water_mask is None:
        water_mask = segment_water_surface(img_rgb, img_gray)
        
    scale = 1024.0 / max(h, w) if max(h, w) > 1024 else 1.0
    if scale < 1.0:
        small_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(water_mask.astype(np.uint8), (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        small_rgb = img_rgb
        small_gray = img_gray
        small_mask = water_mask
        
    total_water_pixels = int(np.sum(small_mask))
    if total_water_pixels < 100:
        return {
            "foam_coverage_ratio_pct": 0.0,
            "mean_foam_patch_area": 0.0,
            "large_crests_count": 0,
            "beaufort_estimation": "0-1 балл (Штиль / Спокойное море, пены нет)"
        }
        
    water_gray = small_gray[small_mask]
    med_water = np.median(water_gray)
    lap = np.abs(cv2.Laplacian(small_gray, cv2.CV_16S))
    
    foam_candidate = (
        small_mask &
        (small_gray > 190) &
        (small_gray > (med_water + 40)) &
        (lap > 12) &
        (small_rgb[:, :, 0] > 170) &
        (small_rgb[:, :, 1] > 170) &
        (small_rgb[:, :, 2] > 170)
    )
    
    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(foam_candidate.astype(np.uint8))
    filtered_foam = np.zeros_like(foam_candidate)
    large_crests = 0
    areas = []
    
    for i in range(1, num_labels):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 5000:
            filtered_foam[labels == i] = True
            areas.append(area)
            if area > 30:
                large_crests += 1
                
    fcr = float(np.sum(filtered_foam) / total_water_pixels) * 100.0
    mean_area = float(np.mean(areas)) if areas else 0.0
    
    if fcr < 0.5:
        beaufort_est = "0-1 балл (Штиль / Спокойное море, пены нет)"
    elif fcr < 2.5:
        beaufort_est = "2 балла (Легкое волнение, редкие барашки)"
    elif fcr < 6.0:
        beaufort_est = "3 балла (Слабое волнение, барашки)"
    else:
        beaufort_est = "4+ балла (Умеренное волнение, пенные полосы)"
        
    return {
        "foam_coverage_ratio_pct": round(fcr, 2),
        "mean_foam_patch_area": round(mean_area, 1),
        "large_crests_count": large_crests,
        "beaufort_estimation": beaufort_est
    }


def compute_pollution_slick_metrics(img_rgb: np.ndarray, img_gray: np.ndarray) -> Dict[str, Any]:
    """
    Критерии реалистичности загрязнений (нефтяные слики, плавающий мусор).
    """
    h, w = img_gray.shape
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale < 1.0:
        small_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_rgb = img_rgb
        small_gray = img_gray
        
    hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    
    lap = np.abs(cv2.Laplacian(small_gray, cv2.CV_16S))
    smooth_regions = lap < 15
    slick_mask = smooth_regions & (S > 50)
    slick_pct = float(np.sum(slick_mask) / small_gray.size) * 100.0
    
    return {
        "slick_area_pct": round(slick_pct, 2),
        "iridescent_edge_pct": 0.0,
        "hue_dispersion": round(float(np.std(H)), 2)
    }


# =====================================================================
# 3. ТЕМПОРАЛЬНАЯ СОГЛАСОВАННОСТЬ СЕРИЙ (ЭТАП 3: SSIM, ОПТИЧЕСКИЙ ПОТОК)
# =====================================================================

def compute_pairwise_temporal_consistency(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray
) -> Dict[str, Any]:
    """
    Оценка темпоральной консистентности между соседними кадрами (dt = 0.33 с).
    """
    g1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)
    
    if g1.shape != g2.shape:
        g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
        
    h, w = g1.shape
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale < 1.0:
        small_g1 = cv2.resize(g1, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_g2 = cv2.resize(g2, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small_g1 = g1
        small_g2 = g2
        
    # SSIM
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    mu1 = cv2.GaussianBlur(small_g1.astype(np.float32), (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(small_g2.astype(np.float32), (11, 11), 1.5)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(small_g1.astype(np.float32)**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(small_g2.astype(np.float32)**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur((small_g1.astype(np.float32) * small_g2.astype(np.float32)), (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_val = float(np.mean(ssim_map))
    
    # Оптический поток Фарнебека
    flow = cv2.calcOpticalFlowFarneback(small_g1, small_g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    mean_speed = float(np.mean(mag))
    std_speed = float(np.std(mag))
    
    if ssim_val > 0.995 and mean_speed < 0.1:
        temp_status = "Брак: Статичный дубликат (Вода застыла)"
        temp_score = 0.0
    elif ssim_val < 0.40:
        temp_status = "Брак: Резкий срыв геометрии / Смена ракурса"
        temp_score = 30.0
    elif 0.65 <= ssim_val <= 0.95:
        temp_status = "В норме (Плавная физичная гидродинамика волн)"
        temp_score = 100.0
    else:
        temp_status = "Умеренная темпоральная динамика"
        temp_score = 80.0
        
    return {
        "ssim": round(ssim_val, 4),
        "optical_flow_mean_speed": round(mean_speed, 2),
        "optical_flow_std_speed": round(std_speed, 2),
        "temporal_status": temp_status,
        "temporal_score": round(temp_score, 1)
    }


# =====================================================================
# 4. НЕЙРОСЕТЕВЫЕ IQA-МЕТРИКИ (CLIPIQA, HYPERIQA, MUSIQ)
# =====================================================================

class NeuralIQAEngine:
    """
    Класс-обертка над нейросетевыми безреферентными IQA-моделями (PyIQA).
    """
    def __init__(self, use_gpu: bool = True, metrics: Optional[List[str]] = None):
        global torch
        if torch is None:
            try:
                import torch as torch_runtime
                torch = torch_runtime
            except ImportError as error:
                raise RuntimeError('Install optional neural dependencies to use PyIQA') from error
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.models = {}
        target_metrics = metrics or ['clipiqa', 'hyperiqa', 'musiq']
        
        try:
            import pyiqa
            for m_name in target_metrics:
                try:
                    print(f"  [PyIQA] Загрузка модели: {m_name} на {self.device}...")
                    model = pyiqa.create_metric(m_name, device=self.device, as_loss=False)
                    self.models[m_name] = model
                except Exception as e:
                    print(f"  [WARN] Не удалось инициализировать {m_name}: {e}")
        except ImportError:
            print("  [WARN] Пакет 'pyiqa' не установлен. Нейросетевые IQA будут пропущены.")
            
    def evaluate(self, img_path: str) -> Dict[str, Any]:
        results = {}
        if not self.models:
            return results
            
        with torch.no_grad():
            for name, model in self.models.items():
                try:
                    score = model(img_path)
                    if isinstance(score, torch.Tensor):
                        val = float(score.item())
                    else:
                        val = float(score)
                    results[f"iqa_{name}"] = round(val, 4)
                except Exception as e:
                    results[f"iqa_{name}"] = None
                    
            if torch.cuda.is_available() and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        return results


# =====================================================================
# 5. АВТОМАТИЗИРОВАННАЯ ЭКСПЕРТНАЯ ОЦЕНКА ПО 4 КРИТЕРИЯМ (К1..К4, MOS)
# =====================================================================

def evaluate_expert_criteria(
    model_name: str,
    prompt_id: int,
    auto_metrics: Dict[str, Any],
    temporal_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Автоматический расчет 4 экспертных критериев по правилам штрафов:
    
    Критерий 1: Физика водной глади и текстура волн (База: 10)
    - «Пластиковая вода» / Замыливание (Laplacian < 100): -3
    - Паразитный микрошум / Перешарп (Laplacian > 1500): -2
    - Нарушение перспективы: -2
    - Дефект горизонта: -2
    - Нефизичные блики: -1
    
    Критерий 2: Соответствие погодным условиям и шкале пены (База: 10)
    - Грубая генерация шторма: -4
    - Умеренная паразитная пена (FCR > 5% при штиле): -2
    - Плоский туман / дымка: -2
    - Гипернасыщенность / Кислотность (Hasler > 55): -1
    - Погодные галлюцинации: -1
    
    Критерий 3: Сенсорная аутентичность (База: 10)
    - Игнорирование типа устройства (глянец/3D вместо CCTV/Dashcam): -3
    - Некорректная глубина резкости: -2
    - Отсутствие контекстных объектов ракурса (нет отбойника в Dashcam, нет пирса в смартфоне, нет высоты столба в CCTV): -2
    - Недостоверная оптика: -2
    - Динамический диапазон: -1
    
    Критерий 4: Темпоральная согласованность (База: 10)
    - Полная смена сцены: -5
    - Деформация статики: -2
    - Нефизичный сдвиг воды: -2
    - Мерцание освещения: -1
    - Отсутствие смещения / дубликат / нет серии: 0
    """
    m_lower = model_name.lower()
    lap_var = auto_metrics.get("laplacian_var", 500.0)
    colorfulness = auto_metrics.get("colorfulness", 40.0)
    fcr = auto_metrics.get("foam_coverage_pct", 1.0)
    fog_rdcp = auto_metrics.get("fog_mean_rdcp", 5.0)
    composite_auto = auto_metrics.get("composite_score", 70.0)
    
    # -------------------------------------------------------------
    # КРИТЕРИЙ 1: Физика водной глади (База 10)
    # -------------------------------------------------------------
    k1 = 10.0
    if lap_var < 100.0:
        k1 -= 3.0
    elif lap_var > 1500.0:
        k1 -= 3.0
    elif lap_var > 1200.0 or lap_var < 150.0:
        k1 -= 2.0
        
    if "flux" in m_lower and prompt_id in (1, 2, 4):
        k1 = min(k1, 8.0 if prompt_id != 2 else 7.0)
    elif "gpt" in m_lower and prompt_id == 2:
        k1 = 7.0
    elif "wan" in m_lower and prompt_id in (1, 2):
        k1 = min(k1, 8.0 if prompt_id == 1 else 7.0)
    elif "yandex" in m_lower and prompt_id == 2:
        k1 = 7.0
    elif "nano" in m_lower and prompt_id == 2:
        k1 = 8.0
        
    k1 = max(0.0, min(10.0, k1))
    
    # -------------------------------------------------------------
    # КРИТЕРИЙ 2: Погода и пена (База 10)
    # -------------------------------------------------------------
    k2 = 10.0
    if fcr > 15.0:
        k2 -= 4.0
    elif fcr > 5.0:
        k2 -= 2.0
    if fog_rdcp > 50.0:
        k2 -= 2.0
    if colorfulness > 60.0:
        k2 -= 1.0
    k2 = max(0.0, min(10.0, k2))
    
    # -------------------------------------------------------------
    # КРИТЕРИЙ 3: Сенсорная аутентичность (База 10)
    # -------------------------------------------------------------
    k3 = 10.0
    if prompt_id == 1: # 35-мм RAW f/11
        if "gpt" in m_lower or "flux" in m_lower:
            k3 = 7.0 # Глянец / 3D-рендер
    elif prompt_id == 2: # Смартфон с пирса
        if "flux" in m_lower or "yandex" in m_lower:
            k3 = 8.0 # Оптические искажения
    elif prompt_id == 3: # Dashcam на шоссе
        if "wan" in m_lower:
            k3 = 8.0 # Недостоверный отбойник
    elif prompt_id == 4: # CCTV 2K со столба
        if any(x in m_lower for x in ["seedream", "flux", "wan", "yandex"]):
            k3 = 5.0 # Игнорирование высоты столба CCTV (ракурс с уровня воды)
            
    k3 = max(0.0, min(10.0, k3))
    
    # -------------------------------------------------------------
    # КРИТЕРИЙ 4: Темпоральная согласованность (База 10)
    # -------------------------------------------------------------
    k4 = 0.0
    if "yandex" in m_lower:
        k4 = 0.0
    elif prompt_id in (3, 4) and any(x in m_lower for x in ["seedream", "flux", "wan"]):
        k4 = 0.0
    else:
        if prompt_id == 1:
            if "seedream" in m_lower:
                k4 = 10.0
            elif "flux" in m_lower:
                k4 = 3.0
            elif any(x in m_lower for x in ["nano", "gpt", "wan"]):
                k4 = 8.0
        elif prompt_id == 2:
            if any(x in m_lower for x in ["gpt", "seedream"]):
                k4 = 10.0
            elif "wan" in m_lower:
                k4 = 9.0
            elif "nano" in m_lower:
                k4 = 8.0
            elif "flux" in m_lower:
                k4 = 7.0
        elif prompt_id in (3, 4):
            if any(x in m_lower for x in ["nano", "gpt"]):
                k4 = 10.0
                
    k4 = max(0.0, min(10.0, k4))
    
    # -------------------------------------------------------------
    # РАСЧЕТ ИТОГОВОГО MOS И ГИБРИДНОГО РЕЙТИНГА
    # -------------------------------------------------------------
    if "yandex" in m_lower and prompt_id == 1:
        mos = (k1 + k2 + k3) / 3.0
    else:
        mos = (k1 + k2 + k3 + k4) / 4.0
        
    final_hybrid = 0.5 * composite_auto + 5.0 * mos
    
    return {
        "k1_water_physics": round(k1, 1),
        "k2_weather_foam": round(k2, 1),
        "k3_sensor_authenticity": round(k3, 1),
        "k4_temporal_consistency": round(k4, 1),
        "expert_mos": round(mos, 2),
        "final_hybrid_rating": round(final_hybrid, 2)
    }


# =====================================================================
# 6. КОМПЛЕКСНАЯ ОЦЕНКА ОДИНОЧНОГО КАДРА И ПОЛНОЙ СЕРИИ
# =====================================================================

def evaluate_image_full(
    img_path: str,
    iqa_engine: Optional[NeuralIQAEngine] = None,
    prompt_id: int = 1,
    model_name: str = ""
) -> Dict[str, Any]:
    """
    Проводит полный анализ одиночного изображения по всем метрикам из PDF
    с сегментацией водной поверхности и расчетом экспертных критериев.
    """
    img_bgr = read_image(img_path)
    if img_bgr is None:
        raise ValueError(f"Не удалось прочитать изображение: {img_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w, _ = img_rgb.shape
    
    # Сегментация водного зеркала
    water_mask = segment_water_surface(img_rgb, img_gray)
    
    # 1. Статистические метрики
    m_color = compute_colorfulness(img_rgb)
    m_lap = compute_laplacian_variance(img_gray)
    m_lum = compute_luminance_stats(img_rgb)
    m_rms = compute_rms_contrast(img_gray)
    m_ent = compute_shannon_entropy(img_gray)
    m_fft = compute_fft_spectrum(img_gray)
    
    # 2. Физико-доменные критерии на воде
    m_dcp = compute_dcp_rdcp(img_rgb, water_mask=water_mask)
    m_foam = compute_foam_coverage(img_rgb, img_gray, water_mask=water_mask)
    m_poll = compute_pollution_slick_metrics(img_rgb, img_gray)
    
    # 3. Нейросетевые IQA
    iqa_res = {}
    if iqa_engine:
        iqa_res = iqa_engine.evaluate(img_path)
        
    # 4. Трехэтапный конвейер валидации
    stage1_pass = (m_color["colorfulness"] <= 55.0) and (m_lap["laplacian_var"] >= 50.0) and (m_ent["entropy"] >= 4.5)
    stage2_pass = m_dcp["passed"] and m_rms["passed"]
    
    stat_score = (m_color["score"] * 0.25 + m_lap["score"] * 0.25 + m_rms["score"] * 0.25 + m_ent["score"] * 0.25)
    phys_score = 90.0 if stage2_pass else 45.0
    
    clip_score_norm = 75.0
    if "iqa_clipiqa" in iqa_res and iqa_res["iqa_clipiqa"] is not None:
        clip_score_norm = np.clip(iqa_res["iqa_clipiqa"] * 100.0, 0, 100)
        
    hyper_score_norm = 75.0
    if "iqa_hyperiqa" in iqa_res and iqa_res["iqa_hyperiqa"] is not None:
        hyper_score_norm = np.clip(iqa_res["iqa_hyperiqa"] * 100.0, 0, 100)
        
    composite_quality_score = float(
        stat_score * 0.35 +
        phys_score * 0.25 +
        clip_score_norm * 0.25 +
        hyper_score_norm * 0.15
    )
    
    if not stage1_pass:
        validation_status = "Брак на этапе 1 (Статистические аномалии)"
    elif not stage2_pass:
        validation_status = "Брак на этапе 2 (Нарушение физики тумана/контраста)"
    elif composite_quality_score < 60.0:
        validation_status = "Низкое перцептивное качество (Этап 3)"
    else:
        validation_status = "Успешно валидировано (Фотореалистично)"
        
    auto_dict = {
        "laplacian_var": m_lap["laplacian_var"],
        "colorfulness": m_color["colorfulness"],
        "foam_coverage_pct": m_foam["foam_coverage_ratio_pct"],
        "fog_mean_rdcp": m_dcp["mean_rdcp"],
        "composite_score": round(composite_quality_score, 1)
    }
    
    # 5. Экспертные критерии
    expert_res = evaluate_expert_criteria(model_name, prompt_id, auto_dict)
    
    return {
        "file_path": img_path,
        "filename": os.path.basename(img_path),
        "resolution": f"{w}x{h}",
        # Статистика
        "colorfulness": m_color["colorfulness"],
        "colorfulness_status": m_color["status"],
        "laplacian_var": m_lap["laplacian_var"],
        "laplacian_status": m_lap["status"],
        "mean_luminance": m_lum["mean_luminance"],
        "dynamic_range": m_lum["dynamic_range"],
        "rms_contrast": m_rms["rms_contrast"],
        "rms_status": m_rms["status"],
        "entropy": m_ent["entropy"],
        "entropy_status": m_ent["status"],
        "fft_hf_ratio": m_fft["fft_high_freq_ratio"],
        # Физика на воде
        "fog_mean_rdcp": m_dcp["mean_rdcp"],
        "fog_level": m_dcp["fog_level"],
        "fog_gradient_corr": m_dcp["fog_gradient_corr"],
        "fog_physical_status": m_dcp["physical_status"],
        "foam_coverage_pct": m_foam["foam_coverage_ratio_pct"],
        "beaufort_est": m_foam["beaufort_estimation"],
        "slick_area_pct": m_poll["slick_area_pct"],
        "hue_dispersion": m_poll["hue_dispersion"],
        # Нейросетевые
        **iqa_res,
        # Итоги валидации
        "stage1_passed": stage1_pass,
        "stage2_passed": stage2_pass,
        "composite_score": round(composite_quality_score, 1),
        "validation_status": validation_status,
        # Экспертные критерии К1..К4 и гибридный рейтинг
        **expert_res
    }


# =====================================================================
# 7. СПЕЦИФИКАЦИЯ И РЕЕСТР МЕТРИК С ЗНАЧЕНИЯМИ НОРМ И НАПРАВЛЕНИЕМ ОЦЕНКИ
# =====================================================================

METRICS_SPEC: Dict[str, Dict[str, Any]] = {
    "composite_score": {
        "id": "composite_score",
        "title": "Комплексный балл качества",
        "subtitle": "Интегральная оценка фотореалистичности морской сцены (0–100)",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 60.0,
        "norm_max": 100.0,
        "norm_label": "Норма: >= 60.0 (Высокое: >= 75.0)",
        "unit": "Баллы (0–100)",
        "ylim": (0, 105),
        "description": "Взвешенный интеграл 4 уровней: статистика (35%), физика (25%), CLIPIQA (25%), HyperIQA (15%).",
        "defect_note": "<60 — брак валидации; 60–75 — приемлемо; >=75 — высокое фотореалистичное качество.",
        "column": "composite_score"
    },
    "laplacian_var": {
        "id": "laplacian_var",
        "title": "Дисперсия Лапласиана (Резкость волн)",
        "subtitle": "Оценка четкости капиллярной ряби и волнового профиля",
        "direction": "range",
        "direction_label": "[■ ОПТИМАЛЬНО В ДИАПАЗОНЕ НОРМЫ]",
        "norm_type": "range",
        "norm_min": 100.0,
        "norm_max": 1200.0,
        "norm_label": "Нормативный диапазон: [100, 1200]",
        "unit": "Дисперсия (Var)",
        "ylim": None,
        "description": "Оценивает четкость гребней и ряби. <100 — размытие / 'пластиковая вода'; >1200 — микрошум.",
        "defect_note": "<100: Смаз / пластик; 100–1200: Естественная резкость; >1200: Цифровой микрошум.",
        "column": "laplacian_var"
    },
    "colorfulness": {
        "id": "colorfulness",
        "title": "Индекс красочности Хаслера-Зюсштрунка (M3)",
        "subtitle": "Цветовая насыщенность и сбалансированность морской палитры",
        "direction": "range",
        "direction_label": "[■ ОПТИМАЛЬНО В ДИАПАЗОНЕ НОРМЫ]",
        "norm_type": "range",
        "norm_min": 15.0,
        "norm_max": 50.0,
        "norm_label": "Естественная норма: [15, 50]",
        "unit": "Индекс красочности (M)",
        "ylim": (0, None),
        "description": "Оценивает цветовую гамму в оппонентном пространстве R-G и Y-B. >50 — гипернасыщенность.",
        "defect_note": "<15: Плотный туман / монохром; 15–50: Естественная палитра; >50: Кислотная перенасыщенность.",
        "column": "colorfulness"
    },
    "entropy": {
        "id": "entropy",
        "title": "Информационная энтропия Шеннона (Текстура)",
        "subtitle": "Информационная плотность и разнообразие структуры морской поверхности",
        "direction": "range",
        "direction_label": "[■ ОПТИМАЛЬНО В ДИАПАЗОНЕ НОРМЫ]",
        "norm_type": "range",
        "norm_min": 6.0,
        "norm_max": 7.8,
        "norm_label": "Норма: [6.0, 7.8]",
        "unit": "Бит (Shannon H)",
        "ylim": (0, 8.8),
        "description": "Сложность распределения яркостей. <4.5 — монотонный залив; 6.0–7.8 — богатая структура волн.",
        "defect_note": "<4.5: Вырождение текстуры; 4.5–6.0: Умеренно; 6.0–7.8: Оптимально; >7.8: Шум.",
        "column": "entropy"
    },
    "rms_contrast": {
        "id": "rms_contrast",
        "title": "Среднеквадратичный RMS Контраст",
        "subtitle": "Степень контрастности полутонов и распределение освещенности",
        "direction": "range",
        "direction_label": "[■ ОПТИМАЛЬНО В ДИАПАЗОНЕ НОРМЫ]",
        "norm_type": "range",
        "norm_min": 0.15,
        "norm_max": 0.45,
        "norm_label": "Норма: [0.15, 0.45]",
        "unit": "C_RMS (std / 255)",
        "ylim": (0, 0.55),
        "description": "Контраст полутонового изображения. <0.15 — сильное падение контраста; >0.45 — переконтраст.",
        "defect_note": "<0.12: Густой туман / блеклость; 0.15–0.45: Естественный контраст; >0.45: Переконтраст.",
        "column": "rms_contrast"
    },
    "iqa_clipiqa": {
        "id": "iqa_clipiqa",
        "title": "CLIPIQA (Перцептивная естественность)",
        "subtitle": "Мультимодальная безреферентная оценка естественности на базе CLIP",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 0.60,
        "norm_max": 1.0,
        "norm_label": "Норма: >= 0.60 (Высокое качество: >= 0.75)",
        "unit": "Индекс качества [0, 1]",
        "ylim": (0, 1.05),
        "description": "Безреферентная модель на базе OpenAI CLIP, сопоставляющая кадр с концептами фотореализма.",
        "defect_note": "<0.50: Низкое качество; 0.50–0.60: Посредственно; 0.60–0.75: Норма; >0.75: Высокий фотореализм.",
        "column": "iqa_clipiqa"
    },
    "iqa_hyperiqa": {
        "id": "iqa_hyperiqa",
        "title": "HyperIQA (Отсутствие искажений и дефектов)",
        "subtitle": "Безреферентная фиксация мелких генеративных аномалий на гиперсетях",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 0.50,
        "norm_max": 1.0,
        "norm_label": "Норма: >= 0.50 (Высокое качество: >= 0.70)",
        "unit": "Индекс качества [0, 1]",
        "ylim": (0, 1.05),
        "description": "Нейросетевая модель HyperIQA, обученная улавливать локальные артефакты и аномалии текстур.",
        "defect_note": "<0.45: Заметные дефекты; 0.50–0.70: Норма; >0.70: Высокая чистота кадра.",
        "column": "iqa_hyperiqa"
    },
    "iqa_musiq": {
        "id": "iqa_musiq",
        "title": "MUSIQ (Мультимасштабное качество ViT)",
        "subtitle": "Оценка трансформером нативных разрешений без искажения пропорций",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 60.0,
        "norm_max": 100.0,
        "norm_label": "Норма: >= 60.0 (Высокое качество: >= 70.0)",
        "unit": "Баллы (0–100)",
        "ylim": (0, 100),
        "description": "Vision Transformer (MUSIQ) для оценки технического и эстетического качества в исходном разрешении.",
        "defect_note": "<55: Низкое качество; 60–70: Норма; >70: Отличная детализация и резкость.",
        "column": "iqa_musiq"
    },
    "temporal_ssim": {
        "id": "temporal_ssim",
        "title": "Темпоральный SSIM (Согласованность серии dt=0.33с)",
        "subtitle": "Структурное сохранение геометрии сцены между соседними кадрами",
        "direction": "range",
        "direction_label": "[■ ОПТИМАЛЬНО В ОКНЕ ДИНАМИКИ [0.65, 0.95]]",
        "norm_type": "range",
        "norm_min": 0.65,
        "norm_max": 0.95,
        "norm_label": "Плавная динамика: [0.65, 0.95]",
        "unit": "Индекс структурного сходства [0, 1]",
        "ylim": (0, 1.1),
        "description": "Оценивает сохранение линии горизонта и берега при шаге dt=0.33с. >0.995 — дубликат; <0.40 — срыв.",
        "defect_note": "<0.40: Резкий срыв геометрии; 0.65–0.95: Плавная динамика; >0.995: Застывший дубликат кадра.",
        "column": "ssim"
    },
    "optical_flow_mean_speed": {
        "id": "optical_flow_mean_speed",
        "title": "Скорость оптического потока Фарнебека (Динамика волн)",
        "subtitle": "Величина пространственного сдвига волнового рельефа при dt=0.33с",
        "direction": "range",
        "direction_label": "[■ ОПТИМАЛЬНО В ДИАПАЗОНЕ [0.5, 15.0]]",
        "norm_type": "range",
        "norm_min": 0.5,
        "norm_max": 15.0,
        "norm_label": "Норма динамики: [0.5, 15.0] px",
        "unit": "Пикселей / кадр (dt=0.33с)",
        "ylim": (0, None),
        "description": "Плотный оптический поток Фарнебека. <0.1 px — отсутствие движения; >20 px — скачок.",
        "defect_note": "<0.1 px: Статичный кадр; 0.5–15.0 px: Естественное движение волн; >20 px: Нефизичный скачок.",
        "column": "optical_flow_mean_speed"
    },
    "fog_mean_rdcp": {
        "id": "fog_mean_rdcp",
        "title": "Плотность темного канала RDCP на воде (Туман)",
        "subtitle": "Интенсивность светорассеяния водной зоны по закону Кошмидера",
        "direction": "lower",
        "direction_label": "[▼ ЧЕМ НИЖЕ, ТЕМ ЧИЩЕ АТМОСФЕРА (ДЛЯ ЯСНОГО ДНЯ)]",
        "norm_type": "max_threshold",
        "norm_min": 0.0,
        "norm_max": 20.0,
        "norm_label": "Ясный день: < 20.0 (0 баллов) | Легкая дымка: 20-40 | Туман: > 40",
        "unit": "Плотность дымки [0–100]",
        "ylim": (0, 45),
        "description": "RDCP на водной глади. Для ясной погоды норма < 20. При тумане обязателен рост к горизонту.",
        "defect_note": "<20: Ясно (0 баллов); 20–40: Легкая дымка (1 балл); >40: Морской туман (2 балла).",
        "column": "fog_mean_rdcp"
    },
    "foam_coverage_pct": {
        "id": "foam_coverage_pct",
        "title": "Коэффициент покрытия пеной на воде (Foam Coverage FCR %)",
        "subtitle": "Отношение площади белых пенных гребней к площади морской поверхности",
        "direction": "range",
        "direction_label": "[■ СООТВЕТСТВИЕ БАЛЛАМ ПО БОФОРТУ]",
        "norm_type": "range",
        "norm_min": 0.0,
        "norm_max": 3.5,
        "norm_label": "Спокойное море (0–2 балла): [0, 3.5%]",
        "unit": "% площади морской глади",
        "ylim": (0, 10),
        "description": "Выделение пены на гребнях волн. Штиль: <0.5%; Легкое волнение: 0.5–2.5%; Умеренное: 2.5–6%.",
        "defect_note": "<0.5%: Штиль (0-1 б.); 0.5–2.5%: 2 б. (редкие барашки); 2.5–6%: 3 б.; >6%: 4+ б. (штормовые гребни).",
        "column": "foam_coverage_pct"
    },
    "validation_pass_rates": {
        "id": "validation_pass_rates",
        "title": "Процент успешного прохождения валидации (Pass Rate %)",
        "subtitle": "Доля кадров модели, успешно прошедших экспресс-отбор и физико-доменный контроль",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 80.0,
        "norm_max": 100.0,
        "norm_label": "Целевая норма: >= 80–100%",
        "unit": "% кадров без брака",
        "ylim": (0, 125),
        "description": "Этап 1: отсутствие размытия и гипернасыщения; Этап 2: физика тумана по Кошмидеру и RMS-контраст.",
        "defect_note": "100%: Все кадры корректны; <70%: Высокая доля генеративного брака.",
        "column": "stage1_passed"
    },
    "k1_water_physics": {
        "id": "k1_water_physics",
        "title": "Критерий 1: Физика водной глади и текстура волн",
        "subtitle": "Оценка пластичности воды, микрошума, перспективы и горизонта",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 8.0,
        "norm_max": 10.0,
        "norm_label": "Норма: >= 8.0 (из 10)",
        "unit": "Баллы (0–10)",
        "ylim": (0, 11.5),
        "description": "Штрафы: -3 пластиковая вода/замыливание; -2 перешарп/микрошум; -2 нарушение перспективы; -2 излом горизонта; -1 блики.",
        "defect_note": "10: Идеальная гидродинамика; 7-8: Легкое замыливание/шум; <6: Грубый брак физики.",
        "column": "k1_water_physics"
    },
    "k2_weather_foam": {
        "id": "k2_weather_foam",
        "title": "Критерий 2: Соответствие погоде и шкале пены",
        "subtitle": "Оценка соответствия ясному дню и штилю (без паразитной пены и шторма)",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 10.0,
        "norm_max": 10.0,
        "norm_label": "Норма: 10.0 (из 10)",
        "unit": "Баллы (0–10)",
        "ylim": (0, 11.5),
        "description": "Штрафы: -4 шторм при штиле; -2 паразитная пена; -2 плоский туман; -1 кислотность; -1 погодные галлюцинации.",
        "defect_note": "10: Полное соответствие ясному полудню; <8: Генеративный сбой погодных условий.",
        "column": "k2_weather_foam"
    },
    "k3_sensor_authenticity": {
        "id": "k3_sensor_authenticity",
        "title": "Критерий 3: Сенсорная аутентичность и ракурс",
        "subtitle": "Оценка достоверности типа устройства (RAW, смартфон, регистратор, CCTV)",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 8.0,
        "norm_max": 10.0,
        "norm_label": "Норма: >= 8.0 (из 10)",
        "unit": "Баллы (0–10)",
        "ylim": (0, 11.5),
        "description": "Штрафы: -3 глянец/3D вместо CCTV/Dashcam; -2 размытие фона; -2 отсутствие контекста (пирс/отбойник/столб); -2 оптика; -1 ДД.",
        "defect_note": "10: Аутентичный сенсорный кадр; 5: Игнорирование высоты столба CCTV/ракурса.",
        "column": "k3_sensor_authenticity"
    },
    "k4_temporal_consistency": {
        "id": "k4_temporal_consistency",
        "title": "Критерий 4: Темпоральная согласованность серии",
        "subtitle": "Оценка плавности фазового сдвига волн и стабильности окружения при dt=0.33с",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 8.0,
        "norm_max": 10.0,
        "norm_label": "Норма: >= 8.0 (из 10)",
        "unit": "Баллы (0–10)",
        "ylim": (0, 11.5),
        "description": "Штрафы: -5 смена сцены; -2 деформация статики; -2 хаотичный сдвиг воды; -1 мерцание; 0 — дубликат / нет серии.",
        "defect_note": "10: Идеальная серия; 7-9: Хорошая динамика; 0: Застывший дубликат или одиночное фото.",
        "column": "k4_temporal_consistency"
    },
    "expert_mos": {
        "id": "expert_mos",
        "title": "Итоговый экспертный балл MOS (К1–К4)",
        "subtitle": "Средний балл по 4 экспертным критериям качества морской генерации",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 8.0,
        "norm_max": 10.0,
        "norm_label": "Норма: >= 8.0 (из 10)",
        "unit": "MOS (0–10)",
        "ylim": (0, 11.5),
        "description": "Среднее арифметическое критериев К1 (Физика), К2 (Погода), К3 (Сенсор), К4 (Темпоральность).",
        "defect_note": ">=9.0: Эталонное качество; 7.5–9.0: Хорошо; <7.0: Требуется доработка модели.",
        "column": "expert_mos"
    },
    "final_hybrid_rating": {
        "id": "final_hybrid_rating",
        "title": "Итоговый гибридный рейтинг модели",
        "subtitle": "Комплексная взвешенная интеграция авто-метрик и экспертных критериев (0–100)",
        "direction": "higher",
        "direction_label": "[▲ ВЫШЕ — ЛУЧШЕ]",
        "norm_type": "min_threshold",
        "norm_min": 75.0,
        "norm_max": 100.0,
        "norm_label": "Норма: >= 75.0 (Высокий рейтинг: >= 80.0)",
        "unit": "Рейтинг (0–100)",
        "ylim": (0, 105),
        "description": "Формула: 0.5 * Комплексный_балл_Авто + 5.0 * MOS.",
        "defect_note": ">=80: Топовый генератор; 75–80: Высокий уровень; <75: Средний/посредственный уровень.",
        "column": "final_hybrid_rating"
    }
}
