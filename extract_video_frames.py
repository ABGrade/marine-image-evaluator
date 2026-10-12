"""
Скрипт для автоматического извлечения второго (t=0.33с) и третьего (t=0.66с) кадров
из видеофайлов модели Nano Banana для последующей оценки темпоральных и пространственных метрик.
"""

import os
import re
import cv2
import glob
from pathlib import Path


def extract_frames_from_video(video_path: str, output_dir: str = None, dt: float = 0.333333):
    """
    Извлекает кадры из видеофайла на временных метках t = dt (кадр 2) и t = 2*dt (кадр 3).
    
    Имена сохраняемых файлов:
    Например: nanoBanana 1.mp4 -> nanoBanana 1-2.jpg, nanoBanana 1-3.jpg
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    stem = video_path.stem  # например 'nanoBanana 1'
    # Извлечем номер промпта из названия (например 'nanoBanana 1' -> prompt_id='1')
    match = re.search(r'(\d+)', stem)
    prompt_id = match.group(1) if match else "1"
    base_prefix = re.sub(r'\s*\d+$', '', stem).strip()
    if not base_prefix:
        base_prefix = "nanoBanana"
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Обработка видео: {video_path.name} | FPS: {fps:.2f} | Всего кадров: {total_frames} | Длительность: {duration:.2f}с")
    
    extracted_files = []
    timestamps = [
        (2, dt, f"{base_prefix} {prompt_id}-2.jpg"),
        (3, 2 * dt, f"{base_prefix} {prompt_id}-3.jpg")
    ]
    
    for frame_num, target_time, out_filename in timestamps:
        target_frame_idx = int(round(target_time * fps))
        if target_frame_idx >= total_frames:
            print(f"[WARN] Кадр {target_frame_idx} выходит за пределы видео ({total_frames} кадров).")
            continue
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"[ERROR] Ошибка чтения кадра {target_frame_idx} (t={target_time:.2f}с) из {video_path.name}")
            continue
            
        out_path = output_dir / out_filename
        # Сохраняем в высоком качестве JPEG (качество 98)
        success = cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if success:
            print(f"  [OK] Сохранен кадр {frame_num} (t={target_time:.2f}с, frame={target_frame_idx}): {out_path.name} ({frame.shape[1]}x{frame.shape[0]})")
            extracted_files.append(str(out_path))
        else:
            print(f"  [ERROR] Не удалось сохранить файл: {out_path}")
            
    cap.release()
    return extracted_files


def extract_all_nano_banana_videos(directory: str = "nano banana"):
    """
    Сканирует указанную директорию и извлекает кадры для всех .mp4 файлов.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[ERROR] Директория не найдена: {directory}")
        return []
        
    mp4_files = sorted(dir_path.glob("*.mp4"))
    if not mp4_files:
        print(f"[INFO] В директории {directory} нет файлов .mp4")
        return []
        
    print(f"=== Найдено видеофайлов: {len(mp4_files)} в папке '{directory}' ===")
    all_extracted = []
    for vf in mp4_files:
        extracted = extract_frames_from_video(vf, dir_path)
        all_extracted.extend(extracted)
        
    print(f"=== Успешно извлечено и сохранено {len(all_extracted)} кадров ===")
    return all_extracted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Извлечение кадров t=0.33с и t=0.66с из видео Nano Banana")
    parser.add_argument("--dir", default="nano banana", help="Директория с видеофайлами (по умолчанию: 'nano banana')")
    args = parser.parse_args()
    
    extract_all_nano_banana_videos(args.dir)
