"""Batch image descriptors with transparent CSV/JSON output."""
import argparse
import csv
import json
from pathlib import Path
from marine_metrics import evaluate_image_full, NeuralIQAEngine
from image_io import read_image

EXTENSIONS={'.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'}
MEASURED_FIELDS=[
    'resolution','colorfulness','laplacian_var','mean_luminance',
    'dynamic_range','rms_contrast','entropy','fft_hf_ratio',
    'fog_mean_rdcp','foam_coverage_pct','slick_area_pct','hue_dispersion',
]

def evaluate_directory(source, output, neural=False):
    source=Path(source).resolve()
    output=Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f'Input directory does not exist: {source}')
    paths=sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS
                 and not p.resolve().is_relative_to(output))
    if not paths:
        raise ValueError('No supported images in input directory')
    engine=NeuralIQAEngine(use_gpu=False) if neural else None
    if neural and not engine.models:
        raise ValueError('Neural IQA was requested, but no model could be loaded')
    rows=[]
    errors=[]
    for path in paths:
        relative=path.relative_to(source).as_posix()
        try:
            image=read_image(path)
            if image is None:
                raise ValueError('Image cannot be decoded')
            if min(image.shape[:2])<16:
                raise ValueError('Image dimensions must be at least 16x16')
            result=evaluate_image_full(str(path),iqa_engine=engine)
            row={'image':relative}
            row.update({field:result[field] for field in MEASURED_FIELDS})
            row.update({key:value for key,value in result.items() if key.startswith('iqa_')})
            rows.append(row)
        except Exception as error:
            errors.append({'image':relative,'error':str(error)})
    report={
        'images_discovered':len(paths),'images_evaluated':len(rows),
        'neural_iqa_requested':neural,'results':rows,'errors':errors,
        'interpretation':'Image descriptors and heuristic water-region statistics; detection accuracy and human MOS were not measured.',
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if rows:
        keys=list(dict.fromkeys(key for row in rows for key in row))
        with (output/'results.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys)
            writer.writeheader();writer.writerows(rows)
    return report

def main():
    parser=argparse.ArgumentParser(description='Evaluate image descriptors and export CSV/JSON')
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('output'))
    parser.add_argument('--neural',action='store_true',help='Explicitly enable optional PyIQA models; may download weights')
    args=parser.parse_args()
    try:
        report=evaluate_directory(args.input,args.output,args.neural)
    except (ValueError,RuntimeError) as error:
        parser.error(str(error))
    print(f"Evaluated {report['images_evaluated']}/{report['images_discovered']} images. Results: {args.output}")
    return 1 if report['errors'] else 0

if __name__=='__main__':
    raise SystemExit(main())
