from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_directory
from marine_metrics import compute_pairwise_temporal_consistency

def test_unicode_paths_and_broken_file(tmp_path):
    source=tmp_path/'изображения';source.mkdir()
    y,x=np.mgrid[:48,:64]
    image=np.stack([(x*4)%255,(y*5)%255,((x+y)*3)%255],axis=-1).astype('uint8')
    Image.fromarray(image).save(source/'море.png')
    (source/'broken.jpg').write_text('invalid image')
    report=evaluate_directory(source,tmp_path/'results')
    assert report['images_evaluated']==1
    assert report['results'][0]['image']=='море.png'
    assert report['results'][0]['resolution']=='64x48'
    assert len(report['errors'])==1
    assert 'expert_mos' not in report['results'][0]

def test_duplicate_frames(tmp_path):
    image=np.random.default_rng(1).integers(0,255,(64,64,3),dtype=np.uint8)
    result=compute_pairwise_temporal_consistency(image,image.copy())
    assert result['ssim']==1
    assert result['optical_flow_mean_speed']<.1
