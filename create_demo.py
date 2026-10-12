"""Create clearly synthetic test patterns without distributing company data."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('demo-images'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    y,x=np.mgrid[:192,:320]
    waves=np.sin(x/7+y/5)*15+np.sin(x/23-y/9)*10
    image=np.clip(np.stack([55+waves,125+waves,170+waves],axis=-1),0,255).astype('uint8')
    Image.fromarray(image).save(args.output/'synthetic-waves.png')
    haze=np.clip(image*.5+125,0,255).astype('uint8')
    Image.fromarray(haze).save(args.output/'synthetic-haze.png')
    print(f'Synthetic demo patterns saved to {args.output}')

if __name__=='__main__':main()
