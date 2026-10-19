"""Calculate dimensionless ratios relative to ASU from archived values."""
from pathlib import Path
import pandas as pd
import argparse

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame=pd.read_csv(args.input)
    reference=frame.loc[frame.structure.eq('ASU'),'source_index'].item()
    frame['dimensionless_ratio']=frame.source_index/reference
    frame.to_csv(args.output,index=False)
    print(frame.to_string(index=False))
if __name__=='__main__':main()
