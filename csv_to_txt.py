"""Load html from files, clean up, split, ingest into Weaviate."""
from dotenv import load_dotenv
from pathlib import Path
import sys
import argparse
import pandas as pd

if getattr(sys, 'frozen', False):
    script_location = Path(sys.executable).parent.resolve()
else:
    script_location = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=script_location / '.env')

parser = argparse.ArgumentParser(description='Ingest data.')
parser.add_argument('-f', '--fileName',
                    help="CSV file name")
parser.add_argument('-o', '--output_name',
                    help="txt file name")
args = parser.parse_args()
fileNmae = args.fileName
output_name = args.output_name


if __name__ == "__main__":
    # Read CSV file
    data = pd.read_csv(script_location / fileNmae)
    # Write to TXT file
    with open(script_location / output_name, 'w') as txtfile:
        for index, row in data.iterrows():
            txtfile.write(f"## {row['Prompt']}\n")
            txtfile.write(f"{row['Answer']}\n\n")
