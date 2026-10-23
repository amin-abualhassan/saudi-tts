"""Run on the source AWS instance to copy the exact export, following codec symlinks."""
import argparse,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args()
    manifest=json.loads((ROOT/'tts/model-artifacts.json').read_text());source=Path(manifest['original_export_path']);destination=Path(args.destination).resolve()
    if not source.is_dir():raise SystemExit('This helper must run on the original AWS host. See tts/README.md for transfer commands.')
    if destination.exists():raise SystemExit('Choose a new empty destination; no files will be overwritten.')
    destination.mkdir(parents=True)
    for name in list(manifest['required_files'])+['export.json']:
        shutil.copy2(source/name,destination/name,follow_symlinks=True)
    print('Copied model artifacts to',destination,'; transfer this directory to your local GPU machine.')
if __name__=='__main__':main()
