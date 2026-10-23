import argparse
from tts.model_files import verify_model
def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args()
    manifest=verify_model(args.directory);print('Verified exact',manifest['model'],'artifacts; no fallback model is permitted.')
if __name__=='__main__':main()
