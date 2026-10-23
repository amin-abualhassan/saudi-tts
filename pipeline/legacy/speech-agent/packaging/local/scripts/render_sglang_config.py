import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--model-directory',required=True);p.add_argument('--output',default='tts/sglang.local.json')
    p.add_argument('--attention-backend',default='fa3');p.add_argument('--memory-fraction',type=float,default=.32);a=p.parse_args()
    path=str(Path(a.model_directory).resolve());config=json.loads((ROOT/'tts/sglang.template.json').read_text());config['model_path']=path;config['name']=path
    for stage in config['stages']:
        if stage['name']=='tts_engine':
            overrides=stage['factory_args']['server_args_overrides'];overrides['attention_backend']=a.attention_backend;overrides['mem_fraction_static']=a.memory_fraction
    Path(a.output).write_text(json.dumps(config,indent=2));print('Wrote',a.output,'; SGLang installation/startup remains your responsibility.')
if __name__=='__main__':main()
