#!/usr/bin/env python3
"""Python standard library only. Never overwrite an existing destination.

Supply the ZIP SHA-256 from the delivery receipt through --sha256. Optional
checksummed binary parts are joined only into a NEW file. Inputs are preserved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

CHUNK=8*1024**2
CODEC_SIZE=1871099728
CODEC_SHA='74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f'

def require(test, message):
    if not test:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
    return h.hexdigest()

def safe_path(name):
    require(isinstance(name,str) and name and name.isascii(),'Invalid archive name')
    require(not any(c in name for c in '\\:\x00<>"|?*'),'Unsafe archive name')
    require(not name.startswith('/'),'Absolute archive name')
    for part in name.split('/'):
        require(part not in ('','.','..') and not part.endswith((' ','.')),'Unsafe path component')
        require(len(part)<=255 and not any(ord(c)<32 for c in part),'Invalid path component')
        require(not re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)',part,re.I),'Reserved Windows path')
    require(str(PurePosixPath(name))==name,'Noncanonical archive name')
    return name.split('/')

def no_link(path):
    path=Path(path).absolute()
    for p in [path,*path.parents]:
        if p.exists() or p.is_symlink():
            require(not p.is_symlink() and not (getattr(p.lstat(),'st_file_attributes',0)&0x400),
                    'Symlink/reparse point in path: '+str(p))

def verify_tree(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['format']=='aljawab-nf4-delivery-v1','Wrong delivery format')
    records=manifest['files']
    require(isinstance(records,dict) and len(records)<10000,'Invalid manifest')
    for name,record in records.items():
        safe_path(name)
        f=root/name
        no_link(f)
        require(f.is_file() and f.stat().st_size==record['bytes'] and sha(f)==record['sha256'],
                'Delivered file checksum mismatch: '+name)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==set(records)|{'manifest.json'},'Unexpected/missing delivered files')
    return manifest

def extract(archive,destination,digest,codec=None):
    archive=Path(archive);destination=Path(destination).absolute()
    require(re.fullmatch('[0-9a-f]{64}',digest) is not None,'Invalid expected SHA-256')
    no_link(archive);no_link(destination)
    require(not destination.exists(),'Destination already exists; choose a NEW folder')
    require(sha(archive)==digest,'ZIP SHA-256 mismatch; extraction refused')
    if codec:
        codec=Path(codec);no_link(codec)
        require(codec.stat().st_size==CODEC_SIZE and sha(codec)==CODEC_SHA,'Existing codec checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=set();folded=set();total=0
        require(len(infos)<10000,'Too many archive entries')
        for info in infos:
            safe_path(info.filename)
            require(not info.is_dir(),'Only regular file entries are supported')
            mode=info.external_attr>>16
            require(stat.S_IFMT(mode) in (0,stat.S_IFREG),'Archive links/devices refused')
            require(not info.flag_bits&1,'Encrypted entry refused')
            require(info.filename not in names and info.filename.casefold() not in folded,'Duplicate/colliding archive path')
            names.add(info.filename);folded.add(info.filename.casefold());total+=info.file_size
            require(total<=10*1024**3,'Archive expansion exceeds delivery bound')
        for name in names:
            require(not any('/'.join(name.split('/')[:i]).casefold() in folded for i in range(1,len(name.split('/')))),
                    'Archive file/directory conflict')
        require('manifest.json' in names,'Missing provenance manifest')
        destination.mkdir(parents=True,exist_ok=False)
        require(shutil.disk_usage(destination).free>=total+100*1024**2,'Insufficient free disk space')
        for info in infos:
            target=destination.joinpath(*safe_path(info.filename))
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('xb') as output:shutil.copyfileobj(source,output,CHUNK)
    result=verify_tree(destination)
    print('Verified extraction:',destination)
    print('Derivative:',result['derivative_identity'])
    if codec:print('Verified existing external codec:',codec)
    return result

def join_parts(manifest_path,archive):
    manifest_path=Path(manifest_path);archive=Path(archive)
    no_link(archive);require(not archive.exists(),'Joined ZIP already exists')
    data=json.loads(manifest_path.read_text(encoding='utf-8'))
    require(data['format']=='aljawab-zip-parts-v1','Invalid part manifest')
    parts=data['parts'];require(1<=len(parts)<=20,'Invalid part count')
    for index,part in enumerate(parts,1):
        require(part['index']==index,'Parts out of order')
        require(len(safe_path(part['name']))==1,'Unsafe part filename')
        f=manifest_path.parent/part['name'];no_link(f)
        require(0<part['bytes']<=1024**3 and f.stat().st_size==part['bytes'] and sha(f)==part['sha256'],
                'Part checksum mismatch: '+part['name'])
    with archive.open('xb') as target:
        for part in parts:
            with (manifest_path.parent/part['name']).open('rb') as source:shutil.copyfileobj(source,target,CHUNK)
    require(archive.stat().st_size==data['archive_bytes'] and sha(archive)==data['archive_sha256'],
            'Joined ZIP checksum mismatch')
    return data['archive_sha256']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--codec',type=Path)
    p.add_argument('--parts-manifest',type=Path)
    args=p.parse_args()
    if args.parts_manifest:
        digest=join_parts(args.parts_manifest,args.zip)
        require(digest==args.sha256,'Parts receipt differs from expected ZIP SHA-256')
    extract(args.zip,args.destination,args.sha256,args.codec)
