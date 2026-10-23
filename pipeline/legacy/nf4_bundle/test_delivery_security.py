"""Focused negative tests for restore and deliberately pinned derivative identity."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from verify_extract import safe_path, extract, join_parts

class RestoreTests(unittest.TestCase):
    def test_windows_paths_rejected(self):
        for path in ['../bad','/bad','C:/bad','a\\b','a/../b','NUL.txt','x:stream','x.','a//b']:
            with self.subTest(path=path),self.assertRaises(ValueError):safe_path(path)

    def test_checksum_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);z=root/'a.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('ok.txt','ok')
            with self.assertRaises(ValueError):extract(z,root/'new','0'*64)
            self.assertFalse((root/'new').exists())
            with self.assertRaises(ValueError):extract(z,root,hashlib.sha256(z.read_bytes()).hexdigest())

    def test_archive_traversal_links_and_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for i,names in enumerate([['../outside'],['a','A'],['a','a/b']]):
                z=root/f'{i}.zip'
                with zipfile.ZipFile(z,'w') as f:
                    for name in names:f.writestr(name,'bad')
                with self.assertRaises(ValueError):extract(z,root/f'dest{i}',hashlib.sha256(z.read_bytes()).hexdigest())
            z=root/'link.zip'
            info=zipfile.ZipInfo('link');info.external_attr=(0o120777<<16)
            with zipfile.ZipFile(z,'w') as f:f.writestr(info,'/etc/passwd')
            with self.assertRaises(ValueError):extract(z,root/'linkdest',hashlib.sha256(z.read_bytes()).hexdigest())

    def test_corrupt_part_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);part=root/'x.part0001';part.write_bytes(b'bad')
            manifest=root/'parts.json';manifest.write_text(json.dumps(dict(format='aljawab-zip-parts-v1',parts=[dict(index=1,name=part.name,bytes=3,sha256='0'*64)])))
            with self.assertRaises(ValueError):join_parts(manifest,root/'joined.zip')
            self.assertEqual(part.read_bytes(),b'bad');self.assertFalse((root/'joined.zip').exists())

if __name__=='__main__':unittest.main()
