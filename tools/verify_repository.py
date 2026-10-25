"""Validate curated files, imported hashes, syntax, JSON, links and privacy patterns."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    count = 0
    patterns = {
        'private IPv4': r'(?<![A-Za-z0-9_.=])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b',
        'private Windows path': r'(?i)[A-Z]:[/\\]Users[/\\][^\s/\\]+',
        'private Linux home': r'/' + r'home/[^\s/$]+/',
        'GPU UUID': r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{20,}',
        'GitHub credential': r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
        'private key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    }
    imported = json.loads((ROOT/'results/manifest.json').read_text())['files']
    for item in imported:
        path = ROOT/item['file']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            errors.append('Imported file hash mismatch: '+item['file'])
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in {'.git','__pycache__'} for p in path.relative_to(ROOT).parts):
            continue
        relative = path.relative_to(ROOT).as_posix()
        count += 1
        if path.name=='.env' or path.suffix in {'.pem','.key','.log'} or path.name.startswith('.env.'):
            errors.append('Disallowed file: '+relative)
            continue  # Do not read forbidden files even if accidentally placed here.
        if path.stat().st_size>=95_000_000:
            errors.append('File too large for normal Git upload: '+relative)
        if path.suffix in {'.png','.mp4','.pt'}:
            continue
        if path.suffix not in {'.py','.md','.json','.gz','.txt','.csv','.yaml','.yml'} and path.name not in {'.gitignore','.gitattributes'}:
            errors.append('Unrecognized curated file type: '+relative)
            continue
        raw = gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        text = raw.decode('utf-8-sig')
        for label,pattern in patterns.items():
            if re.search(pattern,text): errors.append(label+': '+relative)
        if path.suffix=='.py': ast.parse(text,filename=relative)
        if path.suffix=='.json' or path.name.endswith('.json.gz'): json.loads(text)
        if path.suffix=='.md':
            for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)',text):
                link = match.group(1).split('#')[0]
                if not link or '://' in link or link.startswith('mailto:'): continue
                if not (path.parent/link).exists(): errors.append('Broken link: '+relative+' -> '+link)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {count} curated files; {len(imported)} imported checksums; syntax, JSON, links and privacy patterns')


if __name__=='__main__':
    main()
