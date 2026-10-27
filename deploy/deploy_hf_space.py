"""
Put IndicDoc Verify on the web as a Hugging Face Space (free CPU hardware), from this repository.

    1. Make a Hugging Face account, then a token with write access: huggingface.co/settings/tokens
    2. From the repository folder (the Docker image of start.sh has everything this needs):
         docker run --rm -it -v "$PWD:/work" -w /work indicdoc-ai:latest python deploy/deploy_hf_space.py --user <name>
       or, with Python:  pip install huggingface_hub && python deploy/deploy_hf_space.py --user <name>
    3. The Space builds for 5-10 minutes; the app is then at https://<name>-indicdoc-verify.hf.space

The token is read from HF_TOKEN or asked for; it is not stored. Only what the website needs is uploaded: the app,
the API, the ml package, the scripts, the evaluation outputs and the example papers (no reports or slides).
"""
import argparse
import getpass
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = ['app', 'backend', 'ml', 'scripts', 'outputs', 'dataset', '.streamlit', 'requirements.txt',
           'data/processed/sample', 'docs/test_papers']
SKIP_DIRS = {'__pycache__', 'node_modules', '.pytest_cache', '12_tamil_scanned_number_changed'}
SKIP_PATHS = ('outputs/models', 'scripts/verify_bench/tessdata')   # downloaded while the Space image is built
SKIP_SUFFIX = {'.pyc', '.pt', '.pth', '.traineddata', '.log'}
MAX_FILE = 9 * 1024 * 1024


def stage(dest: Path) -> int:
    n = 0
    for item in INCLUDE:
        src = ROOT / item
        if src.is_file():
            files = [src]
        elif src.is_dir():
            files = [p for p in src.rglob('*') if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)
                     and not str(p.relative_to(ROOT).as_posix()).startswith(SKIP_PATHS)]
        else:
            continue
        for f in files:
            if f.suffix in SKIP_SUFFIX or f.stat().st_size > MAX_FILE or f.is_symlink():
                continue
            target = dest / f.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            n += 1
    shutil.copy(ROOT / 'deploy' / 'huggingface' / 'Dockerfile', dest / 'Dockerfile')
    shutil.copy(ROOT / 'deploy' / 'huggingface' / 'README.md', dest / 'README.md')
    return n + 2


def main():
    ap = argparse.ArgumentParser(description='Deploy IndicDoc Verify as a Hugging Face Space')
    ap.add_argument('--user', required=True, help='Hugging Face user (or organisation) name')
    ap.add_argument('--space', default='indicdoc-verify')
    ap.add_argument('--private', action='store_true', help='only you can open the Space')
    args = ap.parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit('pip install huggingface_hub (or run this inside the indicdoc-ai Docker image)')
    token = os.environ.get('HF_TOKEN') or getpass.getpass('Hugging Face token (write access): ').strip()
    api = HfApi(token=token)
    repo_id = f'{args.user}/{args.space}'
    api.create_repo(repo_id, repo_type='space', space_sdk='docker', private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        n = stage(Path(d))
        print(f'uploading {n} files to {repo_id} ...')
        api.upload_folder(folder_path=d, repo_id=repo_id, repo_type='space',
                          commit_message='IndicDoc Verify: deploy from github.com/va-ada/IndicDoc-AI')
    print(f'Space: https://huggingface.co/spaces/{repo_id}  (building, 5-10 minutes)')
    print(f'App:   https://{args.user.lower()}-{args.space}.hf.space')


if __name__ == '__main__':
    main()
