"""Run isolated CPU unit tests; never import a simulator entrypoint by default."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numerical', action='store_true', help='also require NumPy/SciPy/CPU PyTorch')
    parser.add_argument('--isaaclab', action='store_true', help='also test walking metrics in the training environment')
    args = parser.parse_args()
    folders = [ROOT/'deployment/isaacsim', ROOT/'experiments/left2']
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(folders[0]), '-p', 'test_go2_straight_controller.py', '-v'],
        [sys.executable, '-m', 'unittest', 'discover', '-s', str(folders[1]), '-p', 'test_left2_accept.py', '-v'],
    ]
    if args.numerical:
        for group, pattern in [('push','test_push_protocol.py'), ('robust','test_robust_curriculum.py'),
                               ('heldout','test_heldout_analysis.py'), ('left2','test_left2_curriculum.py'),
                               ('ablation','test_ablation.py'), ('ablation','test_default7350.py')]:
            commands.append([sys.executable,'-m','unittest','discover','-s',str(ROOT/'experiments'/group),'-p',pattern,'-v'])
    if args.isaaclab:
        commands.append([sys.executable, str(ROOT/'experiments/walking/test_walking_metrics.py'),
                         str(ROOT/'experiments/walking/walking_evaluate.py')])
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['OMP_NUM_THREADS'] = '2'
    folders += [ROOT/'experiments'/g for g in ['push','robust','heldout','left2','ablation']]
    env['PYTHONPATH'] = os.pathsep.join(map(str, folders)) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=120)
    print(f'PASS: {len(commands)} isolated CPU test suites')


if __name__ == '__main__':
    main()
