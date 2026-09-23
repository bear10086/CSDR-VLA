"""Fetch public simulator resources only; never download policy weights here."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
from csdr_paths import PROJECT,path

def run(*args):subprocess.run(args,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('benchmark',choices=('libero','simpler'));a=p.parse_args()
    if a.benchmark=='libero':
        root=path('libero_repo')
        if root.exists():raise FileExistsError(f'Use an empty destination or an existing configured checkout: {root}')
        root.parent.mkdir(parents=True,exist_ok=True)
        run('git','clone','https://github.com/Lifelong-Robot-Learning/LIBERO.git',str(root))
        run('git','-C',str(root),'checkout','3fc9044a3d16c5142ea449c617af656d820fec85')
    else:
        target=PROJECT/'common/simpler/ManiSkill2_real2sim'
        with tempfile.TemporaryDirectory(prefix='csdr_sim_assets_') as folder:
            source=Path(folder)/'ManiSkill2_real2sim'
            run('git','clone','https://github.com/simpler-env/ManiSkill2_real2sim.git',str(source))
            run('git','-C',str(source),'checkout','cd45dd27dc6bb26d048cb6570cdab4e3f935cc37')
            for relative in ('data','mani_skill2_real2sim/assets'):
                shutil.copytree(source/relative,target/relative,dirs_exist_ok=True)
        print('Simulator code is unchanged; public assets installed.')
if __name__=='__main__':main()
