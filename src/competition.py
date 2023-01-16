import os
import zipfile
from pathlib import Path

from fastkaggle import iskaggle
from fastkaggle.core import import_kaggle


def setup_competition(competition, data_path='', install=''):
    "Get a path to data for `competition`, downloading it if needed"
    if iskaggle:
        if install:
            os.system(f'pip install -Uqq {install}')
        return Path('../input')/competition
    else:
        competition_dir = os.path.join(data_path, competition)
        path = Path(competition_dir)
        api = import_kaggle()
        if not path.exists():
            api.competition_download_cli(str(competition), path=path)
            zipfile.ZipFile(f'{competition_dir}/{competition}.zip').extractall(str(f'{competition_dir}'))
        return path

if __name__ == "__main__":
    from configs import competition, data_path
    path = setup_competition(competition, data_path, install='"fastcore>=1.4.5" "fastai>=2.7.1" "timm>=0.6.2.dev0"')
    print(path)
    print(type(path))