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

def submit_kaggle():
    from fastkaggle.core import iskaggle, push_notebook
    from kaggle import api

    from configs import competition, id, notebook_filename, title, username

    if not iskaggle:
        api.competition_submit_cli('subm.csv', 'initial rn26d 128px', competition)
        push_notebook(
            'khatkeashish', 
            'paddy',
            title='Paddy',
            file='01_eda.ipynb',
            competition=competition, 
            private=False, 
            gpu=True
        )


