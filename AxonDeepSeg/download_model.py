import AxonDeepSeg
from AxonDeepSeg.ads_utils import convert_path, download_data
from pathlib import Path
import shutil
from loguru import logger
import sys
import argparse
import yaml
import textwrap
from typing import Literal, Optional

# exit codes
SUCCESS_EXIT_CODE = 0
MODEL_NOT_FOUND_CODE = 1
DOWNLOAD_ERROR_CODE = 2

MODEL_CARDS_PATH = Path(__file__).parent / 'model_cards.yaml'

def _select_model_weights(
        model_info: dict,
        model_name: str,
        model_type: Optional[Literal['light', 'ensemble']],
    ) -> tuple[str, str]:
    """
    Select the model variant to download.

    If model_type is None, default to light if available, otherwise ensemble.
    If model_type is specified, require that exact variant.
    """
    single_fold_url = model_info['weights']['single_fold']
    ensemble_url = model_info['weights']['ensemble']

    if model_type is None:
        # default to single_fold model if available (lighter and faster)
        if single_fold_url is not None:
            return 'light', single_fold_url
        if ensemble_url is not None:
            return 'ensemble', ensemble_url
    elif model_type == 'light':
        if single_fold_url is not None:
            return 'light', single_fold_url
        logger.error(f'Light model is not available for "{model_name}".')
        sys.exit(MODEL_NOT_FOUND_CODE)
    elif model_type == 'ensemble':
        if ensemble_url is not None:
            return 'ensemble', ensemble_url
        logger.error(f'Ensemble model is not available for "{model_name}".')
        sys.exit(MODEL_NOT_FOUND_CODE)

    logger.error(f'No downloadable weights are available for "{model_name}".')
    sys.exit(MODEL_NOT_FOUND_CODE)

def download_model(model_name='generalist', destination=None, overwrite=True, model_type=None):
    '''
    Download a model for AxonDeepSeg.
    Parameters
    ----------
    model_name : str, optional
        Name of the model, by default 'generalist'. 
    destination : str, optional
        Directory to download the model to. Default: None.
    model_type : Literal['light', 'ensemble'] | None, optional
        If provided, forces the selected variant. If omitted, defaults to
        'light' when available, otherwise 'ensemble'.
    '''
    models = get_model_cards(Path(__file__).parent / 'model_cards.yaml')
    if model_name not in models.keys():
        logger.error('Model not found.')
        sys.exit(MODEL_NOT_FOUND_CODE)

    model_suffix, url_model_destination = _select_model_weights(
        model_info=models[model_name],
        model_name=model_name,
        model_type=model_type,
    )

    full_model_name = f'{models[model_name]["full_name"]}_{model_suffix}'
    if destination is None:
        package_dir = Path(AxonDeepSeg.__file__).parent  # Get AxonDeepSeg installation path
        model_destination = package_dir / "models" / full_model_name
        print('Downloading model to default location: {}'.format(model_destination))
    else:
        destination = Path(destination)
        model_destination = destination / full_model_name
    if model_destination.exists() and overwrite == False:
        logger.info("Overwrite set to False - not deleting old model.")
        return model_destination

    files_before = list(Path.cwd().iterdir())
    if download_data(url_model_destination) == 0:
        logger.info("Model downloaded and unzipped succesfully.")
    else:
        logger.error("An error occured. The model was not downloaded.")
        sys.exit(DOWNLOAD_ERROR_CODE)
    files_after = list(Path.cwd().iterdir())

    # retrieving unknown model folder name
    folder_name = list(set(files_after) - set(files_before))[0]
    output_dir = model_destination.resolve()

    if model_destination.exists():
        logger.info("Model folder already existed - deleting old one")
        shutil.rmtree(str(model_destination))

    shutil.move(folder_name, str(model_destination))

    return output_dir

def print_available_models(model_dict: dict):
    '''
    Print all available models for download.
    '''
    logger.info("Printing available models:")
    for model in model_dict:
        to_print = [
            ["Model name", model],
            ["Nb of classes", model_dict[model]['n_classes']],
            ["Model info", model_dict[model]['model-info']],
            ["Training data", model_dict[model]['training-data']],
        ]
        print("\n")
        for label, content in to_print:
            print(label, '\t', textwrap.fill(str(content), width=90).replace('\n', '\n\t\t '))
            
def get_model_cards(model_list_path=MODEL_CARDS_PATH) -> dict:
    '''
    Load the model list from a YAML file.
    '''
    with open(model_list_path, 'r') as f:
        model_dict = yaml.safe_load(f)
    return model_dict

def get_fullnames_of_downloadable_models() -> list:
    '''
    Get the full names of all downloadable models.
    '''
    model_cards = get_model_cards(MODEL_CARDS_PATH)
    downloadable_models = []
    for model_name, model_info in model_cards.items():
        if model_info['weights']['single_fold'] is not None:
            downloadable_models.append(model_info['full_name'] + '_light')
    return downloadable_models


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model-name",
        required=False,
        help="Model to download. Default: generalist",
        default='generalist',
        type=str,
    )
    ap.add_argument(
        "-l", "--list",
        required=False,
        help="List all available models for download",
        default=False,
        action='store_true',
    )
    ap.add_argument(
        "-d", "--dir",
        required=False,
        help="Directory to download the model to. Default: AxonDeepSeg/models",
        default = None,
    )
    ap.add_argument(
        "-t", "--model-type",
        required=False,
        choices=["light", "ensemble"],
        default=None,
        help="Model variant to download. Default: light if available, otherwise ensemble.",
    )
    args = vars(ap.parse_args(argv))

    model_cards = get_model_cards()

    if args["list"]:
        print_available_models(model_cards)
        sys.exit(SUCCESS_EXIT_CODE)
    else:
        download_model(
            model_name=args["model_name"],
            destination=args["dir"],
            overwrite=True,
            model_type=args["model_type"],
        )

if __name__ == "__main__":
    with logger.catch():
        main()
