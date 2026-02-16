# coding: utf-8

from pathlib import Path
import shutil
from unittest.mock import patch

import pytest

from AxonDeepSeg.download_model import (
    download_model,
    SUCCESS_EXIT_CODE,
    MODEL_NOT_FOUND_CODE,
    DOWNLOAD_ERROR_CODE
)
import AxonDeepSeg
import AxonDeepSeg.download_model


class TestCore(object):
    def setup_method(self):
        # Get the directory where this current file is saved
        self.fullPath = Path(__file__).resolve().parent
        # Move up to the test directory, "test/"
        self.testPath = self.fullPath.parent 
        
        # Create temp folder
        # Get the directory where this current file is saved
        self.tmpPath = self.testPath / '__tmp__'
        if not self.tmpPath.exists():
            self.tmpPath.mkdir()

        self.valid_model = 'generalist'
        self.valid_model_path = self.tmpPath / 'model_seg_generalist_light'

    def teardown_method(self):
        # Get the directory where this current file is saved
        fullPath = Path(__file__).resolve().parent

        # Move up to the test directory, "test/"
        testPath = fullPath.parent 
        tmpPath = testPath / '__tmp__'

        
        if tmpPath.exists():
            shutil.rmtree(tmpPath)
            pass

    # --------------download_model tests-------------- #
    @pytest.mark.unit
    def test_download_valid_model_works(self):

        assert not self.valid_model_path.exists()
        download_model(self.valid_model, self.tmpPath)
        assert self.valid_model_path.exists()

    @pytest.mark.unit
    def test_main_cli_runs_succesfully_no_destination(self):
        cli_test_model_path =  Path(AxonDeepSeg.__file__).parent / 'models' / 'model_seg_generalist_light'
        output_dir = download_model(destination=None, overwrite=False)
        assert output_dir == cli_test_model_path

    @pytest.mark.unit
    def test_redownload_model_multiple_times_works(self):

        download_model(self.valid_model, self.tmpPath)
        download_model(self.valid_model, self.tmpPath)
        
    @pytest.mark.unit
    def test_list_models(self):
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(["-l"])
        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == SUCCESS_EXIT_CODE)

    @pytest.mark.unit
    def test_download_model_selects_ensemble_if_necessary(self):
        # Mock download_data to return 0 (success) without downloading ensemble models (1+ Gb)
        def mock_download_data(url):
            # Create the expected folder that download_data would create
            model_folder = Path.cwd() / 'model_seg_generalist_ensemble_fake'
            model_folder.mkdir(exist_ok=True)
            return 0
        
        with patch('AxonDeepSeg.download_model.download_data', side_effect=mock_download_data) as mock_download:
            fake_url = 'https://github.com/axondeepseg/model_seg_generalist/releases/download/r20240416/model_seg_generalist_ensemble.zip'
            
            # Mock get_model_cards to remove single_fold URL (set to None)
            # This forces download_model to fall back to ensemble
            def mock_get_model_cards(path=None):
                return {
                    'generalist': {
                        'full_name': 'model_seg_generalist',
                        'weights': {
                            'single_fold': None,  # Not available
                            'ensemble': fake_url
                        },
                        'n_classes': 2,
                        'model-info': 'Test model',
                        'training-data': 'Test data'
                    }
                }
            
            with patch('AxonDeepSeg.download_model.get_model_cards', side_effect=mock_get_model_cards):
                result = download_model(self.valid_model, self.tmpPath)

            assert result == self.tmpPath / 'model_seg_generalist_ensemble'
            assert mock_download.called
            called_url = mock_download.call_args[0][0]            
            assert called_url == fake_url

    @pytest.mark.unit
    def test_download_model_prefers_light_by_default_when_both_available(self):
        light_url = 'https://example.com/light.zip'
        ensemble_url = 'https://example.com/ensemble.zip'

        def mock_download_data(url):
            model_folder = Path.cwd() / 'model_seg_generalist_light_fake'
            model_folder.mkdir(exist_ok=True)
            return 0

        def mock_get_model_cards(path=None):
            return {
                'generalist': {
                    'full_name': 'model_seg_generalist',
                    'weights': {
                        'single_fold': light_url,
                        'ensemble': ensemble_url
                    },
                    'n_classes': 2,
                    'model-info': 'Test model',
                    'training-data': 'Test data'
                }
            }

        with patch('AxonDeepSeg.download_model.download_data', side_effect=mock_download_data) as mock_download:
            with patch('AxonDeepSeg.download_model.get_model_cards', side_effect=mock_get_model_cards):
                result = download_model(self.valid_model, self.tmpPath, model_type=None)

        assert result == self.tmpPath / 'model_seg_generalist_light'
        called_url = mock_download.call_args[0][0]
        assert called_url == light_url

    @pytest.mark.unit
    def test_download_model_forces_ensemble_when_requested(self):
        light_url = 'https://example.com/light.zip'
        ensemble_url = 'https://example.com/ensemble.zip'

        def mock_download_data(url):
            model_folder = Path.cwd() / 'model_seg_generalist_ensemble_fake'
            model_folder.mkdir(exist_ok=True)
            return 0

        def mock_get_model_cards(path=None):
            return {
                'generalist': {
                    'full_name': 'model_seg_generalist',
                    'weights': {
                        'single_fold': light_url,
                        'ensemble': ensemble_url
                    },
                    'n_classes': 2,
                    'model-info': 'Test model',
                    'training-data': 'Test data'
                }
            }

        with patch('AxonDeepSeg.download_model.download_data', side_effect=mock_download_data) as mock_download:
            with patch('AxonDeepSeg.download_model.get_model_cards', side_effect=mock_get_model_cards):
                result = download_model(self.valid_model, self.tmpPath, model_type='ensemble')

        assert result == self.tmpPath / 'model_seg_generalist_ensemble'
        called_url = mock_download.call_args[0][0]
        assert called_url == ensemble_url

    @pytest.mark.unit
    def test_download_model_fails_if_requested_model_type_is_unavailable(self):
        def mock_get_model_cards(path=None):
            return {
                'generalist': {
                    'full_name': 'model_seg_generalist',
                    'weights': {
                        'single_fold': None,
                        'ensemble': 'https://example.com/ensemble.zip'
                    },
                    'n_classes': 2,
                    'model-info': 'Test model',
                    'training-data': 'Test data'
                }
            }

        with patch('AxonDeepSeg.download_model.get_model_cards', side_effect=mock_get_model_cards):
            with pytest.raises(SystemExit) as pytest_wrapped_e:
                download_model(self.valid_model, self.tmpPath, model_type='light')

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == MODEL_NOT_FOUND_CODE)

    # --------------main (cli) tests-------------- #
    @pytest.mark.integration
    def test_main_cli_runs_succesfully_for_list_models(self):

        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(["--list"])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == SUCCESS_EXIT_CODE)

    @pytest.mark.integration
    def test_main_cli_downloads_to_path(self):
        cli_test_path = self.tmpPath / 'cli_test'
        cli_test_model_path = cli_test_path / 'model_seg_generalist_light'

        AxonDeepSeg.download_model.main(["-d", str(cli_test_path)])

        assert cli_test_model_path.exists()

    @pytest.mark.unit
    def test_main_cli_downloads_ensemble_to_path_when_requested(self):
        cli_test_path = self.tmpPath / 'cli_test'
        cli_test_model_path = cli_test_path / 'model_seg_generalist_ensemble'

        def mock_download_data(_):
            fake_unzipped_folder = Path.cwd() / 'model_seg_generalist_ensemble_fake'
            fake_unzipped_folder.mkdir(exist_ok=True)
            return 0

        with patch('AxonDeepSeg.download_model.download_data', side_effect=mock_download_data):
            with patch('AxonDeepSeg.download_model.get_model_cards', return_value={
                'generalist': {
                    'full_name': 'model_seg_generalist',
                    'weights': {
                        'single_fold': 'https://example.com/light.zip',
                        'ensemble': 'https://example.com/ensemble.zip'
                    },
                    'n_classes': 2,
                    'model-info': 'Test model',
                    'training-data': 'Test data'
                }
            }):
                AxonDeepSeg.download_model.main(["-d", str(cli_test_path), "-t", "ensemble"])

        assert cli_test_model_path.exists()

    @pytest.mark.integration
    def test_main_cli_fails_for_model_that_does_not_exist(self):
        model_name = "no_model"
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            AxonDeepSeg.download_model.main(["-m", model_name])

        assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == MODEL_NOT_FOUND_CODE)

    @pytest.mark.integration
    def test_download_model_fails_when_download_data_fails(self):
        with patch('AxonDeepSeg.download_model.download_data', return_value=1):
            with pytest.raises(SystemExit) as pytest_wrapped_e:
                download_model(self.valid_model, self.tmpPath)
            
            assert (pytest_wrapped_e.type == SystemExit) and (pytest_wrapped_e.value.code == DOWNLOAD_ERROR_CODE)
