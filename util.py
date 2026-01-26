"""Utility module for the main script."""

import json
import os
import numpy as np
import yaml
from loguru import logger
from yaml import YAMLError

from audio_processing.noise import Noise, convert_to_specific_db_spl
from audio_processing.util import calculate_db_spl, rms_amplitude
from hearing_test.test_manager import ASRTestManager, CliTestManager, TestManager
from hearing_test.test_types import TestTypes
from vocalizer.utils import play_sound
from hearing_test.test_logic import SpeechInNoise


def read_conf(src: str = "config.yaml") -> dict:
    """Read the configuration file.

    Args:
        src (str): Path to the configuration file. Defaults to "config.yaml".

    Returns:
        dict: Return the configuration file as a dictionary.

    Raises:
        YAMLError: If an error occurs while reading or parsing the configuration file.
    """
    with open(src, "r") as f:
        try:
            return yaml.safe_load(f)
        except YAMLError as exc:
            logger.error(f"Error while reading the configuration file: {exc}")
            raise exc


def get_stimuli_signal(
    hearing_test: TestTypes,
    stimuli: list[str],
    noise: Noise,
    signal_level: float,
    noise_level: float,
) -> np.ndarray:
    """generate the stimuli and return the sound wave.

    Args:
        hearing_test (TestTypes): hearing test object. is used to get the sound wave.
        stimuli (list[str]): The stimuli to play.
        noise (Noise): object to generate noise.
        signal_level (float): The level of the signal.
        noise_level (float): The level of the noise.
    Returns:
        np.ndarray: The generated sound wave.

    """
    sample_rate, sound_wave_dict = hearing_test.get_sound(stimuli)

    padding_size = sample_rate // 3

    if "noisy" in sound_wave_dict:
        sound_wave_noisy = np.pad(
            sound_wave_dict["noisy"],
            (padding_size, padding_size),
            "constant",
            constant_values=(0, 0),
        )
        noise_signal = noise.generate_noise(sound_wave_noisy, noise_level)
        sound_wave_noisy = convert_to_specific_db_spl(
            sound_wave_noisy / rms_amplitude(sound_wave_noisy), signal_level
        )
        calculate_db_spl(sound_wave_noisy)
        calculate_db_spl(noise_signal)
        noisy_wave = sound_wave_noisy + noise_signal
        return noisy_wave

    if "clean" in sound_wave_dict:
        sound_wave_clean = np.pad(
            sound_wave_dict["clean"],
            (padding_size, padding_size),
            "constant",
            constant_values=(0, 0),
        )
        sound_wave_clean = convert_to_specific_db_spl(sound_wave_clean, signal_level)
        return sound_wave_clean


def get_test_manager(configs: dict) -> TestManager:
    """Return the proper test manager based on config file.

    Args:
        configs (dict): loaded config file.

    Raises:
        NotImplementedError: If the test type is not implemented.

    Returns:
        TestManager: Appropriate test manager.
    """
    if configs["response_capturing"] == "cli":
        return CliTestManager(configs)
    elif configs["response_capturing"] == "asr":
        return ASRTestManager(configs)
    else:
        raise NotImplementedError


def save_results(results: dict):
    """Save the results of the test.

    Args:
        results (dict): Results of the test.
    """
    filename = f"records/snr_results/{results['config']['test_name_presentation']}/{results['config']['participant_id']}.json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(
        filename,
        "w",
    ) as outfile:
        json.dump(results, outfile)
