"""Class to generate and process the noise signal."""

import random
from abc import ABC, abstractmethod

import numpy as np

from audio_processing.util import (
    convert_to_specific_db_spl,
    read_wav_file,
    rms_amplitude,
)


class Noise(ABC):
    """Abstract class for noise generation."""

    @abstractmethod
    def generate_noise(self, signal: np.ndarray, desired_snr_db: float) -> np.ndarray:
        """Interface for generating noise.

        Args:
            signal (np.ndarray): numpy array containing the signal.
            desired_snr_db (float): desired SNR of the noise relative
                                            to the signal in dB.

        Returns:
            np.ndarray: numpy array containing the noise signal.
        """
        ...

    def _get_noise_amplitude(self, signal: np.ndarray, desired_snr_db: float) -> float:
        """Measure the required amplitude of the noise to achieve the desired SNR.

        Get a signal and an SNR in dB and calculate the amplitude of the noise required
          to achieve the desired SNR.

        Args:
            signal (np.ndarray): numpy array containing the signal.
            desired_snr_db (float): desired SNR of the noise relative

        Returns:
            float: float containing the amplitude of the noise required to achieve
            the desired SNR in dB.
        """
        signal_amplitude = rms_amplitude(signal=signal)
        desired_snr = np.sqrt(10 ** (desired_snr_db / 20))
        noise_amplitude = signal_amplitude / desired_snr
        return noise_amplitude


class WhiteNoise(Noise):
    """A class for generating random gaussian noise with specific db."""

    def generate_noise(self, signal: np.ndarray, desired_snr_db: float) -> np.ndarray:
        """Generate a random gaussian noise signal with a specific SNR.

        Args:
            signal (np.ndarray): numpy array containing the signal.
            desired_snr_db (float): desired SNR of the noise relative
                                            to the signal in dB.

        Returns:
            np.ndarray: numpy array containing the noise signal.
        """
        noise_amplitude = self._get_noise_amplitude(signal, desired_snr_db)
        noise = np.random.normal(scale=noise_amplitude, size=len(signal))

        return noise


class Babble(Noise):
    """A class for generating babble noise from a wave file."""

    def __init__(self, noise_src: str) -> None:
        """Initialize the class and load the noise file.

        Args:
            noise_src (str): path of the wave file containing the noise.
        """
        self._sample_rate, self._noise = read_wav_file(noise_src)

    def generate_noise(
        self,
        signal: np.ndarray,
        noise_db: float,
    ) -> np.ndarray:
        """Generate babble noise with a specific SNR.

        Args:
            signal (np.ndarray): Numpy containing the target signal.
            noise_db (float): Desired level of noise in dB.

        Returns:
            np.ndarray: Generated noise in with the same length as the signal.
        """
        # todo: change back
        scaled_noise = convert_to_specific_db_spl(
            self._noise / (rms_amplitude(self._noise)), noise_db
        )
        signal_len = len(signal)
        noise_start_point = random.randint(0, len(scaled_noise) - signal_len)
        return scaled_noise[noise_start_point : noise_start_point + signal_len]
