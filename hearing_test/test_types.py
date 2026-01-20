"""
This module defines various types of hearing tests, including DIN, ASL, FAAF, and CHAT.

Each test type inherits from the abstract base class TestTypes and implements specific methods.
"""

import string
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

from audio_processing.noise import Babble, Noise, WhiteNoise
from audio_processing.util import convert_to_specific_db_spl, read_wav_file
from hearing_test.util import british_to_american, expand_contractions, lemmatizer
from stimuli_generator.questions import (
    ASLQuestions,
    DigitQuestions,
    CHATQuestions,
    FAAFQuestions,
)


class TestTypes(ABC):
    """Abstract class for SIN test types. Each type of test must use this api."""

    def __init__(self, config: dict) -> None:
        """Initialize the TestTypes.

        Args:
            config (dict): configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI.

        Args:
            response (str): response given by participants.

        Returns:
            list[str]: list of words in the response. after post processing.
        """
        ...

    @abstractmethod
    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR.

        Args:
            response (str): response given by participants.

        Returns:
            list[str]: list of words in the response. after post processing.
        """
        ...

    @abstractmethod
    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            str: The noise src for the test.
        """
        ...

    @abstractmethod
    def get_sound(self, stimuli: list[str]) -> np.ndarray:
        """Get the sound for the test.

        Args:
            stimuli (list[str]): List of stimuli.

        Returns:
            np.ndarray: Audio signal.
        """
        ...

    def _get_stimuli_src(self, test_name) -> Path:
        """Get the stimuli src for the test.

        Args:
            test_name (str): The name of the test.

        Returns:
            Path: The stimuli src for the test.

        Raises:
            NotImplementedError: If the stimuli vocalizer type is not supported.
        """
        if self.config["vocalization_mode"] == "tts":
            return Path(
                self.config["test"]["hearing-test"][test_name]["stimuli-recordings-tts"]
            )
        elif self.config["vocalization_mode"] == "recorded":
            return Path(
                self.config["test"]["hearing-test"][test_name][
                    "stimuli-recordings-natural"
                ]
            )
        else:
            raise NotImplementedError


class SignalProcessingType(Enum):
    NOTHING = "n"
    LOW_PASS = "l"
    MODULATION_MATCHED = "m"
    LOW_PASS_MODULATION_MATCHED = "b"


class ASL(TestTypes):
    """Implementing the ASL test."""

    def __init__(self, config: dict) -> None:
        """Initialize the ASL test.

        Args:
            config (dict): configuration dictionary.
        """
        super().__init__(config)

        self.stimuli_generator = ASLQuestions()
        self.stimuli_src = self._get_stimuli_src("ASL")

    def get_sound(
        self, stimuli: list[str]
    ) -> tuple[int | float, dict[str, np.ndarray]]:
        """Use the recorded audio to generate the waveform.

        Args:
            stimuli (dict[str]): A dictionary with two keys 'noisy' and 'clean' containing the audio signal.
                The noise will be added to the signal in the 'noisy' key only. The values are lists of np.ndarray containing the signal.

        Returns:
            tuple[int | float, dict[str, np.ndarray]]: Sampling rate and a dictionary containing the audio signal.
                Dictionary contain two keys 'noisy' and 'clean' containing the audio signal indication noise should be added to which signal.
        """
        stimulus_id = stimuli[0]

        sample_rate, digit_audio = read_wav_file(
            self.stimuli_src / f"{stimulus_id}.wav"
        )
        converted_audio = convert_to_specific_db_spl(digit_audio, 65)

        return sample_rate, {"noisy": converted_audio}

    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI. and remove any common mistakes.

        Args:
            response (str): Response captured by Keyboard.

        Returns:
            list[str]: List of clean words.
        """
        return list(response)[:1]

    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR. and remove any common mistakes.

        Args:
            response (str): Response transcribed by ASR.

        Returns:
            list[str]: list of clean words.
        """
        # Convert to lowercase
        response = response.lower()
        response = expand_contractions(response)
        # Remove punctuation
        response = response.translate(str.maketrans("", "", string.punctuation))
        response = "".join(
            [char for char in response if char.isalpha() or char.isspace()]
        )
        response = british_to_american(response)

        response = response.replace("'s", " is ")
        response = response.replace("'re", " are ")

        response = response.strip()

        # Tokenize
        clean_response = lemmatizer(response)

        return [clean_response]

    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            Noise: Return the proper noise based on config file.

        Raises:
            NotImplementedError: If the noise type is not supported.
        """
        if configs["test"]["hearing-test"]["ASL"]["noise"]["type"] == "white":
            return WhiteNoise()
        elif configs["test"]["hearing-test"]["ASL"]["noise"]["type"] == "babble":
            if configs["vocalization_mode"] == "tts":
                signal_processing = self.config["signal_processing"]

                if signal_processing == SignalProcessingType.NOTHING.value:
                    return Babble(
                        noise_src=configs["test"]["hearing-test"]["ASL"]["noise"][
                            "src_tts"
                        ]
                    )
                elif signal_processing == SignalProcessingType.LOW_PASS.value:
                    return Babble(
                        noise_src=configs["test"]["hearing-test"]["ASL"]["noise"][
                            "src_tts_low_pass"
                        ]
                    )
                elif signal_processing == SignalProcessingType.MODULATION_MATCHED.value:
                    return Babble(
                        noise_src=configs["test"]["hearing-test"]["ASL"]["noise"][
                            "src_tts_modulation_matched"
                        ]
                    )
                elif (
                    signal_processing
                    == SignalProcessingType.LOW_PASS_MODULATION_MATCHED.value
                ):
                    return Babble(
                        noise_src=configs["test"]["hearing-test"]["ASL"]["noise"][
                            "src_tts_low_pass_modulation_matched"
                        ]
                    )
            else:
                return Babble(
                    noise_src=configs["test"]["hearing-test"]["ASL"]["noise"]["src"]
                )
        raise NotImplementedError

    def _get_stimuli_src(self, test_name) -> Path:
        """Get the stimuli src for the test.

        Args:
            test_name (str): The name of the test.

        Returns:
            Path: The stimuli src for the test.

        Raises:
            NotImplementedError: If the stimuli vocalizer type is not supported.
        """
        if self.config["vocalization_mode"] == "tts":
            if self.config["signal_processing"] == SignalProcessingType.NOTHING.value:
                return Path(
                    self.config["test"]["hearing-test"][test_name][
                        "stimuli-recordings-tts"
                    ]
                )
            elif (
                self.config["signal_processing"] == SignalProcessingType.LOW_PASS.value
            ):
                return Path(
                    self.config["test"]["hearing-test"][test_name][
                        "stimuli_recordings_tts_low_pass"
                    ]
                )
            elif (
                self.config["signal_processing"]
                == SignalProcessingType.MODULATION_MATCHED.value
            ):
                return Path(
                    self.config["test"]["hearing-test"][test_name][
                        "stimuli_recordings_tts_modulation_matched"
                    ]
                )
            elif (
                self.config["signal_processing"]
                == SignalProcessingType.LOW_PASS_MODULATION_MATCHED.value
            ):
                return Path(
                    self.config["test"]["hearing-test"][test_name][
                        "stimuli_recordings_tts_low_pass_modulation_matched"
                    ]
                )
        elif self.config["vocalization_mode"] == "recorded":
            return Path(
                self.config["test"]["hearing-test"][test_name][
                    "stimuli-recordings-natural"
                ]
            )
        else:
            raise NotImplementedError


class FAAF(TestTypes):
    """Implementing the FAAF test."""

    def __init__(self, config: dict) -> None:
        """Initialize the FAAF test.

        Args:
            config (dict): configuration dictionary.
        """
        super().__init__(config)

        self.stimuli_generator = FAAFQuestions()
        self.stimuli_src = self._get_stimuli_src("FAAF")

    def get_sound(
        self, stimuli: list[str]
    ) -> tuple[int | float, dict[str, np.ndarray]]:
        """Use the recorded audio to generate the waveform.

        Args:
            stimuli (dict[str]): A dictionary with two keys 'noisy' and 'clean' containing the audio signal.
                The noise will be added to the signal in the 'noisy' key only. The values are lists of np.ndarray containing the signal.

        Returns:
            tuple[int | float, dict[str, np.ndarray]]: Sampling rate and a dictionary containing the audio signal.
                Dictionary contain two keys 'noisy' and 'clean' containing the audio signal indication noise should be added to which signal.
        """

        stimulus_id = stimuli[0]

        sample_rate, digit_audio = read_wav_file(
            self.stimuli_src / f"FAAF{stimulus_id}.wav"
        )
        converted_audio = convert_to_specific_db_spl(digit_audio, 65)

        return sample_rate, {"noisy": converted_audio}

    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI. and remove any common mistakes.

        Args:
            response (str): Response captured by Keyboard.

        Returns:
            list[str]: List of clean words.
        """
        return list(response)[:1]

    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR. and remove any common mistakes.

            FAAF does not have an ASR mode, thus it is not implemented.

        Args:
            response (str): Response transcribed by ASR.

        Raises:
            NotImplementedError: If the ASR mode is used for FAAF test.

        """
        raise NotImplementedError

    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            Noise: Return the proper noise based on config file.

        Raises:
            NotImplementedError: If the noise type is not supported.
        """
        if configs["test"]["hearing-test"]["FAAF"]["noise"]["type"] == "white":
            return WhiteNoise()
        elif configs["test"]["hearing-test"]["FAAF"]["noise"]["type"] == "babble":
            return Babble(
                noise_src=configs["test"]["hearing-test"]["FAAF"]["noise"]["src"]
            )
        raise NotImplementedError


class CHAT(TestTypes):
    """Implementing the CHAT test."""

    def __init__(self, config: dict) -> None:
        """Initialize the CHAT test.

        Args:
            config (dict): configuration dictionary.
        """
        super().__init__(config)

        self.stimuli_generator = CHATQuestions()
        self.stimuli_src = self._get_stimuli_src("CHAT")

    def get_sound(
        self, stimuli: list[str]
    ) -> tuple[int | float, dict[str, np.ndarray]]:
        """Use the recorded audio to generate the waveform.

        Args:
            stimuli (dict[str]): A dictionary with two keys 'noisy' and 'clean' containing the audio signal.
                The noise will be added to the signal in the 'noisy' key only. The values are lists of np.ndarray containing the signal.

        Returns:
            tuple[int | float, dict[str, np.ndarray]]: Sampling rate and a dictionary containing the audio signal.
                Dictionary contain two keys 'noisy' and 'clean' containing the audio signal indication noise should be added to which signal.
        """
        stimulus_id = stimuli[0]

        sample_rate, statement = read_wav_file(
            self.stimuli_src / f"{stimulus_id}-statement.wav"
        )
        converted_audio_statement = convert_to_specific_db_spl(statement, 65)

        sample_rate, question = read_wav_file(
            self.stimuli_src / f"{stimulus_id}-question.wav"
        )
        converted_audio_question = convert_to_specific_db_spl(question, 65)

        combined_stimuli = np.concatenate(
            (converted_audio_statement, converted_audio_question)
        )
        return sample_rate, {"noisy": combined_stimuli}

    def cli_post_process(self, response: str) -> list[str]:
        """Post process the response from the CLI. and remove any common mistakes.

        Args:
            response (str): Response captured by Keyboard.

        Returns:
            list[str]: List of clean words.
        """
        return list(response)[:1]

    def asr_post_process(self, response: str) -> list[str]:
        """Post process the response from the ASR. and remove any common mistakes.

        Args:
            response (str): Response transcribed by ASR.

        Returns:
            list[str]: list of clean words.
        """
        return [response.lower()]

    def get_noise(self, configs: dict) -> Noise:
        """Get the noise for the test.

        Args:
            configs (dict): Configuration dictionary containing test settings.

        Returns:
            Noise: Return the proper noise based on config file.

        Raises:
            NotImplementedError: If the noise type is not supported.
        """
        if configs["test"]["hearing-test"]["CHAT"]["noise"]["type"] == "white":
            return WhiteNoise()
        elif configs["test"]["hearing-test"]["CHAT"]["noise"]["type"] == "babble":
            return Babble(
                noise_src=configs["test"]["hearing-test"]["CHAT"]["noise"]["src-tts"]
            )
        raise NotImplementedError
