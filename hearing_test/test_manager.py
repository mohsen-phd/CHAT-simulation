"""A module to manage and organize the test procedure."""

import os
from abc import ABC, abstractmethod
import time

from colorama import Fore
from loguru import logger

from get_response.asr import ASR, Whisper
from get_response.base import CaptureResponse
from get_response.cli import CLI
from get_response.recorder import Recorder
from hearing_test.test_logic import SpeechInNoise
from hearing_test.test_types import ASL, CHAT, FAAF


class TestManager(ABC):
    """A class to create the hearing test based on config file."""

    def __init__(self, configs: dict) -> None:
        """Initialize the hearing test and other modules.

        Args:
            configs (dict): Path to the configuration file.
        """
        self.conf = configs
        self.hearing_test = SpeechInNoise(
            correct_threshold=self.conf["test"]["correct_threshold"],
            incorrect_threshold=self.conf["test"]["incorrect_threshold"],
            step_size=self.conf["test"]["step_size"],
            reversal_limit=self.conf["test"]["reversal_limit"],
            minimum_threshold=self.conf["test"]["minimum_threshold"],
        )
        self.test_type = self._get_test_type()

        self.response_capturer = self._capture_method()

        self.recorder = Recorder(
            store=True,
            chunk=1024,
            rms_threshold=10,
            timeout_length=3,
            save_dir=configs["test"]["record_save_dir"],
        )

        self.noise = self.test_type.get_noise(self.conf)

        self.start_snr = self.conf["test"]["start_snr"]

    def _get_test_type(self):
        if self.conf["test_name"] == "din":
            return DIN(self.conf)
        elif self.conf["test_name"] == "asl":
            return ASL(self.conf)
        elif self.conf["test_name"] == "faaf":
            return FAAF(self.conf)
        elif self.conf["test_name"] == "chat":
            return CHAT(self.conf)
        raise NotImplementedError

    @abstractmethod
    def get_response(self, prompt: str) -> list[str]:
        """Get the response from the participant.

        Args:
            prompt (str): The prompt to present to the participant.

        Returns:
            list[str]: List of words in the participant's response.
        """
        ...

    @abstractmethod
    def _capture_method(self) -> CaptureResponse:
        """Select the way the participant will give it's response.

        Returns:
            CaptureResponse: Object responsible for capturing response.
        """
        ...


class CliTestManager(TestManager):
    """Test manager for command line test."""

    def __init__(self, configs: dict) -> None:
        """Initialize the command line test manager.

        Args:
            configs (dict): Loaded configuration.
        """
        super().__init__(configs)

    def _capture_method(self) -> CaptureResponse:
        """Return the object that get response from terminal.

        Returns:
            CaptureResponse: Object responsible for capturing response.
        """
        return CLI()

    def get_response(self, prompt: str) -> list[str]:
        """Get the response from the participant.

        Args:
            prompt (str): The prompt to present to the participant.

        Returns:
            list[str]: List of words in the participant's response.
        """
        os.system("cls" if os.name == "nt" else "clear")
        print(Fore.GREEN + prompt)
        logger.debug(prompt)
        time.sleep(0.2)
        listed_response = self.test_type.cli_post_process(self.response_capturer.get())
        logger.debug(listed_response)
        return listed_response


class ASRTestManager(TestManager):
    """Test manager for ASR test."""

    def __init__(self, configs: dict) -> None:
        """Initialize the ASR test manager.

        Args:
            configs (dict): Loaded configuration.
        """
        super().__init__(configs)

    def _capture_method(self) -> CaptureResponse:
        return self.get_asr()

    def get_asr(self) -> ASR:
        """Get the proper asr engine based on config file.

        Raises:
            NotImplementedError: If the asr type is not implemented.

        Returns:
            ASR: The asr engine.
        """

        if self.conf["ml"]["asr_type"] == "whisper":
            return Whisper()

        raise NotImplementedError

    def get_response(self, prompt: str) -> list[str]:
        """Get the response from the participant.

        Args:
            prompt (str): The prompt to present to the participant.

        Returns:
            list[str]: List of words in the participant's response.
        """
        logger.debug(prompt)

        os.system("cls" if os.name == "nt" else "clear")
        print(Fore.GREEN + prompt)

        file_src = self.recorder.listen()

        os.system("cls" if os.name == "nt" else "clear")
        print(Fore.GREEN + "Please Wait...")

        transcribe = self.response_capturer.get(src=file_src).lower()
        logger.debug(transcribe)
        results = self.test_type.asr_post_process(transcribe)
        logger.debug(results)
        try:
            os.remove(file_src.split("/")[-1])
        except FileNotFoundError:
            logger.debug("file not found")
        return results
