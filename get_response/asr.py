"""Run ASR and convert audio to text."""

import math
from abc import abstractmethod

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from transformers.utils import logging

from get_response.base import CaptureResponse

logging.set_verbosity_error()


class ASR(CaptureResponse):
    """Interface for the ASR system."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.file_path = ""

    @abstractmethod
    def get(self) -> str:
        """Transcribe the input file.

        Returns:
            str: File transcription.
        """
        ...


class Whisper(ASR):
    """Initialize a Whisper model.

    The model is based on this hugging face repository:
    https://huggingface.co/openai/whisper-large-v3
    """

    def __init__(self) -> None:
        """Initialize a Whisper asr model."""
        super().__init__()

        self.device = "mps"  # "cpu , mps"

        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model_id = "openai/whisper-large-v3-turbo"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def get(self, src: str) -> str:
        """Transcribe the input file.

        Args:
            src (str): File address.

        Returns:
            str: File transcription.
        """
        result = self.pipe(src, generate_kwargs={"language": "english"})
        return result["text"]
