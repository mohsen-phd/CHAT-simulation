import numpy as np
from clarity.evaluator.msbg.msbg import Ear
from clarity.evaluator.msbg.msbg_utils import MSBG_FS
from clarity.utils.audiogram import Audiogram, Listener
from audio_processing.util import calculate_db_spl, rms_amplitude
from get_response.asr import ASR, Whisper
from openai import OpenAI
import soundfile as sf
from loguru import logger


class Participant:
    def __init__(self, id: int, hearing_level: dict) -> None:
        self.id = id
        self.hearing_level = hearing_level
        self.ear = self.get_ear()
        self.response_agent = self._get_response_agent()
        self.ASR = self._get_ASR_model()

    def get_ear(self):
        freqs = []
        thresholds_db_hl = []
        for f, t in self.hearing_level.items():
            if t is None:
                continue
            freqs.append(int(f))
            thresholds_db_hl.append(float(t))

        audiogram = Audiogram(
            frequencies=freqs,
            levels=np.array(thresholds_db_hl, dtype=float),
        )

        listener = Listener(audiogram, audiogram)
        ear = Ear(
            src_pos="ff",
            sample_rate=MSBG_FS,
            equiv_0db_spl=65,
            ahr=1.0,
        )
        ear.set_audiogram(listener.audiogram_left)
        return ear

    def process_speech(self, signal_mono):
        return self.ear.process(signal_mono)[0]

    def _get_response_agent(self):
        # Initialize and return the response agent based on configuration
        with open("keys/openai.txt") as f:
            api_key = f.read()
        return OpenAI(api_key=api_key)

    def _get_ASR_model(self) -> ASR:
        # Initialize and return the ASR model based on configuration
        return Whisper()

    def give_response(self, sound_wave) -> list[str]:
        # Use the response agent and ASR model to generate and return the participant's response
        processed_signal = self.process_speech(sound_wave)

        sf.write("output.wav", processed_signal, MSBG_FS)
        transcription = self.ASR.get("output.wav")
        logger.debug(f"Transcription: {transcription}\ n")
        # transcription_2 = self.ASR.get(processed_signal)

        response = self._response_from_agent(transcription)
        return [response]

    def _response_from_agent(self, transcription: str) -> str:
        # Process the transcription to simulate participant's response
        prompt = f"""Consider that you are doing a speech-in-noise test. The test includes a statement 
        and then a question based on that statement. You need to answer the question based on the statement.
          you are representing a person with hearing loss, and the given text represents what you heard.
            So it might be incomplete or incorrect. the statement and question are: {transcription}.
              Answer the question. Do not give uncessesery words in the response.
                You do not need to hear everything perfectly to answer, and it is ok to guess 
    """

        chat_completion = self.response_agent.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        response = chat_completion.choices[0].message.content
        return response
