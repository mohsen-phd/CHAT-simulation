"""Module for storing question and their validation method."""

import json
import os
import random
from abc import ABC, abstractmethod
import re
from openai import OpenAI
from hearing_test.util import lemmatizer, remove_contractions
from loguru import logger


class Questions(ABC):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        self.question = ""
        self.main_words = []
        self.question_id = []
        self.previous_stimuli = set()

    @abstractmethod
    def check_answer(self, answer: str) -> bool:
        """Based on question type, check if the answer is correct or not.

        Args:
            answer (str): answer to the question given by the patient.

        Returns:
            bool: Is a match or not.
        """
        pass

    @abstractmethod
    def get_stimuli(self, test_mode: str) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Args:
            test_mode (str): is it real test or practice test. ('test' or 'practice')

        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli.
        """
        pass

    def _read_json(self, src: str) -> dict:
        """Read the json file and return the data.

        Args:
            src (str): json file location

        Returns:
            dict: data in the json file
        """
        with open(src) as f:
            data = json.load(f)
        return data


class ASLQuestions(Questions):
    """Class for modeling ASL test questions."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_json("media/ASL/sentences.json")

    def get_stimuli(self, test_mode: str) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Generate a sample stimuli consist of three words
        by randomly selecting from the list of vocab.

        Args:
            test_mode (str): is it real test or practice test. ('test' or 'practice')

        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli. The prompt to show the user when
            asking for a response.
        """
        if test_mode == "test":
            counter = 0
            while True:
                set_num = random.randint(2, 18)
                question_num = random.randint(1, 15)
                full_question_id = f"{set_num}-{question_num}"
                if full_question_id not in self.previous_stimuli or counter > 80:
                    break
                counter += 1
            self.previous_stimuli.add(full_question_id)
        else:
            set_num = 1
            question_num = random.randint(1, 15)
            full_question_id = f"{set_num}-{question_num}"

        stimulus = self.stimuli_list[full_question_id]
        self.question = stimulus["text"]
        self.main_words = [
            self.question.split()[idx].lower() for idx in stimulus["keywords"]
        ]
        self.question_id = full_question_id
        response_getting_prompt = "Please repeat the sentence you heard."
        return [full_question_id], self.main_words, response_getting_prompt

    def check_answer(self, answer: list[str]) -> bool:
        """Check the given words are the same as the one presented to the patient.

        Args:
            answer (list[str]): The patient's response.

        Returns:
            bool: Is a match or not.
        """
        if answer == ["0"]:
            return False
        elif answer == ["1"]:
            return True
        else:
            return self._check_asr_answer(answer[0])

    def _check_asr_answer(self, answer: str) -> bool:
        """Check the asr output and see if the given answer was correct.

        Args:
            answer (str): The patient's response, transcribed by ASR.

        Returns:
            bool: Is a match or not.
        """
        # todo: asnwer should be lemmatized and processed
        answer_without_space = re.sub(r"\s+", "", answer)

        correct_count = 0
        processed_main_words = lemmatizer(
            remove_contractions(" ".join(self.main_words))
        ).split(" ")
        for org_word, word in zip(self.main_words, processed_main_words):
            if word in answer_without_space or org_word in answer_without_space:
                correct_count += 1

        if correct_count >= (len(self.main_words) - 1):
            return True
        else:
            return False


class CHATQuestions(Questions):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_chat_json("media/CHAT/text")
        self.chatGPT = self._get_openai_client()

    def _get_openai_client(self) -> OpenAI:
        """Read the openai api key the text file and create a OpenAI client.

        Returns:
            OpenAI: OpenAI client.
        """
        with open("keys/openai.txt") as f:
            api_key = f.read()
        return OpenAI(api_key=api_key)

    def _read_chat_json(self, root_src: str) -> dict:
        """Read all the json file in the root and marge them into one dictionary.

        Args:
            root_src (str): src of root directory.

        Returns:
            dict: Merged json file.
        """
        data = {}
        for root, _, files in os.walk(root_src):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    cat_id = file.split("-")[0]
                    data[cat_id] = json.load(f)
        return data

    def check_answer(self, answer: list[str]) -> bool:
        """Based on question type, check if the answer is correct or not.

        Args:
            answer (list[str]): answer to the question given by the patient.

        Returns:
            bool: Is a match or not.
        """
        if answer[0] == "":
            return False
        statement = self.main_words[0]
        question = self.main_words[1]

        prompt = f"""Consider that you are an audiologist that wants to decided if the patient understood the presented sentence in noise based on their response to the presented question. 
        Statement is:  '{statement}'. Question is: '{question}'. Response is: '{answer[0]}'. Based on the statement and question, is the response correct. 
        Consider these: 
        1)  Ignore the plural and singular differences.
        2)  The response does not have to be exactly the same, but cary the same meaning.
        3)  start your response with yes or no and the give full detail.
        4)  ignore the tense of the sentence. 
        5)  ignore the subject of the sentence.
        6)  response can be empty.
        7)  response was captured by an ASR system and there is a chance of error in transcription."""

        chat_completion = self.chatGPT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        response = chat_completion.choices[0].message.content
        logger.debug(f"ChatGPT response: {response}")
        if "yes" in response.lower():
            return True
        else:
            return False

    def get_stimuli(self, test_mode: str) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Args:
            test_mode (str): is it real test or practice test. ('test' or 'practice')
        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli. The prompt to show the user when.
        """
        if test_mode == "test":
            counter = 0
            while True:
                category_num = random.randint(2, 15)
                stimulus_num = random.randint(1, 35)
                full_question_id = f"{category_num}-{stimulus_num}"
                if full_question_id not in self.previous_stimuli or counter > 80:
                    break
                counter += 1
            self.previous_stimuli.add(full_question_id)
        else:
            category_num = 1
            stimulus_num = random.randint(1, 35)
            full_question_id = f"{category_num}-{stimulus_num}"

        stimulus = self.stimuli_list[str(category_num)][str(stimulus_num)]
        self.main_words = [stimulus["statement"], stimulus["question"]]
        self.question_id = full_question_id
        response_getting_prompt = "Please answer the question based on the statement."
        return [full_question_id], self.main_words, response_getting_prompt


class FAAFQuestions(Questions):
    """Abstract class for questions. Each type of question must use this api."""

    def __init__(self) -> None:
        """Initialize the questions object by storing the text of the question."""
        super().__init__()
        self.stimuli_list = self._read_json("media/FAAF/stimuli.json")

    def check_answer(self, answer: list[str]) -> bool:
        """Check the given words are the same as the one presented to the patient.

        Args:
            answer (list[str]): The patient's response.

        Returns:
            bool: Is a match or not.
        """
        try:
            answer_int = int(answer[0])
        except (ValueError, IndexError):
            return False

        self.main_words
        try:
            answer_word = self.question[answer_int - 1]
        except IndexError:
            return False
        if self.main_words == answer_word:
            return True
        else:
            return False

    def get_stimuli(self, test_mode: str) -> tuple[list[str], list[str], str]:
        """Generate a sample stimuli.

        Generate a sample stimuli consist of three words and the target word
        by randomly selecting from the list of vocab.

        Args:
            test_mode (str): is it real test or practice test. ('test' or 'practice')

        Returns:
            tuple[list[str], list[str],str]: stimuli ID and the main words in the stimuli. The prompt to show the user when
            asking for a response.
        """

        if test_mode == "test":
            counter = 0
            while True:
                stimuli_num = str(random.randint(5, 80))
                if stimuli_num not in self.previous_stimuli or counter > 80:
                    break
                counter += 1
            self.previous_stimuli.add(stimuli_num)
        else:
            stimuli_num = str(random.randint(1, 4))

        stimulus = self.stimuli_list[stimuli_num]
        self.question = stimulus["words"]
        self.main_words = stimulus["keyword"]
        self.question_id = stimuli_num
        response_getting_prompt = f"Select the word that was played \n 1) {stimulus['words'][0]} \n 2) {stimulus['words'][1]} \n 3) {stimulus['words'][2]} \n 4) {stimulus['words'][3]}"
        return [stimuli_num], self.main_words, response_getting_prompt
