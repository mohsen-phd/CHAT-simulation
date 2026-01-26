"""Main entry point of the program."""

import os
import sys
import termios

from colorama import Fore
from loguru import logger

from hearing_test.test_logic import SpeechInNoise
from util import get_test_manager, get_stimuli_signal, read_conf, save_results

logger.remove(0)
# logger.add(sys.stderr, level="DEBUG")
logger.add(sys.stderr, level="INFO")

import warnings
import pandas as pd
from agents.participant import Participant
import math

warnings.filterwarnings("ignore")


def convert_test_name_presentation_to_config(test_name_presentation: str) -> str:
    """Convert test name presentation to config format."""
    return test_name_presentation.replace(" ", "_").lower()


def preparation(id: int) -> dict[str, str]:
    """Prepare the test and logging system.
    Args:
        id (int): participant id.

    Returns:
        dict: dictionary containing the custom configurations.
    """

    # todo:replace the following lines with the above lines
    participant_id = id
    test_number = 1
    response_capturing_mode = "asr"
    test_name = "chat"
    test_name_presentation = "chat"
    vocalization_mode = "tts"
    test_mode = "test"
    signal_processing = "n"

    # todo: remove the following lines after testing
    if test_name == "asl" and response_capturing_mode == "cli":
        logger.add(sys.stderr, level="DEBUG")

    save_dir = f"records/{participant_id}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"save_dir/{test_name_presentation}", exist_ok=True)

    logger.add(f"{save_dir}/{test_name_presentation}/out.log")
    logger.debug(f"\nParticipant ID: {participant_id}")
    custom_config = {
        "participant_id": participant_id,
        "save_dir": save_dir,
        "test_number": test_number,
        "response_capturing_mode": response_capturing_mode,
        "vocalization_mode": vocalization_mode,
        "test_name": test_name,
        "test_name_presentation": test_name_presentation,
        "test_mode": test_mode,
        "signal_processing": signal_processing,
    }
    return custom_config


def read_configs(custom_config: dict) -> dict:
    """Read config.yaml and update the save location and response capturing mode based on user input.

    Args:
        custom_config (dict): custom configurations provided by the user.

    Returns:
        dict: Configurations.
    """
    configs = read_conf("config.yaml")
    configs["test"][
        "record_save_dir"
    ] = f"{custom_config['save_dir']}/{custom_config['test_name_presentation']}"
    configs["response_capturing"] = custom_config["response_capturing_mode"]
    configs["test_name"] = custom_config["test_name"]
    configs["vocalization_mode"] = custom_config["vocalization_mode"]
    configs["test_name_presentation"] = custom_config["test_name_presentation"]
    configs["test_mode"] = custom_config["test_mode"]
    configs["signal_processing"] = custom_config["signal_processing"]
    logger.debug(f"Config file: {configs}")
    return configs


def run_test(id: int, hearing_level: dict, chat_result: dict):
    """Code entry point."""
    custom_config = preparation(id)
    configs = read_configs(custom_config)
    participant = Participant(id, hearing_level)

    manager = get_test_manager(configs)
    track_results = {}
    snr_db = manager.start_snr
    correct_count = incorrect_count = 0
    iteration = 1
    signal_level = 65
    noise_level = 60
    while not manager.hearing_test.stop_condition() or iteration <= 60:
        os.system("cls" if os.name == "nt" else "clear")
        this_round = {}
        this_round["snr"] = snr_db

        stimuli_id, stimuli_text, response_getting_prompt = (
            manager.test_type.stimuli_generator.get_stimuli(
                test_mode=configs["test_mode"]
            )
        )

        logger.debug(f"{iteration} :The stimuli is: {stimuli_text}")
        this_round["stimuli"] = stimuli_text

        signal_level, noise_level = SpeechInNoise.calculate_noise_signal_level(
            signal_level, noise_level, snr_db
        )
        logger.debug(f"Signal level: {signal_level}, Noise level: {noise_level}")
        sound_wave = get_stimuli_signal(
            manager.test_type,
            stimuli_id,
            manager.noise,
            signal_level=signal_level,
            noise_level=noise_level,
        )

        response = participant.give_response(sound_wave)

        this_round["response"] = response
        matched = manager.test_type.stimuli_generator.check_answer(response)
        logger.debug(f"Matched: {matched} \n")
        this_round["matched"] = matched
        if matched:
            correct_count += 1
        else:
            incorrect_count += 1

        new_snr_db = manager.hearing_test.get_next_snr(
            correct_count, incorrect_count, snr_db
        )
        manager.hearing_test.update_variables(matched, snr_db)
        logger.debug(f"New SNR: {new_snr_db}")
        track_results[iteration] = this_round
        if new_snr_db != snr_db:
            snr_db = new_snr_db
            correct_count = incorrect_count = 0
        iteration += 1

    track_results[iteration] = {
        "snr": snr_db,
        "stimuli": stimuli_text,
        "response": response,
        "matched": matched,
    }
    logger.debug(f"Final SRT: {manager.hearing_test.srt} \n")
    track_results["SRT"] = manager.hearing_test.srt
    track_results["config"] = custom_config
    save_results(track_results)
    print(Fore.RED + "End of the test")


if __name__ == "__main__":
    resultd_df = pd.read_excel("combined_pta_chat_results.xlsx")
    included_ids = [39]
    for row in resultd_df.itertuples():
        if row.ID not in included_ids:
            continue
        if math.isnan(row.pta_8000):
            continue
        id = row.ID
        hearing_level = {
            500: row.pta_500,
            1000: row.pta_1000,
            2000: row.pta_2000,
            4000: row.pta_4000,
            8000: row.pta_8000,
        }

        # hearing_level = {
        #     500: 15,
        #     1000: 35,
        #     2000: 70,
        #     4000: 75,
        #     8000: 110,
        # }
        chat_result = {"avg": row.chat_avg, "chat_1": row.chat_1, "chat_2": row.chat_2}
        run_test(id, hearing_level, chat_result)
