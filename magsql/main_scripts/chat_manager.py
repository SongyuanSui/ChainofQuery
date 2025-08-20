# -*- coding: utf-8 -*-
from magsql.main_scripts.MAG import Soft_Schema_linker, Decomposer, Generator, Refiner
from magsql.main_scripts.const import MAX_ROUND, SYSTEM_NAME, SCHEMALINKER_NAME, DECOMPOSER_NAME, REFINER_NAME

from utils.myllm import MyChatGPT

INIT_LOG__PATH_FUNC = None

import time
from pprint import pprint


class ChatManager(object):
    def __init__(self, llm, log_path: str, without_selector: bool=False):
        self.log_path = log_path  # path to record important printed content during running
        self.llm = llm

        self.chat_group = [
            Soft_Schema_linker(llm = self.llm, without_selector = without_selector),
            Decomposer(llm = self.llm),
            Generator(llm = self.llm),
            Refiner(llm = self.llm)
        ]

    def _chat_single_round(self, message: dict):
        # we use `dict` type so value can be changed in the function
        for agent in self.chat_group:
            if message['send_to'] == agent.name:
                try:
                    agent.talk(message)
                except Exception as e:
                    print(f"[ERROR] Agent {agent.name} failed: {e}")
                    raise

    def start(self, user_message: dict):
        # we use `dict` type so value can be changed in the function
        # start_time = time.time()
        if user_message['send_to'] == SYSTEM_NAME:  # in the first round, pass message to prune
            #user_message['send_to'] = SELECTOR_NAME
            user_message['send_to'] = SCHEMALINKER_NAME
        for _ in range(MAX_ROUND):  # start chat in group
            self._chat_single_round(user_message)
            if user_message['send_to'] == SYSTEM_NAME:  # should terminate chat
                break
        # end_time = time.time()
        # exec_time = end_time - start_time
        # print(f"\033[0;34mExecute {exec_time} seconds\033[0m", flush=True)
