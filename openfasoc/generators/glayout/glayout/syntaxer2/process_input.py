import copy
import io
from pathlib import Path
from typing import Optional, Union
from pathlib import Path

import nltk
import glayout.syntaxer.nltk_init_deps
import glayout.syntaxer.dynamic_load
from glayout.syntaxer.relational import GlayoutCode, parse_direction
from glayout.flow.pdk.sky130_mapped import sky130_mapped_pdk

import importlib.util

class Session:
    """The session stores all relevant information for producing code from a conversation"""

    generic_prompt = """Place a cell, move a cell, route, create a parameter, or define a variable.
You can also dump code or save this conversation, or enter "help" to see supported syntax in detail
What would you like to do?"""

    def __init__(
        self,
        outputstream: io.IOBase,
        inputstream: Optional[io.IOBase] = None,
        toplvlname: Optional[str] = None,
    ):
        """initialize a conversation and greet the user
        Args:
                outputstream (io.IOBase): used to print outputs
                inputstream (io.IOBase): saved for (optionally) reading in user input, also just provide a string
                NOTE: if input stream not provided, str input must be provided
                toplvlname (str): in string only mode, you can input toplvl name using this arg
        """
        self.inputstream = inputstream
        self.outputstream = outputstream
        # greet the user and get a top level name for the component
        if toplvlname is None:
            if inputstream is None:
                raise RuntimeError(
                    "you must specify AT LEAST one of inputstream or name"
                )
            self.print_to_stream("Hello!")
            self.print_to_stream(
                "Please provide a name for the Component you want to create"
            )
            self.print_to_stream(
                "remember, this will be the name of your top level component: "
            )
            self.name = self.read_from_stream().strip()
        else:
            self.name = str(toplvlname).strip()
        # save a list of responses to recreate this component from a .conv file
        self.conversation_responses = list()
        self.conversation_responses.append(self.name)
        # init the code object
        self.code = GlayoutCode(self.name)
        # create a backup that goes back exactly one call to process_next_input
        self.backup = self.__backup()
        # list of supported commands
        self.commands = [
            importlib.import_module("glayout.syntaxer2.Commands.exit"),
            importlib.import_module("glayout.syntaxer2.Commands.help"),
            importlib.import_module("glayout.syntaxer2.Commands.import"),
            importlib.import_module("glayout.syntaxer2.Commands.move"),
            importlib.import_module("glayout.syntaxer2.Commands.route"),
        ]

    def __backup(self):
        """Produce an exact copy of this class to revert after an exception"""
        newobj = self.__class__
        backup = newobj.__new__(newobj)
        backup.inputstream = self.inputstream
        backup.outputstream = self.outputstream
        backup.conversation_responses = copy.deepcopy(self.conversation_responses)
        backup.code = copy.deepcopy(self.code)
        backup.backup = None
        return backup
    
    def add_command(self, command_file: Union[Path, str]):
        """this imports a command and stores it in a list
        Args:
            command_file (Union[Path, str]): path to the command file 
        """
        self.commands.append(importlib.import_module(command_file))
    
    def process_next_input(self, text_input: str) -> bool:
        """main driver for doing things
        returns True if session is ongoing
        """
        self.backup = self.__backup()
        sentences = nltk.sent_tokenize(text_input)
        for sentence in sentences:
            saveresponse = True
            sentence = sentence.strip().removesuffix(".")
            words = nltk.word_tokenize(sentence)
            mode_indicator = words[0].strip().replace("-", "").lower()
            
            if saveresponse:
                self.__save_response(sentence)
            return True
