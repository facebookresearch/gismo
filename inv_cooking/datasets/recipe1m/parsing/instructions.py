from typing import Dict, List, Tuple


class InstructionParser:
    def __init__(self, replace_dict: Dict[str, List[str]]):
        self.replace_dict = replace_dict

    def parse_entry(self, entry) -> Tuple[int, List[str]]:
        """
        Parse a entry containing a list of instructions and return a tuple
        with the total length of parsed text and the list of instructions
        """
        instrs_list = []
        acc_len = 0
        instrs = entry["instructions"]
        for instr in instrs:
            instr = self.clean(instr["text"])
            if len(instr) > 0:
                acc_len += len(instr)
                instrs_list.append(instr)
        return acc_len, instrs_list

    def clean(self, instruction):
        """
        Read an ingredient ingredient and clean the ingredient
        - remove case
        - replace some characters
        - get rid of sentences starting with a digit
        """

        instruction = instruction.lower()

        for rep, char_list in self.replace_dict.items():
            for c_ in char_list:
                if c_ in instruction:
                    instruction = instruction.replace(c_, rep)
            instruction = instruction.strip()

        # remove sentences starting with "1.", "2.", ... from the targets
        if len(instruction) > 0 and instruction[0].isdigit():
            instruction = ""
        return instruction
