# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import nltk


class TitleParser:
    def __init__(self):
        pass

    def parse_entry(self, entry) -> Tuple[int, List[str]]:
        """
        Parse a entry containing a list of instructions and return a tuple
        with the total length of parsed text and the list of instructions
        """
        title = entry["title"]
        return nltk.tokenize.word_tokenize(title.lower())
