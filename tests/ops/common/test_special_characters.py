import unittest

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class SpecialCharactersTest(DataJuicerTestCaseBase):

    def test_special_characters_is_a_set(self):
        from data_juicer.ops.common.special_characters import \
            SPECIAL_CHARACTERS
        self.assertIsInstance(SPECIAL_CHARACTERS, set)

    def test_special_characters_contains_standard_punctuation(self):
        import string
        from data_juicer.ops.common.special_characters import \
            SPECIAL_CHARACTERS
        for ch in string.punctuation:
            self.assertIn(ch, SPECIAL_CHARACTERS)

    def test_special_characters_contains_digits(self):
        import string
        from data_juicer.ops.common.special_characters import \
            SPECIAL_CHARACTERS
        for ch in string.digits:
            self.assertIn(ch, SPECIAL_CHARACTERS)

    def test_special_characters_contains_whitespace(self):
        import string
        from data_juicer.ops.common.special_characters import \
            SPECIAL_CHARACTERS
        for ch in string.whitespace:
            self.assertIn(ch, SPECIAL_CHARACTERS)

    def test_emoji_is_non_empty_list(self):
        from data_juicer.ops.common.special_characters import EMOJI
        self.assertIsInstance(EMOJI, list)
        self.assertGreater(len(EMOJI), 0)

    def test_special_characters_includes_all_main_special_characters(self):
        from data_juicer.ops.common.special_characters import (
            MAIN_SPECIAL_CHARACTERS, SPECIAL_CHARACTERS)
        for ch in MAIN_SPECIAL_CHARACTERS:
            self.assertIn(ch, SPECIAL_CHARACTERS)

    def test_various_whitespaces_is_a_set(self):
        from data_juicer.ops.common.special_characters import \
            VARIOUS_WHITESPACES
        self.assertIsInstance(VARIOUS_WHITESPACES, set)

    def test_various_whitespaces_contains_space_and_tab(self):
        from data_juicer.ops.common.special_characters import \
            VARIOUS_WHITESPACES
        self.assertIn(' ', VARIOUS_WHITESPACES)
        self.assertIn('\t', VARIOUS_WHITESPACES)

    def test_known_emojis_in_special_characters(self):
        from data_juicer.ops.common.special_characters import \
            SPECIAL_CHARACTERS
        # Common well-known emojis
        known_emojis = ['\U0001f600', '\U0001f44d', '❤️']
        for em in known_emojis:
            self.assertIn(em, SPECIAL_CHARACTERS)

    def test_special_characters_count_is_reasonable(self):
        from data_juicer.ops.common.special_characters import \
            SPECIAL_CHARACTERS
        # Due to emoji inclusion the set should have > 500 elements
        self.assertGreater(len(SPECIAL_CHARACTERS), 500)


if __name__ == '__main__':
    unittest.main()
