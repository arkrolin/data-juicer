import unittest

from data_juicer.ops.common.helper_func import (
    UnionFind,
    get_sentences_from_document,
    get_words_from_document,
    merge_on_whitespace_tab_newline,
    split_on_newline_tab_whitespace,
    split_on_whitespace,
    split_text_by_punctuation,
    strip,
    words_augmentation,
    words_refinement,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class UnionFindTest(DataJuicerTestCaseBase):

    def test_single_element(self):
        uf = UnionFind()
        self.assertEqual(uf.find(1), 1)

    def test_union_two_elements(self):
        uf = UnionFind()
        uf.union(1, 2)
        self.assertEqual(uf.find(1), uf.find(2))
        # The parent should be the minimum of the two
        self.assertEqual(uf.find(1), 1)
        self.assertEqual(uf.find(2), 1)

    def test_union_multiple_elements(self):
        uf = UnionFind()
        uf.union(3, 5)
        uf.union(5, 7)
        # All should share the same root (minimum = 3)
        self.assertEqual(uf.find(3), 3)
        self.assertEqual(uf.find(5), 3)
        self.assertEqual(uf.find(7), 3)

    def test_union_already_connected(self):
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(1, 2)
        self.assertEqual(uf.find(1), 1)
        self.assertEqual(uf.find(2), 1)

    def test_path_compression(self):
        uf = UnionFind()
        # Create a chain: 5->4->3->2->1
        uf.union(2, 1)
        uf.union(3, 2)
        uf.union(4, 3)
        uf.union(5, 4)
        # After find with path compression, all should point to 1
        self.assertEqual(uf.find(5), 1)
        # Check path compression happened
        self.assertEqual(uf.parent[5], 1)

    def test_union_disjoint_sets(self):
        uf = UnionFind()
        uf.union(1, 2)
        uf.union(3, 4)
        # Before union, they are separate
        self.assertNotEqual(uf.find(1), uf.find(3))
        # After union
        uf.union(2, 4)
        self.assertEqual(uf.find(1), uf.find(3))
        self.assertEqual(uf.find(1), 1)

    def test_find_initializes_parent(self):
        uf = UnionFind()
        self.assertNotIn(42, uf.parent)
        result = uf.find(42)
        self.assertEqual(result, 42)
        self.assertIn(42, uf.parent)


class StripTest(DataJuicerTestCaseBase):

    def test_basic_strip(self):
        result = strip("  hello  ", {" "})
        self.assertEqual(result, "hello")

    def test_strip_multiple_characters(self):
        result = strip("##hello##", {"#"})
        self.assertEqual(result, "hello")

    def test_strip_mixed_characters(self):
        result = strip("!@hello@!", {"!", "@"})
        self.assertEqual(result, "hello")

    def test_strip_empty_string(self):
        result = strip("", {"a", "b"})
        self.assertEqual(result, "")

    def test_strip_none_document(self):
        result = strip(None, {"a"})
        self.assertIsNone(result)

    def test_strip_all_characters_stripped(self):
        result = strip("aaa", {"a"})
        self.assertEqual(result, "")

    def test_strip_no_characters_to_strip(self):
        result = strip("hello", set())
        self.assertEqual(result, "hello")

    def test_strip_unicode_characters(self):
        result = strip("\U0001f600hello\U0001f600", {"\U0001f600"})
        self.assertEqual(result, "hello")

    def test_strip_only_leading(self):
        result = strip("###hello", {"#"})
        self.assertEqual(result, "hello")

    def test_strip_only_trailing(self):
        result = strip("hello###", {"#"})
        self.assertEqual(result, "hello")

    def test_strip_characters_in_middle_not_removed(self):
        result = strip("he#llo", {"#"})
        self.assertEqual(result, "he#llo")

    def test_strip_cjk_characters(self):
        result = strip("。。你好。。", {"。"})
        self.assertEqual(result, "你好")


class SplitOnWhitespaceTest(DataJuicerTestCaseBase):

    def test_basic_split(self):
        result = split_on_whitespace("hello world")
        self.assertEqual(result, ["hello", "world"])

    def test_multiple_spaces(self):
        result = split_on_whitespace("hello   world")
        self.assertEqual(result, ["hello", "world"])

    def test_no_split_on_newline_by_default(self):
        result = split_on_whitespace("hello\nworld")
        self.assertEqual(result, ["hello\nworld"])

    def test_split_on_newline(self):
        result = split_on_whitespace("hello\nworld", new_line=True)
        self.assertEqual(result, ["hello", "world"])

    def test_split_on_tab(self):
        result = split_on_whitespace("hello\tworld", tab=True)
        self.assertEqual(result, ["hello", "world"])

    def test_split_on_all(self):
        result = split_on_whitespace("hello \n\t world", new_line=True, tab=True)
        self.assertEqual(result, ["hello", "world"])

    def test_empty_string(self):
        result = split_on_whitespace("")
        self.assertEqual(result, [])

    def test_only_spaces(self):
        result = split_on_whitespace("     ")
        self.assertEqual(result, [])

    def test_single_word(self):
        result = split_on_whitespace("hello")
        self.assertEqual(result, ["hello"])

    def test_cjk_text(self):
        result = split_on_whitespace("你好 世界")
        self.assertEqual(result, ["你好", "世界"])

    def test_leading_trailing_spaces(self):
        result = split_on_whitespace("  hello world  ")
        self.assertEqual(result, ["hello", "world"])


class SplitOnNewlineTabWhitespaceTest(DataJuicerTestCaseBase):

    def test_basic_split(self):
        result = split_on_newline_tab_whitespace("hello world")
        self.assertEqual(result, [[["hello", "world"]]])

    def test_newline_split(self):
        result = split_on_newline_tab_whitespace("hello\nworld")
        self.assertEqual(result, [[["hello"]], [["world"]]])

    def test_tab_split(self):
        result = split_on_newline_tab_whitespace("hello\tworld")
        self.assertEqual(result, [[["hello"], ["world"]]])

    def test_combined_split(self):
        result = split_on_newline_tab_whitespace("a b\tc d\ne f")
        self.assertEqual(result, [[["a", "b"], ["c", "d"]], [["e", "f"]]])

    def test_empty_string(self):
        result = split_on_newline_tab_whitespace("")
        self.assertEqual(result, [[[]]])

    def test_only_newlines(self):
        result = split_on_newline_tab_whitespace("\n\n")
        self.assertEqual(result, [[[]], [[]], [[]]])

    def test_complex_document(self):
        doc = "line1 word1 word2\tword3\nline2 word4"
        result = split_on_newline_tab_whitespace(doc)
        self.assertEqual(
            result,
            [[["line1", "word1", "word2"], ["word3"]], [["line2", "word4"]]],
        )


class MergeOnWhitespaceTabNewlineTest(DataJuicerTestCaseBase):

    def test_basic_merge(self):
        sentences = [[["hello", "world"]]]
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "hello world")

    def test_tab_merge(self):
        sentences = [[["hello"], ["world"]]]
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "hello\tworld")

    def test_newline_merge(self):
        sentences = [[["hello"]], [["world"]]]
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "hello\nworld")

    def test_combined_merge(self):
        sentences = [[["a", "b"], ["c", "d"]], [["e", "f"]]]
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "a b\tc d\ne f")

    def test_empty_sentences(self):
        sentences = [[[]]]
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "")

    def test_empty_list(self):
        sentences = []
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "")

    def test_roundtrip(self):
        doc = "hello world\tfoo bar\nline2 a\tb"
        sentences = split_on_newline_tab_whitespace(doc)
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, doc)

    def test_filters_empty_subsentences(self):
        sentences = [[["hello"], [], ["world"]]]
        result = merge_on_whitespace_tab_newline(sentences)
        self.assertEqual(result, "hello\tworld")


class WordsAugmentationTest(DataJuicerTestCaseBase):

    def test_basic_augmentation(self):
        words = ["a", "b", "c"]
        result = words_augmentation(words, group_size=2, join_char="")
        self.assertEqual(result, ["ab", "bc"])

    def test_group_size_3(self):
        words = ["a", "b", "c", "d"]
        result = words_augmentation(words, group_size=3, join_char="")
        self.assertEqual(result, ["abc", "bcd"])

    def test_join_char(self):
        words = ["hello", "world", "foo"]
        result = words_augmentation(words, group_size=2, join_char=" ")
        self.assertEqual(result, ["hello world", "world foo"])

    def test_single_word(self):
        words = ["hello"]
        result = words_augmentation(words, group_size=1, join_char="")
        self.assertEqual(result, ["hello"])

    def test_group_size_equals_length(self):
        words = ["a", "b", "c"]
        result = words_augmentation(words, group_size=3, join_char="")
        self.assertEqual(result, ["abc"])

    def test_group_size_exceeds_length(self):
        words = ["a", "b"]
        result = words_augmentation(words, group_size=3, join_char="")
        self.assertEqual(result, [])

    def test_empty_words(self):
        words = []
        result = words_augmentation(words, group_size=2, join_char="")
        self.assertEqual(result, [])

    def test_cjk_augmentation(self):
        words = ["你", "好", "世", "界"]
        result = words_augmentation(words, group_size=2, join_char="")
        self.assertEqual(result, ["你好", "好世", "世界"])


class GetWordsFromDocumentTest(DataJuicerTestCaseBase):

    def test_basic(self):
        result = get_words_from_document("hello world")
        self.assertEqual(result, ["hello", "world"])

    def test_with_newline(self):
        result = get_words_from_document("hello\nworld", new_line=True)
        self.assertEqual(result, ["hello", "world"])

    def test_with_tab(self):
        result = get_words_from_document("hello\tworld", tab=True)
        self.assertEqual(result, ["hello", "world"])

    def test_with_token_func(self):
        token_func = lambda doc: list(doc)
        result = get_words_from_document("abc", token_func=token_func)
        self.assertEqual(result, ["a", "b", "c"])

    def test_empty_document(self):
        result = get_words_from_document("")
        self.assertEqual(result, [])

    def test_no_newline_split(self):
        result = get_words_from_document("hello\nworld", new_line=False)
        self.assertEqual(result, ["hello\nworld"])


class WordsRefinementTest(DataJuicerTestCaseBase):

    def test_lower_case(self):
        words = ["Hello", "WORLD"]
        result = words_refinement(words, lower_case=True)
        self.assertEqual(result, ["hello", "world"])

    def test_strip_chars(self):
        words = ["#hello#", "!world!"]
        result = words_refinement(words, strip_chars={"#", "!"})
        self.assertEqual(result, ["hello", "world"])

    def test_strip_removes_empty_words(self):
        words = ["###", "hello", "!!!"]
        result = words_refinement(words, strip_chars={"#", "!"})
        self.assertEqual(result, ["hello"])

    def test_words_augmentation(self):
        words = ["a", "b", "c"]
        result = words_refinement(
            words, use_words_aug=True, words_aug_group_sizes=[2], words_aug_join_char=""
        )
        self.assertEqual(result, ["a", "b", "c", "ab", "bc"])

    def test_words_augmentation_multiple_sizes(self):
        words = ["a", "b", "c"]
        result = words_refinement(
            words, use_words_aug=True, words_aug_group_sizes=[2, 3], words_aug_join_char=""
        )
        self.assertEqual(result, ["a", "b", "c", "ab", "bc", "abc"])

    def test_combined_refinement(self):
        words = ["#Hello#", "!WORLD!"]
        result = words_refinement(
            words,
            lower_case=True,
            strip_chars={"#", "!"},
            use_words_aug=True,
            words_aug_group_sizes=[2],
            words_aug_join_char="",
        )
        self.assertEqual(result, ["hello", "world", "helloworld"])

    def test_no_refinement(self):
        words = ["Hello", "World"]
        result = words_refinement(words)
        self.assertEqual(result, ["Hello", "World"])

    def test_empty_words(self):
        result = words_refinement([])
        self.assertEqual(result, [])


class GetSentencesFromDocumentTest(DataJuicerTestCaseBase):

    def test_basic(self):
        result = get_sentences_from_document("line1\nline2\nline3")
        self.assertEqual(result, "line1\nline2\nline3")

    def test_single_line(self):
        result = get_sentences_from_document("hello world")
        self.assertEqual(result, "hello world")

    def test_empty_document(self):
        result = get_sentences_from_document("")
        self.assertEqual(result, "")

    def test_with_model_func(self):
        model_func = lambda doc: doc.split(". ")
        result = get_sentences_from_document("hello. world. foo", model_func=model_func)
        self.assertEqual(result, "hello\nworld\nfoo")

    def test_preserves_lines(self):
        doc = "first\nsecond\nthird"
        result = get_sentences_from_document(doc)
        self.assertEqual(result, doc)


class SplitTextByPunctuationTest(DataJuicerTestCaseBase):

    def test_basic_english_punctuation(self):
        result = split_text_by_punctuation("hello, world")
        self.assertEqual(result, ["hello", "world"])

    def test_multiple_punctuation(self):
        result = split_text_by_punctuation("hello! how are you? fine.")
        self.assertEqual(result, ["hello", "how are you", "fine"])

    def test_chinese_punctuation(self):
        result = split_text_by_punctuation("你好，世界！")
        self.assertEqual(result, ["你好", "世界"])

    def test_no_punctuation(self):
        result = split_text_by_punctuation("hello world")
        self.assertEqual(result, ["hello world"])

    def test_only_punctuation(self):
        result = split_text_by_punctuation("!!!")
        self.assertEqual(result, ["!!!"])

    def test_empty_string(self):
        result = split_text_by_punctuation("")
        self.assertEqual(result, [""])

    def test_mixed_zh_en_punctuation(self):
        result = split_text_by_punctuation("hello，world！foo.bar")
        self.assertEqual(result, ["hello", "world", "foo", "bar"])

    def test_consecutive_punctuation(self):
        result = split_text_by_punctuation("hello...world")
        self.assertEqual(result, ["hello", "world"])

    def test_special_chars(self):
        result = split_text_by_punctuation("a@b#c$d")
        self.assertEqual(result, ["a", "b", "c", "d"])

    def test_parentheses_and_brackets(self):
        result = split_text_by_punctuation("hello(world)[foo]")
        self.assertEqual(result, ["hello", "world", "foo"])


if __name__ == "__main__":
    unittest.main()
