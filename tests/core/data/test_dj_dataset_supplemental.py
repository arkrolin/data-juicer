"""
Supplemental tests for data_juicer/core/data/dj_dataset.py covering
NestedDataset methods and nested_query that are not covered in the
existing test_dj_dataset.py.
"""
import unittest
from typing import Any, List

from datasets import Dataset

from data_juicer.core.data import NestedDataset
from data_juicer.core.data.dj_dataset import (
    NestedQueryDict,
    add_same_content_to_new_column,
    nested_obj_factory,
    nested_query,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestNestedDatasetFromListAndFromDict(DataJuicerTestCaseBase):
    """Test from_list() and from_dict() class methods."""

    def test_from_list_basic(self):
        data = [
            {'text': 'hello', 'score': 1},
            {'text': 'world', 'score': 2},
        ]
        ds = NestedDataset.from_list(data)
        self.assertIsInstance(ds, NestedDataset)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]['text'], 'hello')
        self.assertEqual(ds[1]['score'], 2)

    def test_from_list_with_nested_meta(self):
        data = [
            {'text': 'hello', '__dj__meta__': {'score': 1, 'lang': 'en'}},
            {'text': 'world', '__dj__meta__': {'score': 2, 'lang': 'es'}},
        ]
        ds = NestedDataset.from_list(data)
        self.assertIsInstance(ds, NestedDataset)
        self.assertEqual(len(ds), 2)
        # Access nested field via dot notation through nested_query
        meta_col = ds['__dj__meta__']
        self.assertEqual(meta_col[0]['score'], 1)
        self.assertEqual(meta_col[1]['lang'], 'es')

    def test_from_list_empty(self):
        ds = NestedDataset.from_list([])
        self.assertIsInstance(ds, NestedDataset)
        self.assertEqual(len(ds), 0)

    def test_from_dict_basic(self):
        data = {
            'text': ['hello', 'world', 'test'],
            'score': [1, 2, 3],
        }
        ds = NestedDataset.from_dict(data)
        self.assertIsInstance(ds, NestedDataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0]['text'], 'hello')
        self.assertEqual(ds[2]['score'], 3)

    def test_from_dict_with_nested_dicts(self):
        data = {
            'text': ['a', 'b'],
            'meta': [{'k': 'v1'}, {'k': 'v2'}],
        }
        ds = NestedDataset.from_dict(data)
        self.assertIsInstance(ds, NestedDataset)
        self.assertEqual(ds[0]['meta'], {'k': 'v1'})


class TestNestedDatasetMapMethod(DataJuicerTestCaseBase):
    """Test map() with nested field access."""

    def test_map_simple_transform(self):
        ds = NestedDataset.from_list([
            {'text': 'hello', 'score': 1},
            {'text': 'world', 'score': 2},
        ])
        result = ds.map(lambda x: {'text': x['text'].upper(), 'score': x['score']})
        self.assertIsInstance(result, NestedDataset)
        self.assertEqual(result[0]['text'], 'HELLO')
        self.assertEqual(result[1]['text'], 'WORLD')

    def test_map_with_nested_dict_access(self):
        ds = NestedDataset.from_list([
            {'text': 'hi', 'meta': {'lang': 'en'}},
            {'text': 'hola', 'meta': {'lang': 'es'}},
        ])
        # Access nested field via x['meta']['lang'] within map
        result = ds.map(lambda x: {
            'text': x['text'],
            'meta': x['meta'],
            'lang_copy': x['meta']['lang'],
        })
        self.assertIsInstance(result, NestedDataset)
        self.assertEqual(result[0]['lang_copy'], 'en')
        self.assertEqual(result[1]['lang_copy'], 'es')

    def test_map_none_function_identity(self):
        ds = NestedDataset.from_list([
            {'text': 'hello', 'val': 42},
        ])
        result = ds.map(None)
        self.assertIsInstance(result, NestedDataset)
        self.assertEqual(result[0]['text'], 'hello')
        self.assertEqual(result[0]['val'], 42)


class TestNestedDatasetSelectAndFilter(DataJuicerTestCaseBase):
    """Test select() and filter() methods."""

    def setUp(self):
        super().setUp()
        self.ds = NestedDataset.from_list([
            {'text': 'a', 'score': 10},
            {'text': 'b', 'score': 20},
            {'text': 'c', 'score': 30},
            {'text': 'd', 'score': 40},
        ])

    def test_select_by_indices(self):
        selected = self.ds.select([0, 2])
        self.assertIsInstance(selected, NestedDataset)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]['text'], 'a')
        self.assertEqual(selected[1]['text'], 'c')

    def test_select_single_index(self):
        selected = self.ds.select([1])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['text'], 'b')

    def test_select_empty(self):
        selected = self.ds.select([])
        self.assertIsInstance(selected, NestedDataset)
        self.assertEqual(len(selected), 0)

    def test_filter_basic(self):
        filtered = self.ds.filter(lambda x: x['score'] > 20)
        self.assertIsInstance(filtered, NestedDataset)
        self.assertEqual(len(filtered), 2)
        texts = filtered['text']
        self.assertIn('c', texts)
        self.assertIn('d', texts)

    def test_filter_no_match(self):
        filtered = self.ds.filter(lambda x: x['score'] > 100)
        self.assertIsInstance(filtered, NestedDataset)
        self.assertEqual(len(filtered), 0)

    def test_filter_all_match(self):
        filtered = self.ds.filter(lambda x: x['score'] > 0)
        self.assertIsInstance(filtered, NestedDataset)
        self.assertEqual(len(filtered), 4)


class TestNestedDatasetAddColumn(DataJuicerTestCaseBase):
    """Test add_column() method."""

    def test_add_column_basic(self):
        ds = NestedDataset.from_list([
            {'text': 'hello'},
            {'text': 'world'},
        ])
        result = ds.add_column('idx', [0, 1])
        self.assertIsInstance(result, NestedDataset)
        self.assertEqual(result[0]['idx'], 0)
        self.assertEqual(result[1]['idx'], 1)
        # Original columns still present
        self.assertEqual(result[0]['text'], 'hello')

    def test_add_column_strings(self):
        ds = NestedDataset.from_list([
            {'val': 1},
            {'val': 2},
        ])
        result = ds.add_column('label', ['pos', 'neg'])
        self.assertEqual(result[0]['label'], 'pos')
        self.assertEqual(result[1]['label'], 'neg')


class TestNestedDatasetSchema(DataJuicerTestCaseBase):
    """Test schema() property."""

    def test_schema_columns(self):
        ds = NestedDataset.from_list([
            {'text': 'hello', 'score': 1, 'tags': [1, 2]},
        ])
        schema = ds.schema()
        self.assertIn('text', schema.columns)
        self.assertIn('score', schema.columns)
        self.assertIn('tags', schema.columns)

    def test_schema_types(self):
        ds = NestedDataset.from_list([
            {'name': 'alice', 'age': 30, 'height': 1.65},
        ])
        schema = ds.schema()
        self.assertEqual(schema.column_types['name'], str)
        self.assertEqual(schema.column_types['age'], int)
        self.assertEqual(schema.column_types['height'], float)

    def test_schema_nested_dict_column(self):
        ds = NestedDataset.from_list([
            {'text': 'x', 'meta': {'k': 'v'}},
        ])
        schema = ds.schema()
        self.assertIn('meta', schema.columns)


class TestNestedQueryFunction(DataJuicerTestCaseBase):
    """Test nested_query() function with nested dicts."""

    def test_nested_query_flat_key(self):
        d = NestedQueryDict({'a': 1, 'b': 2})
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)

    def test_nested_query_dot_notation(self):
        d = NestedQueryDict({'meta': {'lang': 'en', 'source': 'web'}})
        self.assertEqual(d['meta.lang'], 'en')
        self.assertEqual(d['meta.source'], 'web')

    def test_nested_query_deep_nesting(self):
        d = NestedQueryDict({'a': {'b': {'c': 42}}})
        self.assertEqual(d['a.b.c'], 42)

    def test_nested_query_missing_key_returns_none(self):
        d = NestedQueryDict({'a': 1})
        result = d['nonexistent.key']
        self.assertIsNone(result)

    def test_nested_query_on_dataset(self):
        ds = NestedDataset.from_list([
            {'text': 'hi', 'meta': {'score': 99}},
            {'text': 'lo', 'meta': {'score': 1}},
        ])
        # Accessing a nested field on the dataset via dot notation
        # should return list of values
        scores = ds['meta.score']
        self.assertEqual(scores, [99, 1])


class TestAddSameContentToNewColumn(DataJuicerTestCaseBase):
    """Test add_same_content_to_new_column helper."""

    def test_add_new_column_with_initial_value(self):
        sample = {'text': 'hello'}
        result = add_same_content_to_new_column(sample, 'flag', True)
        self.assertEqual(result['flag'], True)
        self.assertEqual(result['text'], 'hello')

    def test_add_new_column_with_none(self):
        sample = {'x': 1}
        result = add_same_content_to_new_column(sample, 'y', None)
        self.assertIsNone(result['y'])

    def test_add_new_column_with_dict(self):
        sample = {'text': 'hi'}
        result = add_same_content_to_new_column(sample, 'meta', {'k': 'v'})
        self.assertEqual(result['meta'], {'k': 'v'})


class TestNestedDatasetOtherMethods(DataJuicerTestCaseBase):
    """Test count(), to_list(), select_columns(), remove_columns()."""

    def setUp(self):
        super().setUp()
        self.ds = NestedDataset.from_list([
            {'text': 'a', 'score': 1, 'tag': 'x'},
            {'text': 'b', 'score': 2, 'tag': 'y'},
        ])

    def test_count(self):
        self.assertEqual(self.ds.count(), 2)

    def test_to_list(self):
        lst = self.ds.to_list()
        self.assertIsInstance(lst, list)
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0]['text'], 'a')

    def test_select_columns(self):
        result = self.ds.select_columns(['text', 'score'])
        self.assertIsInstance(result, NestedDataset)
        self.assertIn('text', result.column_names)
        self.assertIn('score', result.column_names)
        self.assertNotIn('tag', result.column_names)

    def test_remove_columns(self):
        result = self.ds.remove_columns(['tag'])
        self.assertIsInstance(result, NestedDataset)
        self.assertNotIn('tag', result.column_names)
        self.assertIn('text', result.column_names)

    def test_contain_column(self):
        self.assertTrue(self.ds.contain_column('text'))
        self.assertFalse(self.ds.contain_column('nonexistent'))


if __name__ == '__main__':
    unittest.main()
