import unittest

from datasets import Dataset

from data_juicer.core.data.dj_dataset import (
    NestedDataset,
    NestedDatasetDict,
    NestedQueryDict,
    add_same_content_to_new_column,
    nested_obj_factory,
    nested_query,
    wrap_func_with_nested_access,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestNestedQuery(DataJuicerTestCaseBase):

    def test_flat_key_in_nested_query_dict(self):
        d = NestedQueryDict({'name': 'alice', 'age': 30})
        self.assertEqual(nested_query(d, 'name'), 'alice')
        self.assertEqual(nested_query(d, 'age'), 30)

    def test_dotted_key_in_nested_dict(self):
        d = NestedQueryDict({'meta': {'score': 5, 'tag': 'good'}})
        self.assertEqual(nested_query(d, 'meta.score'), 5)
        self.assertEqual(nested_query(d, 'meta.tag'), 'good')

    def test_missing_key_returns_none(self):
        d = NestedQueryDict({'a': 1})
        self.assertIsNone(nested_query(d, 'nonexistent'))

    def test_deeply_nested(self):
        d = NestedQueryDict({'a': {'b': {'c': 42}}})
        self.assertEqual(nested_query(d, 'a.b.c'), 42)

    def test_nested_query_on_dataset(self):
        ds = Dataset.from_dict({'text': ['hello'], 'meta': [{'score': 3}]})
        nds = NestedDataset(ds)
        result = nested_query(nds, 'text')
        self.assertEqual(result, ['hello'])

    def test_nested_query_dataset_dotted(self):
        ds = Dataset.from_dict({
            'text': ['hello'],
            'meta': [{'score': 3, 'info': {'level': 'high'}}],
        })
        nds = NestedDataset(ds)
        result = nested_query(nds, 'meta.score')
        self.assertEqual(result, [3])


class TestNestedObjFactory(DataJuicerTestCaseBase):

    def test_dict_becomes_nested_query_dict(self):
        result = nested_obj_factory({'a': 1})
        self.assertIsInstance(result, NestedQueryDict)

    def test_dataset_becomes_nested_dataset(self):
        ds = Dataset.from_dict({'x': [1, 2]})
        result = nested_obj_factory(ds)
        self.assertIsInstance(result, NestedDataset)

    def test_list_wraps_elements(self):
        result = nested_obj_factory([{'a': 1}, {'b': 2}])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], NestedQueryDict)

    def test_primitive_passes_through(self):
        self.assertEqual(nested_obj_factory(42), 42)
        self.assertEqual(nested_obj_factory('hello'), 'hello')
        self.assertIsNone(nested_obj_factory(None))


class TestAddSameContentToNewColumn(DataJuicerTestCaseBase):

    def test_adds_column_with_none(self):
        sample = {'text': 'hi'}
        result = add_same_content_to_new_column(sample, 'new_col')
        self.assertIn('new_col', result)
        self.assertIsNone(result['new_col'])

    def test_adds_column_with_value(self):
        sample = {'text': 'hi'}
        result = add_same_content_to_new_column(sample, 'score', 0.5)
        self.assertEqual(result['score'], 0.5)

    def test_adds_column_with_dict(self):
        sample = {'text': 'hi'}
        result = add_same_content_to_new_column(sample, 'meta', {})
        self.assertEqual(result['meta'], {})


class TestWrapFuncWithNestedAccess(DataJuicerTestCaseBase):

    def test_wrapped_function_preserves_return(self):
        def my_func(x):
            return x

        wrapped = wrap_func_with_nested_access(my_func)
        result = wrapped({'a': 1})
        self.assertIsInstance(result, NestedQueryDict)

    def test_wrapped_function_with_kwargs(self):
        def my_func(data=None):
            return data

        wrapped = wrap_func_with_nested_access(my_func)
        result = wrapped(data={'key': 'val'})
        self.assertIsInstance(result, NestedQueryDict)


class TestNestedQueryDict(DataJuicerTestCaseBase):

    def test_getitem_flat(self):
        d = NestedQueryDict({'x': 10, 'y': 20})
        self.assertEqual(d['x'], 10)

    def test_getitem_nested(self):
        d = NestedQueryDict({'meta': {'score': 5}})
        self.assertEqual(d['meta.score'], 5)

    def test_nested_list_of_dicts_wrapped(self):
        d = NestedQueryDict({
            'items': [{'name': 'a'}, {'name': 'b'}],
        })
        self.assertIsInstance(d['items'][0], NestedQueryDict)

    def test_getitem_missing_returns_none(self):
        d = NestedQueryDict({'a': 1})
        self.assertIsNone(d['missing_key'])


class TestNestedDatasetMethods(DataJuicerTestCaseBase):

    def test_schema(self):
        ds = Dataset.from_dict({'text': ['hi', 'bye'], 'num': [1, 2]})
        nds = NestedDataset(ds)
        schema = nds.schema()
        self.assertIn('text', schema.columns)
        self.assertIn('num', schema.columns)

    def test_count(self):
        ds = Dataset.from_dict({'text': ['a', 'b', 'c']})
        nds = NestedDataset(ds)
        self.assertEqual(nds.count(), 3)

    def test_get(self):
        ds = Dataset.from_dict({'text': ['a', 'b', 'c', 'd']})
        nds = NestedDataset(ds)
        rows = nds.get(2)
        self.assertEqual(len(rows), 2)

    def test_to_list(self):
        ds = Dataset.from_dict({'text': ['a', 'b']})
        nds = NestedDataset(ds)
        result = nds.to_list()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'a')

    def test_contain_column(self):
        ds = Dataset.from_dict({'text': ['a'], 'num': [1]})
        nds = NestedDataset(ds)
        self.assertTrue(nds.contain_column('text'))
        self.assertFalse(nds.contain_column('nonexistent'))

    def test_from_dict(self):
        nds = NestedDataset.from_dict({'text': ['a', 'b']})
        self.assertIsInstance(nds, NestedDataset)
        self.assertEqual(len(nds), 2)

    def test_select_columns(self):
        ds = Dataset.from_dict({'text': ['a'], 'num': [1], 'extra': [True]})
        nds = NestedDataset(ds)
        result = nds.select_columns(['text', 'num'])
        self.assertIn('text', result.features)
        self.assertNotIn('extra', result.features)

    def test_remove_columns(self):
        ds = Dataset.from_dict({'text': ['a'], 'num': [1]})
        nds = NestedDataset(ds)
        result = nds.remove_columns(['num'])
        self.assertIn('text', result.features)
        self.assertNotIn('num', result.features)


if __name__ == '__main__':
    unittest.main()
