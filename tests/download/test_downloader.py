import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from data_juicer.download.downloader import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    _download_and_extract_single_partition,
    download_and_extract,
    validate_snapshot_format,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class _FakeDownloader(DocumentDownloader):
    def __init__(self, content_items):
        super().__init__()
        self.content_items = content_items
        self._downloaded_path = None

    def download(self, url):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
        tmp.write(b'fake')
        tmp.close()
        self._downloaded_path = tmp.name
        return tmp.name


class _FakeIterator(DocumentIterator):
    def __init__(self, items):
        super().__init__()
        self.items = items

    def iterate(self, file_path):
        for meta, content in self.items:
            yield meta, content


class _FakeExtractor(DocumentExtractor):
    def __init__(self, extract_fn=None):
        super().__init__()
        self.extract_fn = extract_fn or (lambda c: ({}, c))

    def extract(self, content):
        return self.extract_fn(content)


class TestDownloadAndExtractValidation(DataJuicerTestCaseBase):

    def test_empty_urls_raises(self):
        with self.assertRaises(ValueError) as ctx:
            download_and_extract(
                urls=[],
                output_paths=[],
                downloader=_FakeDownloader([]),
                iterator=_FakeIterator([]),
                extractor=_FakeExtractor(),
                output_format={'text': str},
            )
        self.assertIn('No urls', str(ctx.exception))

    def test_mismatched_urls_and_paths_raises(self):
        with self.assertRaises(ValueError) as ctx:
            download_and_extract(
                urls=['http://a.com/1', 'http://a.com/2'],
                output_paths=['/tmp/out1'],
                downloader=_FakeDownloader([]),
                iterator=_FakeIterator([]),
                extractor=_FakeExtractor(),
                output_format={'text': str},
            )
        self.assertIn('Different number', str(ctx.exception))


class TestDownloadAndExtractSinglePartition(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def test_basic_extraction(self):
        items = [
            ({'id': '1'}, 'hello world'),
            ({'id': '2'}, 'second doc'),
        ]
        downloader = _FakeDownloader(items)
        iterator = _FakeIterator(items)
        extractor = _FakeExtractor(lambda c: ({}, c))

        output_path = os.path.join(self.tmp_dir, 'out.jsonl')
        result = _download_and_extract_single_partition(
            paths=('http://fake.url', output_path),
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            output_type='jsonl',
            keep_raw_download=False,
            force_download=True,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn('text', result.columns)

    def test_item_limit(self):
        items = [({'id': str(i)}, f'doc {i}') for i in range(10)]
        downloader = _FakeDownloader(items)
        iterator = _FakeIterator(items)
        extractor = _FakeExtractor(lambda c: ({}, c))

        output_path = os.path.join(self.tmp_dir, 'out.jsonl')
        result = _download_and_extract_single_partition(
            paths=('http://fake.url', output_path),
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            output_type='jsonl',
            keep_raw_download=True,
            force_download=True,
            item_limit=3,
        )
        self.assertEqual(len(result), 3)

    def test_extractor_returns_none_skips_record(self):
        items = [
            ({'id': '1'}, 'good'),
            ({'id': '2'}, None),
            ({'id': '3'}, 'also good'),
        ]
        downloader = _FakeDownloader(items)
        iterator = _FakeIterator(items)

        def extract_fn(content):
            if content is None:
                return None
            return ({}, content)

        extractor = _FakeExtractor(extract_fn)

        output_path = os.path.join(self.tmp_dir, 'out.jsonl')
        result = _download_and_extract_single_partition(
            paths=('http://fake.url', output_path),
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            output_type='jsonl',
            keep_raw_download=False,
            force_download=True,
        )
        self.assertEqual(len(result), 2)

    def test_extractor_returns_none_text_skips(self):
        items = [({'id': '1'}, 'content')]
        downloader = _FakeDownloader(items)
        iterator = _FakeIterator(items)
        extractor = _FakeExtractor(lambda c: ({}, None))

        output_path = os.path.join(self.tmp_dir, 'out.jsonl')
        result = _download_and_extract_single_partition(
            paths=('http://fake.url', output_path),
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            output_type='jsonl',
            keep_raw_download=False,
            force_download=True,
        )
        self.assertEqual(len(result), 0)

    def test_keep_raw_download_preserves_file(self):
        items = [({'id': '1'}, 'text')]
        downloader = _FakeDownloader(items)
        iterator = _FakeIterator(items)
        extractor = _FakeExtractor(lambda c: ({}, c))

        output_path = os.path.join(self.tmp_dir, 'out.jsonl')
        _download_and_extract_single_partition(
            paths=('http://fake.url', output_path),
            downloader=downloader,
            iterator=iterator,
            extractor=extractor,
            output_type='jsonl',
            keep_raw_download=True,
            force_download=True,
        )
        if downloader._downloaded_path:
            self.assertTrue(os.path.exists(downloader._downloaded_path))
            os.unlink(downloader._downloaded_path)


class TestValidateSnapshotFormatExtra(DataJuicerTestCaseBase):

    def test_none_is_ok(self):
        validate_snapshot_format(None)

    def test_valid_boundary(self):
        validate_snapshot_format('2024-01')
        validate_snapshot_format('2024-53')

    def test_invalid_year_boundary(self):
        with self.assertRaises(ValueError):
            validate_snapshot_format('1999-01')
        with self.assertRaises(ValueError):
            validate_snapshot_format('2101-01')

    def test_week_54_invalid(self):
        with self.assertRaises(ValueError):
            validate_snapshot_format('2024-54')


if __name__ == '__main__':
    unittest.main()
