import os
import json
import unittest

from datasets import config, load_dataset

from data_juicer.core import NestedDataset
from data_juicer.utils.compress import compress, decompress, cleanup_compressed_cache_files, CompressionOff
from data_juicer.utils import cache_utils
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class CacheCompressTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = 'tmp/test_compress/'
        self.test_data_path = self.temp_output_path + 'test.json'
        os.makedirs(self.temp_output_path, exist_ok=True)
        with open(self.test_data_path, 'w') as fout:
            json.dump([{'test_key_1': 'test_val_1'}], fout)
        self.ori_cache_dir = config.HF_DATASETS_CACHE
        config.HF_DATASETS_CACHE = self.temp_output_path

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        config.HF_DATASETS_CACHE = self.ori_cache_dir

        super().tearDown()

    def test_basic_func(self):
        cache_utils.CACHE_COMPRESS = 'zstd'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # won't compress original dataset
        compress(ds, prev_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # compress previous dataset
        compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # decompress the previous dataset
        decompress(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

        # clean up the compressed cache files of the previous dataset
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

    def test_dif_compress_method(self):
        cache_utils.CACHE_COMPRESS = 'gzip'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # won't compress original dataset
        compress(ds, prev_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # compress previous dataset
        compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # decompress the previous dataset
        decompress(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

        # clean up the compressed cache files of the previous dataset
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

    def test_multiprocessing(self):
        cache_utils.CACHE_COMPRESS = 'zstd'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        compress(prev_ds, curr_ds, num_proc=2)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # decompress the previous dataset
        decompress(prev_ds, num_proc=2)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

        # clean up the compressed cache files of the previous dataset
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))

    def test_compression_off(self):
        cache_utils.CACHE_COMPRESS = 'lz4'
        ds = load_dataset('json', data_files=self.test_data_path, split='train')
        prev_ds = ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        curr_ds = prev_ds.map(lambda s: {'test_key_3': 'test_val_3', **s})
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # disable cache compression
        with CompressionOff():
            compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
            self.assertFalse(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

        # re-enable cache compression
        compress(prev_ds, curr_ds)
        # cache files of the original dataset always exist
        for fn in ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))
        # cache files of the previous dataset are deleted
        for fn in prev_ds.cache_files:
            self.assertFalse(os.path.exists(fn['filename']))
            self.assertTrue(os.path.exists(fn['filename'] + f'.{cache_utils.CACHE_COMPRESS}'))
        # cache files of the current dataset are kept
        for fn in curr_ds.cache_files:
            self.assertTrue(os.path.exists(fn['filename']))

    def test_dataset_without_cache(self):
        prev_ds = NestedDataset.from_list([{'test_key': 'test_val'}])
        curr_ds = prev_ds.map(lambda s: {'test_key_2': 'test_val_2', **s})
        # dataset from list does not have cache files
        self.assertTrue(len(prev_ds.cache_files) == 0)
        self.assertTrue(len(curr_ds.cache_files) == 0)
        compress(prev_ds, curr_ds)
        decompress(prev_ds)
        cleanup_compressed_cache_files(prev_ds)
        self.assertTrue(len(prev_ds.cache_files) == 0)
        self.assertTrue(len(curr_ds.cache_files) == 0)


class CacheCompressManagerHelperTest(DataJuicerTestCaseBase):
    """Test helper methods of CacheCompressManager directly."""

    def setUp(self):
        super().setUp()
        from data_juicer.utils.compress import CacheCompressManager
        self.manager = CacheCompressManager(compressor_format='gzip')

    def test_get_raw_filename(self):
        raw = self.manager._get_raw_filename('cache-abc123.arrow.gzip')
        self.assertEqual(raw, 'cache-abc123.arrow')

    def test_get_raw_filename_strips_only_extension(self):
        raw = self.manager._get_raw_filename('/path/to/cache-def.arrow.gzip')
        self.assertEqual(raw, '/path/to/cache-def.arrow')

    def test_get_raw_filename_asserts_on_wrong_extension(self):
        with self.assertRaises(AssertionError):
            self.manager._get_raw_filename('cache-abc.arrow.zstd')

    def test_get_compressed_filename(self):
        compressed = self.manager._get_compressed_filename('cache-abc.arrow')
        self.assertEqual(compressed, 'cache-abc.arrow.gzip')

    def test_get_compressed_filename_with_path(self):
        compressed = self.manager._get_compressed_filename('/data/cache-x.arrow')
        self.assertEqual(compressed, '/data/cache-x.arrow.gzip')

    def test_format_cache_file_name_replaces_shard_number(self):
        name = '/data/cache-abc_00003_of_00010.arrow'
        result = self.manager.format_cache_file_name(name)
        self.assertEqual(result, '/data/cache-abc_*_of_00010.arrow')

    def test_format_cache_file_name_no_shard(self):
        name = '/data/cache-abc.arrow'
        result = self.manager.format_cache_file_name(name)
        self.assertEqual(result, '/data/cache-abc.arrow')

    def test_format_cache_file_name_none_input(self):
        result = self.manager.format_cache_file_name(None)
        self.assertIsNone(result)

    def test_format_cache_file_name_empty_string(self):
        result = self.manager.format_cache_file_name('')
        self.assertEqual(result, '')

    def test_get_cache_file_names_none_directory(self):
        result = self.manager._get_cache_file_names(None)
        self.assertEqual(result, [])

    def test_get_cache_file_names_finds_arrow_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache files
            open(os.path.join(tmpdir, 'cache-abc123.arrow'), 'w').close()
            open(os.path.join(tmpdir, 'cache-def456.arrow'), 'w').close()
            # Create non-cache files that should be excluded
            open(os.path.join(tmpdir, 'other-file.arrow'), 'w').close()
            open(os.path.join(tmpdir, 'cache-ghi.txt'), 'w').close()

            result = self.manager._get_cache_file_names(tmpdir)
            self.assertEqual(sorted(result),
                             ['cache-abc123.arrow', 'cache-def456.arrow'])

    def test_get_cache_file_names_with_fingerprint_string(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, 'cache-abc123.arrow'), 'w').close()
            open(os.path.join(tmpdir, 'cache-def456.arrow'), 'w').close()

            result = self.manager._get_cache_file_names(tmpdir, fingerprints='abc123')
            self.assertEqual(result, ['cache-abc123.arrow'])

    def test_get_cache_file_names_with_fingerprint_list(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, 'cache-abc123.arrow'), 'w').close()
            open(os.path.join(tmpdir, 'cache-def456.arrow'), 'w').close()
            open(os.path.join(tmpdir, 'cache-ghi789.arrow'), 'w').close()

            result = self.manager._get_cache_file_names(
                tmpdir, fingerprints=['abc123', 'ghi789'])
            self.assertEqual(sorted(result),
                             ['cache-abc123.arrow', 'cache-ghi789.arrow'])

    def test_get_cache_file_names_with_custom_extension(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, 'cache-abc.arrow'), 'w').close()
            open(os.path.join(tmpdir, 'cache-abc.arrow.gzip'), 'w').close()

            result = self.manager._get_cache_file_names(
                tmpdir, extension='.gzip')
            self.assertEqual(result, ['cache-abc.arrow.gzip'])


class CompressManagerRoundtripTest(DataJuicerTestCaseBase):
    """Test CompressManager compress/decompress roundtrip with gzip."""

    def test_gzip_roundtrip(self):
        import tempfile
        from data_juicer.utils.compress import CompressManager

        manager = CompressManager(compressor_format='gzip')
        original_content = b'Hello, this is test data for compression roundtrip!'

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.txt')
            compressed_path = os.path.join(tmpdir, 'compressed.txt.gz')
            output_path = os.path.join(tmpdir, 'output.txt')

            # Write original
            with open(input_path, 'wb') as f:
                f.write(original_content)

            # Compress
            manager.compress(input_path, compressed_path)
            self.assertTrue(os.path.exists(compressed_path))

            # Verify compressed file is different from original
            with open(compressed_path, 'rb') as f:
                compressed_content = f.read()
            self.assertNotEqual(compressed_content, original_content)

            # Decompress
            manager.decompress(compressed_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify roundtrip
            with open(output_path, 'rb') as f:
                result_content = f.read()
            self.assertEqual(result_content, original_content)


class FileLockAutoRemoveTest(DataJuicerTestCaseBase):
    """Test that FileLock auto-removes the lock file on release."""

    def test_lock_file_removed_after_context_exit(self):
        import tempfile
        from data_juicer.utils.compress import FileLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, 'test.lock')
            with FileLock(lock_path):
                # Lock file should exist while held
                self.assertTrue(os.path.exists(lock_path))
            # Lock file should be removed after release
            self.assertFalse(os.path.exists(lock_path))

    def test_lock_file_removed_after_explicit_release(self):
        import tempfile
        from data_juicer.utils.compress import FileLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, 'explicit.lock')
            lock = FileLock(lock_path)
            lock.acquire()
            self.assertTrue(os.path.exists(lock_path))
            lock.release()
            self.assertFalse(os.path.exists(lock_path))

    def test_lock_file_already_deleted_no_error(self):
        import tempfile
        from data_juicer.utils.compress import FileLock

        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, 'vanish.lock')
            lock = FileLock(lock_path)
            lock.acquire()
            # Manually remove the lock file before release
            os.remove(lock_path)
            # Release should not raise even though file is gone
            lock.release()
            self.assertFalse(os.path.exists(lock_path))


class CompressorRegistryTest(DataJuicerTestCaseBase):
    """Test that the Compressor registry contains expected formats."""

    def test_registry_contains_gzip(self):
        from data_juicer.utils.compress import Compressor
        self.assertIn('gzip', Compressor.compressors)

    def test_registry_contains_zstd(self):
        from data_juicer.utils.compress import Compressor
        self.assertIn('zstd', Compressor.compressors)

    def test_registry_contains_lz4(self):
        from data_juicer.utils.compress import Compressor
        self.assertIn('lz4', Compressor.compressors)

    def test_registry_maps_to_correct_classes(self):
        from data_juicer.utils.compress import (Compressor, GzipCompressor,
                                                 Lz4Compressor,
                                                 ZstdCompressor)
        self.assertIs(Compressor.compressors['gzip'], GzipCompressor)
        self.assertIs(Compressor.compressors['zstd'], ZstdCompressor)
        self.assertIs(Compressor.compressors['lz4'], Lz4Compressor)

    def test_invalid_format_raises_assertion(self):
        from data_juicer.utils.compress import Compressor
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'dummy.txt')
            output_path = os.path.join(tmpdir, 'dummy.txt.fake')
            with open(input_path, 'wb') as f:
                f.write(b'data')
            with self.assertRaises(AssertionError):
                Compressor.compress(input_path, output_path, 'invalid_format')

    def test_compress_manager_invalid_format_raises_assertion(self):
        from data_juicer.utils.compress import CompressManager
        with self.assertRaises(AssertionError):
            CompressManager(compressor_format='nonexistent')


class ZstdRoundtripTest(DataJuicerTestCaseBase):
    """Test real compress/decompress round-trip with zstd."""

    def test_zstd_roundtrip(self):
        import tempfile
        from data_juicer.utils.compress import CompressManager

        manager = CompressManager(compressor_format='zstd')
        original_content = b'The quick brown fox jumps over the lazy dog. ' * 100

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.txt')
            compressed_path = os.path.join(tmpdir, 'compressed.txt.zst')
            output_path = os.path.join(tmpdir, 'output.txt')

            with open(input_path, 'wb') as f:
                f.write(original_content)

            manager.compress(input_path, compressed_path)
            self.assertTrue(os.path.exists(compressed_path))

            # Compressed file should be smaller than original for repetitive data
            compressed_size = os.path.getsize(compressed_path)
            original_size = os.path.getsize(input_path)
            self.assertLess(compressed_size, original_size)

            manager.decompress(compressed_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, 'rb') as f:
                result = f.read()
            self.assertEqual(result, original_content)

    def test_zstd_empty_file(self):
        import tempfile
        from data_juicer.utils.compress import CompressManager

        manager = CompressManager(compressor_format='zstd')

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'empty.txt')
            compressed_path = os.path.join(tmpdir, 'empty.txt.zst')
            output_path = os.path.join(tmpdir, 'output_empty.txt')

            with open(input_path, 'wb') as f:
                f.write(b'')

            manager.compress(input_path, compressed_path)
            self.assertTrue(os.path.exists(compressed_path))

            manager.decompress(compressed_path, output_path)
            with open(output_path, 'rb') as f:
                result = f.read()
            self.assertEqual(result, b'')

    def test_zstd_binary_data(self):
        import tempfile
        from data_juicer.utils.compress import CompressManager

        manager = CompressManager(compressor_format='zstd')
        # Random-ish binary content
        original_content = bytes(range(256)) * 50

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'binary.bin')
            compressed_path = os.path.join(tmpdir, 'binary.bin.zst')
            output_path = os.path.join(tmpdir, 'output_binary.bin')

            with open(input_path, 'wb') as f:
                f.write(original_content)

            manager.compress(input_path, compressed_path)
            manager.decompress(compressed_path, output_path)

            with open(output_path, 'rb') as f:
                result = f.read()
            self.assertEqual(result, original_content)


class GzipRoundtripTest(DataJuicerTestCaseBase):
    """Test real compress/decompress round-trip with gzip."""

    def test_gzip_roundtrip_large_content(self):
        import tempfile
        from data_juicer.utils.compress import CompressManager

        manager = CompressManager(compressor_format='gzip')
        original_content = b'A' * 10000 + b'B' * 10000 + b'C' * 10000

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'large.txt')
            compressed_path = os.path.join(tmpdir, 'large.txt.gz')
            output_path = os.path.join(tmpdir, 'large_out.txt')

            with open(input_path, 'wb') as f:
                f.write(original_content)

            manager.compress(input_path, compressed_path)
            self.assertTrue(os.path.exists(compressed_path))

            compressed_size = os.path.getsize(compressed_path)
            original_size = os.path.getsize(input_path)
            self.assertLess(compressed_size, original_size)

            manager.decompress(compressed_path, output_path)
            with open(output_path, 'rb') as f:
                result = f.read()
            self.assertEqual(result, original_content)

    def test_gzip_creates_output_directory(self):
        import tempfile
        from data_juicer.utils.compress import CompressManager

        manager = CompressManager(compressor_format='gzip')
        original_content = b'directory creation test'

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.txt')
            # Nested output directory that does not exist yet
            compressed_path = os.path.join(tmpdir, 'sub', 'dir', 'out.gz')
            output_path = os.path.join(tmpdir, 'sub2', 'dir2', 'result.txt')

            with open(input_path, 'wb') as f:
                f.write(original_content)

            manager.compress(input_path, compressed_path)
            self.assertTrue(os.path.exists(compressed_path))

            manager.decompress(compressed_path, output_path)
            with open(output_path, 'rb') as f:
                result = f.read()
            self.assertEqual(result, original_content)


class CleanupCompressedCacheFilesTest(DataJuicerTestCaseBase):
    """Test cleanup_compressed_cache_files removes compressed files."""

    def setUp(self):
        super().setUp()
        self.temp_output_path = 'tmp/test_cleanup_compress/'
        self.test_data_path = self.temp_output_path + 'test.json'
        os.makedirs(self.temp_output_path, exist_ok=True)
        with open(self.test_data_path, 'w') as fout:
            json.dump([{'key': 'value'}], fout)
        from datasets import config
        self.ori_cache_dir = config.HF_DATASETS_CACHE
        config.HF_DATASETS_CACHE = self.temp_output_path

    def tearDown(self):
        from datasets import config
        if os.path.exists(self.temp_output_path):
            os.system(f'rm -rf {self.temp_output_path}')
        config.HF_DATASETS_CACHE = self.ori_cache_dir
        super().tearDown()

    def test_cleanup_removes_compressed_files_with_format_set(self):
        cache_utils.CACHE_COMPRESS = 'zstd'
        ds = load_dataset('json', data_files=self.test_data_path,
                          split='train')
        prev_ds = ds.map(lambda s: {'key2': 'val2', **s})
        curr_ds = prev_ds.map(lambda s: {'key3': 'val3', **s})

        # Compress prev_ds cache files
        compress(prev_ds, curr_ds)

        # Verify compressed files exist
        for fn in prev_ds.cache_files:
            compressed_name = fn['filename'] + '.zstd'
            self.assertTrue(os.path.exists(compressed_name))

        # Cleanup should remove them
        cleanup_compressed_cache_files(prev_ds)
        for fn in prev_ds.cache_files:
            compressed_name = fn['filename'] + '.zstd'
            self.assertFalse(os.path.exists(compressed_name))

    def test_cleanup_when_cache_compress_is_none(self):
        # When CACHE_COMPRESS is None, cleanup should iterate all formats
        cache_utils.CACHE_COMPRESS = 'gzip'
        ds = load_dataset('json', data_files=self.test_data_path,
                          split='train')
        prev_ds = ds.map(lambda s: {'key2': 'val2', **s})
        curr_ds = prev_ds.map(lambda s: {'key3': 'val3', **s})

        compress(prev_ds, curr_ds)

        # Now set to None so cleanup iterates all formats
        cache_utils.CACHE_COMPRESS = None
        cleanup_compressed_cache_files(prev_ds)

        # Verify .gzip files are removed
        for fn in prev_ds.cache_files:
            compressed_name = fn['filename'] + '.gzip'
            self.assertFalse(os.path.exists(compressed_name))


class CompressorClassDirectTest(DataJuicerTestCaseBase):
    """Test the Compressor class compress method directly."""

    def test_compressor_zstd_creates_file(self):
        import tempfile
        from data_juicer.utils.compress import Compressor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'src.txt')
            output_path = os.path.join(tmpdir, 'src.txt.zst')
            with open(input_path, 'wb') as f:
                f.write(b'compressor class test data')

            Compressor.compress(input_path, output_path, 'zstd')
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_compressor_gzip_creates_file(self):
        import tempfile
        from data_juicer.utils.compress import Compressor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'src.txt')
            output_path = os.path.join(tmpdir, 'src.txt.gz')
            with open(input_path, 'wb') as f:
                f.write(b'gzip compressor class test')

            Compressor.compress(input_path, output_path, 'gzip')
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_compressor_creates_output_dir(self):
        import tempfile
        from data_juicer.utils.compress import Compressor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'src.txt')
            output_path = os.path.join(tmpdir, 'nested', 'dir', 'out.zst')
            with open(input_path, 'wb') as f:
                f.write(b'nested directory test')

            Compressor.compress(input_path, output_path, 'zstd')
            self.assertTrue(os.path.exists(output_path))

    def test_compressor_lock_file_cleaned_up(self):
        import tempfile
        from data_juicer.utils.compress import Compressor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'src.txt')
            output_path = os.path.join(tmpdir, 'out.zst')
            lock_path = os.path.join(tmpdir, 'out.lock')
            with open(input_path, 'wb') as f:
                f.write(b'lock cleanup test')

            Compressor.compress(input_path, output_path, 'zstd')
            # Lock file should be removed after compress completes
            self.assertFalse(os.path.exists(lock_path))


if __name__ == '__main__':
    unittest.main()
