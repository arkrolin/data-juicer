import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class FileLockReleaseTest(DataJuicerTestCaseBase):
    """Test FileLock._release() edge cases."""

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        super().tearDown()

    @TEST_TAG("standalone")
    def test_release_when_os_remove_raises_oserror(self):
        """FileLock._release() should catch OSError silently when
        os.remove raises it (e.g., lock file already deleted)."""
        from data_juicer.utils.compress import FileLock

        lock_path = os.path.join(self.tmpdir, 'test.lock')

        # Acquire and release the lock normally first to create the file
        lock = FileLock(lock_path)
        lock.acquire()

        # Now mock os.remove to raise OSError during _release
        with patch('data_juicer.utils.compress.os.remove',
                   side_effect=OSError("file already removed")):
            # _release should not raise - it catches OSError silently
            result = lock._release()
            self.assertIsNone(result)

    @TEST_TAG("standalone")
    def test_release_normal_removes_lock_file(self):
        """FileLock._release() should remove the lock file on success."""
        from data_juicer.utils.compress import FileLock

        lock_path = os.path.join(self.tmpdir, 'test_normal.lock')

        lock = FileLock(lock_path)
        lock.acquire()
        self.assertTrue(os.path.exists(lock_path))

        lock._release()
        # Lock file should be removed
        self.assertFalse(os.path.exists(lock_path))


class CacheCompressManagerCompressExistsTest(DataJuicerTestCaseBase):
    """Test CacheCompressManager.compress() when compressed file
    already exists (should skip compression)."""

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        super().tearDown()

    @TEST_TAG("standalone")
    def test_compress_skips_when_compressed_file_exists(self):
        """When the compressed file already exists, compress() should
        skip compression and log debug instead of compressing again."""
        from data_juicer.utils.compress import CacheCompressManager

        manager = CacheCompressManager(compressor_format='zstd')

        # Create a fake cache file
        cache_file = os.path.join(self.tmpdir, 'cache-abc123.arrow')
        with open(cache_file, 'wb') as f:
            f.write(b'fake arrow data')

        # Create the compressed file already (simulating previous compression)
        compressed_file = cache_file + '.zstd'
        with open(compressed_file, 'wb') as f:
            f.write(b'fake compressed data')

        # Create mock datasets
        prev_ds = MagicMock()
        prev_ds.cache_files = [{'filename': cache_file}]
        this_ds = MagicMock()
        this_ds.cache_files = []

        # Patch compress_manager.compress to verify it's NOT called
        with patch.object(manager.compress_manager, 'compress') as mock_compress:
            manager.compress(prev_ds, this_ds)
            mock_compress.assert_not_called()

        # The original file should be removed (cleanup step)
        self.assertFalse(os.path.exists(cache_file))
        # The compressed file should still exist
        self.assertTrue(os.path.exists(compressed_file))


class CacheCompressManagerDecompressExistsTest(DataJuicerTestCaseBase):
    """Test CacheCompressManager.decompress() when decompressed file
    already exists (the else branch logging debug)."""

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        super().tearDown()

    @TEST_TAG("standalone")
    def test_decompress_skips_when_decompressed_file_exists(self):
        """When the decompressed file already exists, decompress() should
        skip decompression and log debug."""
        from data_juicer.utils.compress import CacheCompressManager

        manager = CacheCompressManager(compressor_format='zstd')

        # Create both compressed and decompressed files
        raw_file = os.path.join(self.tmpdir, 'cache-abc123.arrow')
        compressed_file = raw_file + '.zstd'
        with open(raw_file, 'wb') as f:
            f.write(b'raw data')
        with open(compressed_file, 'wb') as f:
            f.write(b'compressed data')

        # Create mock dataset pointing to this cache directory
        ds = MagicMock()
        ds.cache_files = [{'filename': os.path.join(self.tmpdir, 'somefile')}]

        # Patch compress_manager.decompress to verify it's NOT called
        with patch.object(manager.compress_manager, 'decompress') as mock_decompress:
            manager.decompress(ds)
            mock_decompress.assert_not_called()

        # Both files should still exist
        self.assertTrue(os.path.exists(raw_file))
        self.assertTrue(os.path.exists(compressed_file))


class CleanupCompressedCacheFilesTest(DataJuicerTestCaseBase):
    """Test cleanup_compressed_cache_files() edge cases."""

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        super().tearDown()

    @TEST_TAG("standalone")
    def test_cleanup_when_cache_compress_is_none_iterates_all_formats(self):
        """When CACHE_COMPRESS is None, cleanup_compressed_cache_files()
        should iterate all compressor formats."""
        from data_juicer.utils.compress import (
            Compressor,
            cleanup_compressed_cache_files,
        )
        from data_juicer.utils import cache_utils

        original = cache_utils.CACHE_COMPRESS
        try:
            cache_utils.CACHE_COMPRESS = None

            # Create compressed cache files for multiple formats
            for fmt in Compressor.compressors.keys():
                fname = os.path.join(self.tmpdir,
                                     f'cache-test123.arrow.{fmt}')
                with open(fname, 'wb') as f:
                    f.write(b'fake data')

            # Create mock dataset
            ds = MagicMock()
            ds.cache_files = [
                {'filename': os.path.join(self.tmpdir, 'somefile')}
            ]

            cleanup_compressed_cache_files(ds)

            # All compressed files should be removed
            for fmt in Compressor.compressors.keys():
                fname = os.path.join(self.tmpdir,
                                     f'cache-test123.arrow.{fmt}')
                self.assertFalse(os.path.exists(fname),
                                 f"File {fname} should have been removed")
        finally:
            cache_utils.CACHE_COMPRESS = original

    @TEST_TAG("standalone")
    def test_cleanup_basic_flow_with_compressed_files(self):
        """cleanup_compressed_cache_files() should remove all compressed
        cache files in the dataset's cache directory."""
        from data_juicer.utils.compress import cleanup_compressed_cache_files
        from data_juicer.utils import cache_utils

        original = cache_utils.CACHE_COMPRESS
        try:
            cache_utils.CACHE_COMPRESS = 'gzip'

            # Create several compressed cache files
            files = [
                'cache-aaa111.arrow.gzip',
                'cache-bbb222.arrow.gzip',
                'cache-ccc333_00001_of_00003.arrow.gzip',
            ]
            for fname in files:
                path = os.path.join(self.tmpdir, fname)
                with open(path, 'wb') as f:
                    f.write(b'compressed data')

            # Also create a non-cache file that should NOT be removed
            non_cache = os.path.join(self.tmpdir, 'other-file.gzip')
            with open(non_cache, 'wb') as f:
                f.write(b'other data')

            ds = MagicMock()
            ds.cache_files = [
                {'filename': os.path.join(self.tmpdir, 'somefile')}
            ]

            cleanup_compressed_cache_files(ds)

            # All cache-*.gzip files should be removed
            for fname in files:
                path = os.path.join(self.tmpdir, fname)
                self.assertFalse(os.path.exists(path),
                                 f"File {path} should have been removed")

            # Non-cache file should still exist
            self.assertTrue(os.path.exists(non_cache))
        finally:
            cache_utils.CACHE_COMPRESS = original


class BaseCompressorAbstractTest(DataJuicerTestCaseBase):
    """Test that BaseCompressor is abstract and cannot be instantiated."""

    @TEST_TAG("standalone")
    def test_base_compressor_cannot_be_instantiated(self):
        """BaseCompressor is abstract - instantiation should raise
        TypeError."""
        from data_juicer.utils.compress import BaseCompressor

        with self.assertRaises(TypeError):
            BaseCompressor()

    @TEST_TAG("standalone")
    def test_subclass_without_compress_cannot_be_instantiated(self):
        """A subclass of BaseCompressor that does not implement compress()
        should raise TypeError on instantiation."""
        from data_juicer.utils.compress import BaseCompressor

        class IncompleteCompressor(BaseCompressor):
            pass

        with self.assertRaises(TypeError):
            IncompleteCompressor()


class ExtractorBaseExtractTest(DataJuicerTestCaseBase):
    """Test Extractor.extract() with invalid format (KeyError from
    extractors dict)."""

    @TEST_TAG("standalone")
    def test_extractor_extract_invalid_format_raises_key_error(self):
        """Calling Extractor.extract() with an unsupported format should
        raise KeyError from the extractors dict lookup."""
        from data_juicer.utils.compress import Extractor

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, 'input.dat')
            output_path = os.path.join(tmpdir, 'output.dat')
            with open(input_path, 'wb') as f:
                f.write(b'test data')

            with self.assertRaises(KeyError):
                Extractor.extract(input_path, output_path,
                                  'nonexistent_format')


if __name__ == '__main__':
    unittest.main()
