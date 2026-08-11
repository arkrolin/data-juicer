import unittest
from enum import Enum

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FieldsTest(DataJuicerTestCaseBase):

    def test_stats_starts_with_default_prefix(self):
        from data_juicer.utils.constant import DEFAULT_PREFIX, Fields
        self.assertTrue(Fields.stats.startswith(DEFAULT_PREFIX))

    def test_fields_has_expected_attributes(self):
        from data_juicer.utils.constant import Fields
        self.assertTrue(hasattr(Fields, 'stats'))
        self.assertTrue(hasattr(Fields, 'meta'))
        self.assertTrue(hasattr(Fields, 'context'))
        self.assertTrue(hasattr(Fields, 'suffix'))
        self.assertTrue(hasattr(Fields, 'text_tags'))
        self.assertTrue(hasattr(Fields, 'source_file'))


class MetaKeysTest(DataJuicerTestCaseBase):

    def test_metakeys_has_dialog_sentiment_intensity(self):
        from data_juicer.utils.constant import MetaKeys
        self.assertEqual(MetaKeys.dialog_sentiment_intensity,
                         'dialog_sentiment_intensity')

    def test_metakeys_has_video_frame_tags(self):
        from data_juicer.utils.constant import MetaKeys
        self.assertEqual(MetaKeys.video_frame_tags, 'video_frame_tags')


class StatsKeysMetaTest(DataJuicerTestCaseBase):

    def test_getattr_tracks_access(self):
        from data_juicer.utils.constant import StatsKeys, StatsKeysMeta
        # Clear any previous access log
        StatsKeysMeta._accessed_by = {}
        # Access an attribute through StatsKeys (uses the metaclass)
        val = StatsKeys.alnum_ratio
        self.assertEqual(val, 'alnum_ratio')
        # Verify that the access was tracked
        self.assertIn('test_constant_supplemental',
                      StatsKeysMeta._accessed_by)
        self.assertIn('alnum_ratio',
                      StatsKeysMeta._accessed_by['test_constant_supplemental'])

    def test_get_access_log_returns_tracked_dict(self):
        from data_juicer.utils.constant import StatsKeys, StatsKeysMeta
        # Clear and trigger access
        StatsKeysMeta._accessed_by = {}
        _ = StatsKeys.text_len
        log = StatsKeys.get_access_log()
        self.assertIsInstance(log, dict)
        self.assertIn('test_constant_supplemental', log)
        self.assertIn('text_len', log['test_constant_supplemental'])


class HashKeysTest(DataJuicerTestCaseBase):

    def test_hashkeys_has_expected_attributes(self):
        from data_juicer.utils.constant import DEFAULT_PREFIX, HashKeys
        self.assertEqual(HashKeys.hash, DEFAULT_PREFIX + 'hash')
        self.assertEqual(HashKeys.minhash, DEFAULT_PREFIX + 'minhash')
        self.assertEqual(HashKeys.simhash, DEFAULT_PREFIX + 'simhash')
        self.assertTrue(hasattr(HashKeys, 'uid'))
        self.assertTrue(hasattr(HashKeys, 'imagehash'))
        self.assertTrue(hasattr(HashKeys, 'videohash'))
        self.assertTrue(hasattr(HashKeys, 'is_unique'))


class JobRequiredKeysTest(DataJuicerTestCaseBase):

    def test_is_enum(self):
        from data_juicer.utils.constant import JobRequiredKeys
        self.assertTrue(issubclass(JobRequiredKeys, Enum))

    def test_has_expected_members(self):
        from data_juicer.utils.constant import JobRequiredKeys
        expected = ['hook', 'meta_name', 'input', 'output', 'local',
                    'dj_configs', 'extra_configs']
        for name in expected:
            self.assertIn(name, JobRequiredKeys.__members__)


class InterVarsTest(DataJuicerTestCaseBase):

    def test_has_expected_intermediate_variable_names(self):
        from data_juicer.utils.constant import DEFAULT_PREFIX, InterVars
        self.assertEqual(InterVars.lines, DEFAULT_PREFIX + 'lines')
        self.assertEqual(InterVars.words, DEFAULT_PREFIX + 'words')
        self.assertTrue(hasattr(InterVars, 'refined_words'))
        self.assertTrue(hasattr(InterVars, 'loaded_images'))
        self.assertTrue(hasattr(InterVars, 'loaded_audios'))
        self.assertTrue(hasattr(InterVars, 'loaded_videos'))
        self.assertTrue(hasattr(InterVars, 'sampled_frames'))


if __name__ == '__main__':
    unittest.main()
