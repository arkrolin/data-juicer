import unittest

from data_juicer.ops.base_op import (
    OPERATORS,
    TAGGING_OPS,
    Filter,
    Mapper,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


# ---------------------------------------------------------------------------
# Helper ops used across tests
# ---------------------------------------------------------------------------

class _UpperMapper(Mapper):
    """Simple mapper that uppercases text."""
    _batched_op = True

    def process_batched(self, samples):
        samples['text'] = [t.upper() for t in samples['text']]
        return samples


class _SuffixMapper(Mapper):
    """Simple mapper that appends a suffix."""
    _batched_op = True

    def __init__(self, suffix='_ok', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suffix = suffix

    def process_batched(self, samples):
        samples['text'] = [t + self.suffix for t in samples['text']]
        return samples


class _LengthFilter(Filter):
    """Filter that keeps samples with text length >= min_len."""
    _batched_op = True

    def __init__(self, min_len=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_len = min_len

    def compute_stats_batched(self, samples, **kwargs):
        if Fields.stats not in samples:
            samples[Fields.stats] = [{} for _ in samples['text']]
        for i, t in enumerate(samples['text']):
            samples[Fields.stats][i]['text_len'] = len(t)
        return samples

    def process_batched(self, samples):
        stats = samples.get(Fields.stats, [{}] * len(samples['text']))
        return [s.get('text_len', 0) >= self.min_len for s in stats]


# Register the helper ops so that op_specs mode can find them.
OPERATORS._register_module(module_name='_test_upper_mapper', module_cls=_UpperMapper, force=True)
OPERATORS._register_module(module_name='_test_suffix_mapper', module_cls=_SuffixMapper, force=True)
OPERATORS._register_module(module_name='_test_length_filter', module_cls=_LengthFilter, force=True)


class TestFusedSequentialBatchOp(DataJuicerTestCaseBase):
    """Tests for FusedSequentialBatchOp."""

    def _make_samples(self, texts=None):
        if texts is None:
            texts = ['hello', 'world', 'hi']
        return {
            'text': list(texts),
            Fields.stats: [{} for _ in texts],
        }

    def _get_cls(self):
        from data_juicer.ops.fused_sequential_batch_op import FusedSequentialBatchOp
        return FusedSequentialBatchOp

    # ------------------------------------------------------------------
    # __init__ tests
    # ------------------------------------------------------------------

    @TEST_TAG("standalone")
    def test_init_with_fused_ops(self):
        """Init with a list of pre-built op instances."""
        cls = self._get_cls()
        ops = [_UpperMapper(), _SuffixMapper(suffix='!')]
        fused = cls(fused_ops=ops)
        self.assertEqual(len(fused._fused_ops_input), 2)
        self.assertEqual(fused.op_specs, [])
        self.assertEqual(fused.group_name, 'fused')

    @TEST_TAG("standalone")
    def test_init_with_op_specs(self):
        """Init with op_specs list."""
        cls = self._get_cls()
        specs = [
            {'class_name': '_test_upper_mapper', 'kwargs': {}},
            {'class_name': '_test_suffix_mapper', 'kwargs': {'suffix': '!'}},
        ]
        fused = cls(op_specs=specs)
        self.assertIsNone(fused._fused_ops_input)
        self.assertEqual(len(fused.op_specs), 2)

    @TEST_TAG("standalone")
    def test_init_raises_when_both_provided(self):
        """ValueError raised when both fused_ops and op_specs provided."""
        cls = self._get_cls()
        ops = [_UpperMapper()]
        specs = [{'class_name': '_test_upper_mapper', 'kwargs': {}}]
        with self.assertRaises(ValueError):
            cls(fused_ops=ops, op_specs=specs)

    @TEST_TAG("standalone")
    def test_init_group_name_default(self):
        """group_name defaults to 'fused' when not provided."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper()])
        self.assertEqual(fused.group_name, 'fused')

    @TEST_TAG("standalone")
    def test_init_group_name_custom(self):
        """group_name uses custom value when provided."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper()], group_name='my_group')
        self.assertEqual(fused.group_name, 'my_group')

    # ------------------------------------------------------------------
    # _detect_tagging_ops tests
    # ------------------------------------------------------------------

    @TEST_TAG("standalone")
    def test_detect_tagging_ops_returns_false_for_non_tagging(self):
        """_detect_tagging_ops returns False for ordinary ops."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper(), _SuffixMapper()])
        self.assertFalse(fused._contains_tagging_ops)

    @TEST_TAG("standalone")
    def test_detect_tagging_ops_returns_true_for_tagging_op(self):
        """_detect_tagging_ops returns True when a tagging op is present."""
        cls = self._get_cls()
        # Register a fake tagging op
        TAGGING_OPS._register_module(
            module_name='_test_tagging_mapper', module_cls=_UpperMapper, force=True
        )
        try:
            # Create an instance whose _name matches the tagging registry
            tagging_op = _UpperMapper()
            tagging_op._name = '_test_tagging_mapper'
            fused = cls(fused_ops=[tagging_op])
            self.assertTrue(fused._contains_tagging_ops)
        finally:
            # Cleanup
            TAGGING_OPS._modules.pop('_test_tagging_mapper', None)

    @TEST_TAG("standalone")
    def test_detect_tagging_ops_with_op_specs(self):
        """_detect_tagging_ops checks op_specs class_name against TAGGING_OPS."""
        cls = self._get_cls()
        TAGGING_OPS._register_module(
            module_name='_test_tagging_mapper', module_cls=_UpperMapper, force=True
        )
        try:
            specs = [{'class_name': '_test_tagging_mapper', 'kwargs': {}}]
            fused = cls(op_specs=specs)
            self.assertTrue(fused._contains_tagging_ops)
        finally:
            TAGGING_OPS._modules.pop('_test_tagging_mapper', None)

    # ------------------------------------------------------------------
    # _ensure_ops tests
    # ------------------------------------------------------------------

    @TEST_TAG("standalone")
    def test_ensure_ops_from_fused_ops(self):
        """_ensure_ops populates _ops from pre-built fused_ops."""
        cls = self._get_cls()
        ops = [_UpperMapper(), _SuffixMapper(suffix='!')]
        fused = cls(fused_ops=ops)
        self.assertIsNone(fused._ops)
        fused._ensure_ops()
        self.assertEqual(len(fused._ops), 2)

    @TEST_TAG("standalone")
    def test_ensure_ops_from_op_specs(self):
        """_ensure_ops builds ops from op_specs using the registry."""
        cls = self._get_cls()
        specs = [
            {'class_name': '_test_upper_mapper', 'kwargs': {}},
            {'class_name': '_test_suffix_mapper', 'kwargs': {'suffix': '_done'}},
        ]
        fused = cls(op_specs=specs)
        fused._ensure_ops()
        self.assertEqual(len(fused._ops), 2)
        # Verify the second op has the correct suffix
        self.assertEqual(fused._ops[1].suffix, '_done')

    @TEST_TAG("standalone")
    def test_ensure_ops_strips_ray_kwargs(self):
        """Ray scheduling kwargs are stripped before constructing sub-ops."""
        cls = self._get_cls()
        specs = [
            {
                'class_name': '_test_suffix_mapper',
                'kwargs': {
                    'suffix': '_x',
                    'num_gpus': 4,
                    'num_proc': 8,
                    'memory': '16G',
                    'runtime_env': {'pip': ['torch']},
                    'cpu_required': 2,
                    'gpu_required': 1,
                    'mem_required': '8G',
                },
            },
        ]
        fused = cls(op_specs=specs)
        fused._ensure_ops()
        # The op should be constructed successfully, Ray kwargs stripped
        self.assertEqual(len(fused._ops), 1)
        self.assertEqual(fused._ops[0].suffix, '_x')

    @TEST_TAG("standalone")
    def test_ensure_ops_raises_for_unknown_op(self):
        """_ensure_ops raises ValueError for unregistered op name."""
        cls = self._get_cls()
        specs = [{'class_name': '_nonexistent_op_xyz', 'kwargs': {}}]
        fused = cls(op_specs=specs)
        with self.assertRaises(ValueError) as ctx:
            fused._ensure_ops()
        self.assertIn('_nonexistent_op_xyz', str(ctx.exception))

    @TEST_TAG("standalone")
    def test_ensure_ops_raises_for_missing_class_name(self):
        """_ensure_ops raises ValueError when spec has no class_name."""
        cls = self._get_cls()
        specs = [{'kwargs': {}}]
        fused = cls(op_specs=specs)
        with self.assertRaises(ValueError) as ctx:
            fused._ensure_ops()
        self.assertIn("missing 'class_name'", str(ctx.exception))

    # ------------------------------------------------------------------
    # process_batched tests
    # ------------------------------------------------------------------

    @TEST_TAG("standalone")
    def test_process_batched_single_mapper(self):
        """process_batched applies a single mapper to all samples."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper()])
        samples = self._make_samples(['hello', 'world', 'foo'])
        result = fused.process_batched(samples)
        self.assertEqual(result['text'], ['HELLO', 'WORLD', 'FOO'])

    @TEST_TAG("standalone")
    def test_process_batched_chained_mappers(self):
        """process_batched chains multiple mappers sequentially."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper(), _SuffixMapper(suffix='!')])
        samples = self._make_samples(['hi', 'there'])
        result = fused.process_batched(samples)
        self.assertEqual(result['text'], ['HI!', 'THERE!'])

    @TEST_TAG("standalone")
    def test_process_batched_mapper_then_filter(self):
        """process_batched with mapper + filter chains correctly."""
        cls = self._get_cls()
        # Upper first, then filter out anything with len < 4
        fused = cls(fused_ops=[_UpperMapper(), _LengthFilter(min_len=4)])
        samples = self._make_samples(['hi', 'world', 'foo'])
        result = fused.process_batched(samples)
        # 'HI' (len 2) filtered out, 'WORLD' (5) and 'FOO' (3) -> FOO is 3, not >= 4
        self.assertEqual(result['text'], ['WORLD'])

    @TEST_TAG("standalone")
    def test_process_batched_empty_batch(self):
        """process_batched returns empty batch when input is empty."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper()])
        samples = {'text': [], Fields.stats: []}
        result = fused.process_batched(samples)
        self.assertEqual(result['text'], [])

    @TEST_TAG("standalone")
    def test_process_batched_no_ops(self):
        """process_batched returns samples unchanged when no ops configured."""
        cls = self._get_cls()
        fused = cls(fused_ops=[])
        samples = self._make_samples(['abc'])
        result = fused.process_batched(samples)
        self.assertEqual(result['text'], ['abc'])

    # ------------------------------------------------------------------
    # cleanup_columns tests
    # ------------------------------------------------------------------

    @TEST_TAG("standalone")
    def test_cleanup_columns_removed_from_output(self):
        """Columns listed in cleanup_columns are removed from output."""
        cls = self._get_cls()
        fused = cls(fused_ops=[_UpperMapper()], cleanup_columns=['_extra'])
        samples = self._make_samples(['hello'])
        samples['_extra'] = ['should_be_removed']
        result = fused.process_batched(samples)
        self.assertNotIn('_extra', result)
        self.assertEqual(result['text'], ['HELLO'])

    # ------------------------------------------------------------------
    # _strip_ray_kwargs (via op_specs) tests
    # ------------------------------------------------------------------

    @TEST_TAG("standalone")
    def test_strip_ray_kwargs_all_removed(self):
        """All Ray scheduling kwargs are stripped from spec kwargs."""
        from data_juicer.ops.fused_sequential_batch_op import _RAY_SCHED_KWARGS
        cls = self._get_cls()
        ray_kwargs = {k: 'dummy' for k in _RAY_SCHED_KWARGS}
        ray_kwargs['suffix'] = '_y'
        specs = [{'class_name': '_test_suffix_mapper', 'kwargs': ray_kwargs}]
        fused = cls(op_specs=specs)
        fused._ensure_ops()
        self.assertEqual(fused._ops[0].suffix, '_y')


if __name__ == '__main__':
    unittest.main()
