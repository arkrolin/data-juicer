# Copyright 2025 The Data-Juicer Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from data_juicer.ops.mapper.agent_cross_model_pair_mapper import AgentCrossModelPairMapper
from data_juicer.utils.constant import Fields, MetaKeys


class TestAgentCrossModelPairMapper(unittest.TestCase):
    def test_pairs_same_sample_id(self):
        op = AgentCrossModelPairMapper()
        rows = [
            {
                "id": "a",
                Fields.meta: {
                    "agent_lineage_sample_id": "S1",
                    "agent_lineage_tag_model": "m1",
                    "agent_lineage_tag_quality": 0.9,
                },
            },
            {
                "id": "b",
                Fields.meta: {
                    "agent_lineage_sample_id": "S1",
                    "agent_lineage_tag_model": "m2",
                    "agent_lineage_tag_quality": 0.7,
                },
            },
            {
                "id": "c",
                Fields.meta: {"agent_lineage_sample_id": "", "agent_lineage_tag_model": "m3"},
            },
        ]
        from data_juicer.core.data.dj_dataset import NestedDataset

        ds = NestedDataset.from_list(rows)
        out = op.run(ds)
        lst = out.to_list()
        m0 = lst[0][Fields.meta][MetaKeys.agent_cross_model_pair]
        self.assertTrue(m0["has_pairwise_contrast"])
        self.assertEqual(m0["best_model"], "m1")
        self.assertEqual(m0["my_model"], "m1")
        self.assertIn("m2", m0["peer_models"])
        m1 = lst[1][Fields.meta][MetaKeys.agent_cross_model_pair]
        self.assertEqual(m1["my_model"], "m2")
        self.assertAlmostEqual(m1["delta_to_best"], 0.2, places=5)

    def test_normalized_query_groups_different_sample_ids(self):
        op = AgentCrossModelPairMapper(group_key_mode="normalized_query", query_key="query")
        rows = [
            {
                "id": "a",
                "query": "  Fix  the\tBug  ",
                Fields.meta: {
                    "agent_lineage_sample_id": "A",
                    "agent_lineage_tag_model": "m1",
                    "agent_lineage_tag_quality": 0.9,
                },
            },
            {
                "id": "b",
                "query": "fix the bug",
                Fields.meta: {
                    "agent_lineage_sample_id": "B",
                    "agent_lineage_tag_model": "m2",
                    "agent_lineage_tag_quality": 0.6,
                },
            },
        ]
        from data_juicer.core.data.dj_dataset import NestedDataset

        ds = NestedDataset.from_list(rows)
        out = op.run(ds)
        lst = out.to_list()
        p0 = lst[0][Fields.meta][MetaKeys.agent_cross_model_pair]
        self.assertEqual(p0["match_basis"], "normalized_query")
        self.assertTrue(p0["has_pairwise_contrast"])
        self.assertEqual(p0["group_size"], 2)

    def test_simhash_lsh_groups_identical_blob(self):
        op = AgentCrossModelPairMapper(
            group_key_mode="simhash_lsh",
            query_key="query",
            simhash_max_hamming=0,
        )
        q = "same query text for both rows"
        rows = [
            {
                "id": "a",
                "query": q,
                Fields.meta: {
                    "agent_lineage_sample_id": "x1",
                    "agent_lineage_tag_model": "m1",
                    "agent_lineage_tag_quality": 1.0,
                },
            },
            {
                "id": "b",
                "query": q,
                Fields.meta: {
                    "agent_lineage_sample_id": "x2",
                    "agent_lineage_tag_model": "m2",
                    "agent_lineage_tag_quality": 0.5,
                },
            },
        ]
        from data_juicer.core.data.dj_dataset import NestedDataset

        ds = NestedDataset.from_list(rows)
        out = op.run(ds)
        lst = out.to_list()
        self.assertEqual(lst[0][Fields.meta][MetaKeys.agent_cross_model_pair]["match_basis"], "simhash_lsh")
        self.assertTrue(lst[0][Fields.meta][MetaKeys.agent_cross_model_pair]["has_pairwise_contrast"])


if __name__ == "__main__":
    unittest.main()
