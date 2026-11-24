"""Analytic unit tests for the paper field and reward definitions."""
import copy
import unittest
from vlm_sp_rew.field_contract import PaperParameters, value_from_hops, hops_from_paths, repair_unreachable_hops, select_sample, sign_reward


def config():
    return dict(method_contract='paper_v1', vlm_sp_rew_params=dict(d_min=1, delta_vis=2, delta_novis=5, alpha_k=[5, 4, 3], beta_k=[2, 1, 0, -1], K_vis=3, beta_tail_step=-1, tau=3, w_hi=1, w_lo=.5, r_up=1, r_down=-1))


class PaperContractTests(unittest.TestCase):
    def setUp(self):
        self.p = PaperParameters.from_config(config())

    def test_visible_boundaries_are_uniform_original_bands(self):
        self.assertEqual([value_from_hops(True, d, self.p) for d in [1, 2, 3, 4, 5, 6, 20]], [5, 5, 4, 4, 3, 3, 3])

    def test_hidden_boundaries_and_explicit_beta_tail(self):
        self.assertEqual([value_from_hops(False, d, self.p) for d in [1, 5, 6, 10, 11, 20, 21, 30]], [2, 2, 1, 1, 0, -1, -2, -3])

    def test_reject_legacy_threshold_list_in_paper_config(self):
        c = config(); c['vlm_sp_rew_params']['delta_vis'] = [2, 5]
        with self.assertRaises(ValueError): PaperParameters.from_config(c)

    def test_no_silent_zero_or_metric_to_hops_conversion(self):
        for d in [0, -1, 1.5, float('inf'), float('nan')]:
            with self.assertRaises(ValueError): value_from_hops(True, d, self.p)

    def test_beta_continuation_must_match_declared_prefix(self):
        c = config(); c['vlm_sp_rew_params']['beta_tail_step'] = -.5
        with self.assertRaises(ValueError): PaperParameters.from_config(c)

    def test_bfs_paths_use_hops_not_number_of_vertices(self):
        self.assertEqual(hops_from_paths({0: [0], 1: [1, 0], 2: [2, 1, 0], 3: None}, 0), {0: 0, 1: 1, 2: 2})

    def test_repairs_include_disconnected_components_and_keep_graph_unmodified(self):
        nodes = [dict(id=i, position=dict(x=x, z=0)) for i, x in [(0, 0), (1, 1), (2, 10), (3, 11), (4, 12)]]
        # Nodes3/4 could be connected to one another: infinite distance alone,
        # not degree zero, is the paper's repair condition.
        paths = {0: [0], 1: [1, 0], 2: [2, 1, 0], 3: None, 4: None}
        distances = hops_from_paths(paths, 0); labels = {1: 'A', 2: 'B', 3: 'A', 4: 'A'}
        before = copy.deepcopy((nodes, distances, labels))
        result, repairs = repair_unreachable_hops(nodes, distances, 0, labels)
        self.assertEqual(result, {0: 0, 1: 1, 2: 2, 3: 2, 4: 2})
        self.assertTrue(all(r['donor_node_id'] == 1 and r['added_one'] and r['graph_edges_not_created'] for r in repairs))
        self.assertEqual((nodes, distances, labels), before)

    def test_global_fallback_and_no_recursive_repair(self):
        nodes = [dict(id=i, position=dict(x=x, z=0)) for i, x in [(0, 0), (1, 1), (2, 5), (3, 5.1)]]
        result, repairs = repair_unreachable_hops(nodes, {0: 0, 1: 3}, 0, {1: 'A', 2: 'B', 3: 'B'})
        self.assertEqual([result[2], result[3]], [4, 4])
        self.assertEqual([r['donor_node_id'] for r in repairs], [1, 1])

    def test_no_reachable_sample_is_not_fabricated(self):
        with self.assertRaises(ValueError): repair_unreachable_hops([dict(id=0, position=dict(x=0, z=0)), dict(id=1, position=dict(x=1, z=0))], {0: 0}, 0, {1: 'A'})

    def test_near_radius_scales_with_spacing(self):
        points = [dict(id=0, position=dict(x=.35, z=0), room_label='B'), dict(id=1, position=dict(x=.5, z=0), room_label='A')]
        self.assertEqual(select_sample(points, dict(x=0, z=0), 'A', 2)['id'], 0)
        self.assertEqual(select_sample(points, dict(x=0, z=0), 'A', 1)['id'], 1)

    def test_same_room_radius_scales_and_has_global_fallback(self):
        points = [dict(id=0, position=dict(x=.6, z=0), room_label='B'), dict(id=1, position=dict(x=2, z=0), room_label='A')]
        self.assertEqual(select_sample(points, dict(x=0, z=0), 'A', 2)['id'], 1)
        self.assertEqual(select_sample(points, dict(x=0, z=0), 'A', 1)['id'], 0)

    def test_reward_threshold_crossing_and_equal_field(self):
        self.assertEqual([sign_reward(a, b, self.p) for a, b in [(0, 2), (2, 3), (3, 2), (2, 0), (3, 3)]], [.5, 1, -1, -.5, 0])

    def test_coefficient_zero_and_one_reuse_original_reward(self):
        self.assertEqual([sign_reward(2, 3, self.p, c) for c in [0, .5, 1, 2]], [0, .5, 1, 2])

    def test_regional_sign_rule_does_not_telescope(self):
        c = config(); c['vlm_sp_rew_params'].update(w_hi=1, w_lo=1)
        p = PaperParameters.from_config(c)
        self.assertEqual(sum(sign_reward(a, b, p) for a, b in [(0, 1), (1, 2), (2, 0)]), 1)


if __name__ == '__main__':
    unittest.main()
