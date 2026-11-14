"""Original manuscript field and sign reward; no model/simulator imports.

This implements the paper definitions, NOT the recovered prototype's hops+1,
nonuniform thresholds, or spatially weighted room similarity. Engineering
checks of this module are not a reproduction of any reported navigation score.
"""
from dataclasses import dataclass
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), name + ' must be numeric')
    require(math.isfinite(value), name + ' must be finite')
    return float(value)


@dataclass(frozen=True)
class PaperParameters:
    d_min: float
    delta_vis: float
    delta_novis: float
    alpha: tuple
    beta: tuple
    k_vis: int
    beta_tail_step: float
    tau: float
    w_hi: float
    w_lo: float
    r_up: float
    r_down: float

    @classmethod
    def from_config(cls, config):
        p = config['vlm_sp_rew_params']
        require(config.get('method_contract') == 'original_manuscript_r5', 'An explicit paper-contract configuration is required')
        scalars = {k: number(p[k], k) for k in ['d_min', 'delta_vis', 'delta_novis', 'tau', 'w_hi', 'w_lo', 'r_up', 'r_down', 'beta_tail_step']}
        require(all(scalars[k] > 0 for k in ['d_min', 'delta_vis', 'delta_novis']), 'Positive distance widths required; delta_vis is a scalar, not a threshold list')
        require(type(p['K_vis']) is int and p['K_vis'] > 0, 'Explicit positive K_vis required')
        alpha = tuple(number(v, 'alpha_k') for v in p['alpha_k'])
        beta = tuple(number(v, 'beta_k') for v in p['beta_k'])
        require(len(alpha) >= p['K_vis'] and beta, 'Value sequences must cover the specified bands')
        require(alpha[0] > beta[0], 'Original visible/nonvisible ordering required')
        require(scalars['r_up'] > 0 and scalars['r_down'] < 0, 'Original up/down signs required')
        require(scalars['w_hi'] >= 0 and scalars['w_lo'] >= 0, 'Nonnegative regional weights required')
        # Table 4 uses arithmetic beta tails; retain finite prefixes exactly and
        # require the continuation convention instead of inventing later values.
        require(all(abs((b - a) - scalars['beta_tail_step']) < 1e-12 for a, b in zip(beta, beta[1:])), 'Declared beta continuation differs from the finite sequence')
        return cls(alpha=alpha, beta=beta, k_vis=p['K_vis'], **scalars)


def value_from_hops(visible, hops, p):
    """Original Eq(value); exclude the virtual goal from sampled field nodes."""
    require(type(visible) is bool, 'Goal visibility must be a boolean annotation')
    hops = number(hops, 'graph hop distance')
    require(hops > 0 and hops.is_integer(), 'A sampled node needs positive unweighted hops; do not silently add one or invent index zero')
    if visible:
        index = min(p.k_vis, math.ceil(hops / p.delta_vis))
        return p.alpha[index - 1]
    index = math.ceil(hops / p.delta_novis)
    if index <= len(p.beta):
        return p.beta[index - 1]
    return p.beta[-1] + (index - len(p.beta)) * p.beta_tail_step


def hops_from_paths(paths, goal_id):
    distances = {}
    for node, path in paths.items():
        if path is None:
            continue
        require(path and path[0] == node and path[-1] == goal_id, 'Path endpoints mismatch')
        require(len(path) == len(set(path)), 'A shortest unweighted path cannot contain a cycle')
        distances[node] = len(path) - 1
    require(distances.get(goal_id) == 0, 'Virtual goal must have zero hops')
    return distances


def horizontal_distance(a, b):
    return math.hypot(number(a['x'], 'x') - number(b['x'], 'x'), number(a['z'], 'z') - number(b['z'], 'z'))


def repair_unreachable_hops(nodes, distances, goal_id, room_labels):
    """Original nearest finite same-room point +1; never invent graph edges.

    Only originally reachable sampled points are donors. Repairs are therefore
    independent of iteration order and cannot recursively create fake routes.
    A copied scalar is still not proof of a traversable connection.
    """
    by_id = {n['id']: n for n in nodes}
    require(len(by_id) == len(nodes) and goal_id in by_id, 'Unique node identities required')
    require(set(distances) <= set(by_id), 'Distances contain unknown nodes')
    finite = sorted(i for i in distances if i != goal_id)
    require(finite, 'No sampled point reaches the goal; retain preprocessing failure')
    for i in by_id:
        if i != goal_id:
            require(i in room_labels, 'Every sampled point needs an archived refined room label')
    result, repairs = dict(distances), []
    for i in sorted(set(by_id) - set(distances) - {goal_id}):
        candidates = [j for j in finite if room_labels[j] == room_labels[i]] or finite
        donor = min(candidates, key=lambda j: (horizontal_distance(by_id[i]['position'], by_id[j]['position']), j))
        result[i] = distances[donor] + 1
        repairs.append(dict(node_id=i, donor_node_id=donor, donor_originally_reachable=True, inherited_hops=result[i], added_one=True, graph_edges_not_created=True))
    return result, repairs


def select_sample(points, position, room_label, d_min):
    """Original Eq(index), with thresholds scaled by d_min.

    The caller must record where its training-side room label comes from; this
    function neither reads simulator room ground truth nor exposes it to actors.
    """
    d_min = number(d_min, 'd_min')
    require(points and d_min > 0 and isinstance(room_label, str), 'Nonempty field, positive spacing and explicit room label required')
    ranked = sorted(enumerate(points), key=lambda item: (horizontal_distance(item[1]['position'], position), item[0]))
    near = [point for _, point in ranked if horizontal_distance(point['position'], position) <= .2 * d_min]
    if near:
        return near[0]
    same = [point for _, point in ranked if point.get('room_label') == room_label and horizontal_distance(point['position'], position) <= 1.5 * d_min]
    return same[0] if same else ranked[0][1]


def sign_reward(previous_value, next_value, p, coefficient=1):
    """Original Eq(reward); coefficient belongs only to the requested ablation."""
    a, b = number(previous_value, 'previous field value'), number(next_value, 'next field value')
    coefficient = number(coefficient, 'outer coefficient')
    require(coefficient >= 0, 'Use the declared nonnegative ablation coefficient')
    weight = p.w_hi if max(a, b) >= p.tau else p.w_lo
    direction = p.r_up if b > a else p.r_down if b < a else 0.
    return coefficient * weight * direction
