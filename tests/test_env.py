import copy

import numpy as np
import pytest

from coop_comm.envs.dual_route_env import AGENT_OF, DualRouteEnv
from coop_comm.envs.layout_generator import LayoutGenerator
from coop_comm.envs.objects import Layout
from coop_comm.envs.oracle import ScriptedAgent

LEVELS = ["L1", "L2", "L3"]
N_EPISODES = 200


def run_episode(env, force_none=False, max_steps=200):
    obs, infos = env.reset()
    view_radius = env.V // 2
    agents = {
        "agent_0": ScriptedAgent("a", env._layout, epsilon=0.0, rng=np.random.default_rng(1), view_radius=view_radius),
        "agent_1": ScriptedAgent("b", env._layout, epsilon=0.0, rng=np.random.default_rng(2), view_radius=view_radius),
    }
    for t in range(max_steps):
        actions = {}
        for label in ("a", "b"):
            agent = AGENT_OF[label]
            other = "b" if label == "a" else "a"
            act, msg = agents[agent].act(
                env._pos[label], env._toggles[label], env._pos[other], env._toggles[other], env._msg_in[label], env.vocab
            )
            if force_none:
                msg = env.vocab.none_idx
            actions[agent] = np.array([int(act), msg])
        obs, rewards, terms, truncs, infos = env.step(actions)
        if all(terms.values()) or all(truncs.values()):
            return all(terms.values()), t + 1
    return False, max_steps


@pytest.mark.parametrize("level", LEVELS)
def test_scripted_agent_pair_clears_at_least_99_percent(level):
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    env = DualRouteEnv(layout_generator=gen, level=level, split="train", max_steps=200)
    successes = 0
    for i in range(N_EPISODES):
        env.reset(options={"layout_index": i})
        ok, _ = run_episode(env)
        successes += ok
    rate = successes / N_EPISODES
    assert rate >= 0.99, f"{level}: scripted-agent clear rate {rate:.3f} < 0.99"


@pytest.mark.parametrize("level", LEVELS)
def test_cutting_communication_tanks_success_rate(level):
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    env = DualRouteEnv(layout_generator=gen, level=level, split="train", max_steps=200)
    successes = 0
    for i in range(100):
        env.reset(options={"layout_index": i})
        ok, _ = run_episode(env, force_none=True)
        successes += ok
    rate = successes / 100
    assert rate <= 0.1, f"{level}: success rate with msg forced to NONE is {rate:.3f}, env doesn't require comms"


def test_observation_does_not_leak_partner_route_structure():
    """agent_0's grid observation must be a function of (route_a layout, agent_b's
    position, agent_b's toggles) only -- never of route_b's static cell types."""
    gen = LayoutGenerator(K=3, excluded_combos=[("OBSTACLE", 2)])
    layout1 = gen.sample("train", "L2", 0)
    layout2 = gen.sample("train", "L2", 1)
    assert layout1.route_b.grid_type.tolist() != layout2.route_b.grid_type.tolist(), "test needs differing route_b"

    mixed_1 = Layout(
        H=layout1.H, W=layout1.W, colors=layout1.colors, route_a=layout1.route_a, route_b=layout1.route_b,
        level=layout1.level, seed=layout1.seed,
    )
    mixed_2 = Layout(
        H=layout1.H, W=layout1.W, colors=layout1.colors, route_a=layout1.route_a, route_b=layout2.route_b,
        level=layout1.level, seed=layout1.seed,
    )

    def get_obs_a(layout):
        env = DualRouteEnv(fixed_layout=layout, level=layout.level, max_steps=200)
        obs, _ = env.reset()
        # advance agent_0 a couple of steps while agent_1 stays put, so pos != start
        for _ in range(2):
            actions = {
                "agent_0": np.array([3, env.vocab.none_idx]),  # RIGHT
                "agent_1": np.array([4, env.vocab.none_idx]),  # STAY
            }
            obs, *_ = env.step(actions)
        return obs["agent_0"]["grid"]

    grid1 = get_obs_a(mixed_1)
    grid2 = get_obs_a(mixed_2)
    np.testing.assert_array_equal(grid1, grid2)


def test_fixed_layout_reset_is_deterministic():
    gen = LayoutGenerator(K=3)
    layout = gen.sample("train", "L1", 0)
    env = DualRouteEnv(fixed_layout=layout, level="L1", max_steps=200)
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    np.testing.assert_array_equal(obs1["agent_0"]["grid"], obs2["agent_0"]["grid"])
