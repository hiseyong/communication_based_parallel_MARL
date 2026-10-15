# Design Notes

Decisions the spec left ambiguous, made arbitrarily and recorded here per the
prompt's requirement (`claude_code_prompt_stage1-2.md` §8).

## Observation leak in `oracle_message`, and the 3-bug cascade fixing it caused

Spec 3.8 says the oracle requests help for "내 주 경로상 **시야 내** 첫 번째 닫힌
장벽" -- the first closed barrier *within sight*. The original implementation
checked only column order (`col < barrier_col`), with no visibility gate at
all, so it announced REQ_* for barriers the agent's actual egocentric
observation (V=7, so a 3-cell radius) could not possibly see yet. This was
caught empirically, not by re-reading the spec text: stage-1 Speaker training
plateaued at ~67-70% val accuracy (target was >=95%), and checking the
collected data directly showed **65% of all REQ-labeled samples had zero
barrier-channel activation anywhere in the observed grid** -- the Speaker
was being asked to predict something not present in its input. This is
exactly the "관측 누락" (observation gap) root-cause category §8 asks for.

Fixing it (`oracle_message(..., view_radius=V//2)`, threaded through every
`ScriptedAgent(...)` construction and the `own_send_mode="oracle"` path in
`train_listener.py`) broke the ScriptedAgent's >=99% clear-rate test, in two
separate ways, each requiring its own fix -- documented in sequence because
each was invisible until the previous one was fixed:

1. **`_advance` had no partial-pathfinding fallback.** It BFS'd straight to
   the exit and gave up entirely (STAY forever at START) if any barrier
   blocked the *whole* route -- fine when REQ fired from step 0 regardless of
   distance, fatal once REQ only fires near the barrier, because the agent
   never walked close enough to bring it into view. Fix: `bfs_path_towards`
   walks to the reachable position closest to the goal instead of giving up
   (`envs/oracle.py`).
2. **An agent could reach its own exit (and permanently freeze there, per
   spec 3.6) before a correctly-delayed request for its help ever arrived.**
   Previously the requester announced its need from step 0, so the helper
   had the whole episode to notice; view-gating can push that announcement
   to arrive after the on-paper-faster helper has already finished. Fix:
   `ScriptedAgent._still_possibly_needed` + `_crosses_own_barrier` hold the
   agent at its own exit *and* at its own (currently-open) barrier -- using
   privileged position state for patience only, never to trigger the help
   action itself, so the no-communication ablation test still correctly
   collapses success to ~0.
3. **A route's own device could sit behind that route's own DOOR-type
   barrier**, which re-closes once its presser leaves (unlike the one-shot
   OBSTACLE/LEVER). If the holder had already crossed its own barrier
   (opened earlier, independently, by the partner) before receiving a
   request for a device on the near side, the door closing behind it made
   that device permanently unreachable. Fix #2 above actually only *masks*
   this for the exit case; the barrier-crossing case needed both the
   generator constraint (`layout_generator.py`: each route's own device must
   be placed at a column *before* that route's own barrier) and
   `_crosses_own_barrier` to stop the agent from crossing its own barrier at
   all while an unconsumed device might still be needed behind it.

Net effect verified by `tests/test_env.py`'s clear-rate test (all three
levels now >=99%, L1 at 100%) and a new unit test
(`tests/test_oracle.py::test_oracle_does_not_leak_a_barrier_outside_view_radius`).
The full-scale Speaker/Listener training launched before this fix was killed
and restarted from scratch on the corrected oracle.

## Stage 2 diagnosis: curriculum promoted on a misleadingly easy metric (§8 "원인 진단" requirement)

Found via a 300-400k step diagnostic run before committing to the full-scale
background training (`runs/listener_exp1` configs reflect the fix):

- **Symptom**: with `curriculum_window=40` (an early default), the L1->L2
  promotion fired at step ~22k, but a periodic *greedy* (argmax) eval showed
  true L1 competence was only ~37% at that point, and L2 greedy success
  stayed at **exactly 0** for the next 265k steps (complete stagnation).
- **Root cause**: the curriculum's rolling success rate is measured over PPO's
  *stochastic* rollouts (`dist.sample()`), which explore into the correct
  "go help partner" behavior often enough to rack up reward and cross the
  0.8 threshold, while the policy's *mode* (what greedy/deployed inference
  actually does) hasn't shifted there yet. L1 has an easy ~50% floor for free
  (whichever route has no barrier just walks to its own exit and happens to
  pass by construction), which further inflates the metric independent of any
  learned cooperation.
- **Evidence it's "slow learning," not a dead end**: re-run with
  `curriculum_window=200` kept the greedy success rate pinned at a frozen
  0.3667 for ~320k steps while the *stochastic* success rate kept climbing
  (0.1 -> 0.5+ by step 363k) — i.e. probability mass is visibly shifting
  toward the correct behavior, it just hadn't overtaken the "selfish, walk to
  my own exit" mode yet. This is the expected shape of a sparse/delayed
  cooperative credit-assignment problem (the helpful action — walking to a
  device and pressing it — earns no reward itself; only the partner's later
  barrier-pass, +1, and eventual team completion, +10, pay for it), not a
  code bug.
- **Fix applied**: raised the default curriculum window (`configs/listener.yaml`)
  so promotion requires a sustained rate over many more episodes, and the
  full-scale runs use a timestep budget an order of magnitude larger than
  this diagnostic. Reward *weights* were deliberately left at the spec's
  defaults (not reshaped) since the trend showed real, if slow, progress —
  reshaping was the fallback only if the full-scale run also plateaus.
- **How to apply**: when reading `greedy_eval.csv` vs `log.csv`'s
  `success_rate` column for any checkpoint, remember they measure different
  things (deployed/greedy vs. exploratory/stochastic) and are expected to
  diverge, especially early in training on L2/L3.

## DONE timing
`DONE` is a pure function of the requester's current column relative to its
own barrier's column (`oracle.py: oracle_message`): emitted while
`barrier_col <= own_col <= barrier_col + done_hold` (default `done_hold=2`).
No episode history/timer is tracked — "just passed" is approximated as
"within `done_hold` columns past it," which is equivalent as long as the
agent moves roughly 1 cell/step and doesn't loiter there. This keeps the
oracle a stateless function of `(route, pos)`, reusable identically by
`collect_speaker_data.py` later.

## Barrier open-state observability
A route's own `DOOR_c`/`OBSTACLE_c` open/closed state is derived from the
*other* route's live position/toggle-state (`is_passable` in `dynamics.py`),
never from the other route's static cell types. `DualRouteEnv._build_obs`
only reads the other route's device *positions* (`plate_pos_by_color`) and
*toggle set*, so an agent's grid observation is a function of
`(own route layout, other agent's position, other agent's toggles)` and
never leaks the other route's geometry — verified by
`test_env.py::test_observation_does_not_leak_partner_route_structure`
(same route_a + two different route_b's -> byte-identical agent_0 obs).

## L3 ("상호 의존 A→B→A") construction
Built as a single forced chain rather than a literal cycle (an actual cycle
would be unsolvable): route `first` holds an *unconditioned* `device1`
(reachable with no precondition) that opens route `second`'s `barrier1`.
`second`'s `barrier1` gates access to `device2`, which opens route `first`'s
`barrier2`. Column ordering guarantees the dependency is acyclic:
`col(device1) < col(barrier2)` in `first`, and `col(barrier1) < col(device2)`
in `second`. Which route is `first` vs `second` is randomized per layout.

## Reward: solo_exit vs team_exit
Both are additive, not mutually exclusive: if both agents happen to reach
their exits on the same step, that step awards `solo_exit` for each plus
`team_exit` once. `barrier_pass` is credited the first time an agent's
position becomes a barrier cell it owns (requires the move to have actually
succeeded, i.e. the barrier was open), tracked via a per-episode
`(route_label, color)` set so it fires at most once per barrier.

## Egocentric grid padding
Cells outside the route's grid (near START/EXIT at the corridor ends) are
encoded as `WALL` in the egocentric crop — there is no separate "out of
bounds" channel.

## Scripted agent request/response priority
`ScriptedAgent` prioritizes fulfilling an incoming `REQ_*` over advancing
toward its own exit, but falls back to advancing if the requested device is
currently unreachable (e.g. gated behind its own still-closed barrier in the
L3 chain) instead of idling — this is what makes the forced ordering resolve
itself without deadlock. A `LEVER` response is one-shot: once toggled to the
requested state, the agent stops re-interacting with it (an earlier bug
re-toggled it every step it remained in `REQ` state, flip-flopping the
obstacle open/closed forever — fixed before the L2 success-rate test passed).

## Split seeding
`train`/`val`/`test_layout`/`test_combo` use disjoint seed offset ranges
(`LayoutGenerator.SEED_OFFSETS`) so they never emit the same realized map by
construction. `test_layout` still avoids `excluded_combos` (it isolates
"unseen map arrangement," not combo generalization); only `test_combo` forces
one of the excluded `(barrier_kind, color)` pairs into the layout.
