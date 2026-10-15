# coop_comm

Cooperative MARL with structured communication: two agents, each confined to
their own independent route, must send/interpret a small fixed vocabulary of
messages to ask each other to operate devices that control barriers on the
*other* agent's path. No LLMs; PettingZoo `ParallelEnv` + a CleanRL-style
single-file PPO. See `claude_code_prompt_stage1-2.md` for the full spec and
`DESIGN_NOTES.md` for every ambiguous decision made along the way (including
two real bugs found via diagnostic training runs).

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

(Needs Python >= 3.10, PettingZoo's `ParallelEnv` API.)

## Stage 1: environment

```bash
pytest tests/ -q
```
Covers: 1000 generated layouts per level all solvable (joint-state BFS),
train/val/test_layout/test_combo split seed-disjointness and combo exclusion,
ScriptedAgent-pair clear rate on L1-L3 (>=99%), communication-cutoff success
collapse, and no partner-route leakage in observations.

## Stage 2: Speaker (message classification)

```bash
# 1. roll out ScriptedAgent(epsilon) pairs -> (obs, oracle label) data
python -m coop_comm.train.collect_speaker_data --config coop_comm/configs/speaker_data.yaml --split train
python -m coop_comm.train.collect_speaker_data --config coop_comm/configs/speaker_data.yaml --split val --episodes_per_level 400
python -m coop_comm.train.collect_speaker_data --config coop_comm/configs/speaker_data.yaml --split test_layout --episodes_per_level 400
python -m coop_comm.train.collect_speaker_data --config coop_comm/configs/speaker_data.yaml --split test_combo --episodes_per_level 400

# 2. train (class-weighted cross-entropy)
python -m coop_comm.train.train_speaker --config coop_comm/configs/speaker.yaml

# 3. evaluate: offline (accuracy/F1/confusion matrix) + online (speaker swapped
#    in for a ScriptedAgent's outgoing message in a live episode)
python -m coop_comm.eval.eval_speaker --config coop_comm/configs/eval_speaker.yaml
```
**Result** (438k train samples, 30 epochs): offline accuracy 100% on val and
test_layout (target was >=95%), 78% on test_combo; online task success 100%
on val/test_layout, **0%** on test_combo. The Speaker generalizes perfectly
to unseen map *arrangements* and not at all to the unseen (barrier, color)
*combination* -- it never saw `OBSTACLE_BLUE` in training and simply doesn't
know what to call it.

## Stage 3 (of this spec's 2-stage scope): Listener (PPO + curriculum)

```bash
# single seed
python -m coop_comm.train.train_listener --config coop_comm/configs/listener.yaml --seed 0

# 3 seeds + aggregated mean+-std table/figure
./scripts/run_seeds.sh   # SEEDS="0 1 2" EXP_DIR=runs/listener_exp1

python -m coop_comm.eval.eval_listener --config coop_comm/configs/eval_listener.yaml
```
The partner is always a fixed ScriptedAgent (oracle messages, epsilon=0); the
listener only learns the *move* action (its own outgoing message is forced to
`NONE` or its own oracle label via `own_send_mode`, per config). Curriculum
advances L1->L2->L3 once a rolling success rate crosses `curriculum_threshold`
over `curriculum_window` episodes. `runs/<exp>/<seed>/greedy_eval.csv` logs a
*deterministic* (argmax) success rate at periodic checkpoints — read this,
not just `log.csv`'s stochastic rollout rate, to judge real competence (see
DESIGN_NOTES.md: these two numbers can diverge a lot early in training).

## Beyond the original 2-stage scope: does the learned language actually work?

The spec explicitly scoped stage 3 (combining Speaker + Listener into one
agent and training end-to-end) out, asking only for an `eval_pair.py` stub.
Per a follow-up request, that stub is filled in as a **composition-only**
check (no joint fine-tuning): plug the trained Speaker into the message head
and the trained Listener into the move head of *both* agents (one shared set
of weights, used symmetrically — the collected data and PPO partner are
already symmetric across routes), and run pure self-play with no
ScriptedAgent on either side:

```bash
python -m coop_comm.eval.eval_pair --config coop_comm/configs/eval_pair.yaml
  # speaker_checkpoint / listener_checkpoint pick which seeds to pair
```

It reports self-play success rate per split/level against the
ScriptedAgent/ScriptedAgent ceiling — the gap between them is the direct
answer to "can the separately-learned send/listen skills solve the real
task," including on `test_layout`/`test_combo` (map and combo
generalization).

**Result** (`runs/eval_pair/results.json`, speaker seed 0 x listener seed 1,
300 episodes/cell): self-play clears the task well above chance but well
below the oracle ceiling, and composition error compounds across the two
learned modules (both now have to get it right simultaneously, vs.
`eval_listener`'s one-learned-side setup):

| split | L1 | L2 | L3 |
|---|---|---|---|
| val | 0.55 | 0.32 | 0.32 |
| test_layout | 0.62 | 0.30 | 0.35 |
| test_combo | 0.00 | 0.00 | 0.00 |
| scripted ceiling | 1.00 | 1.00 | 1.00 |

So: **composing two separately-trained modules with no joint fine-tuning
partially works** (31-62% on in-distribution/novel-layout splits, far above
the ~0% a broken pipeline would show) but is nowhere near the oracle ceiling,
and **completely fails to transfer to the unseen combo** — compounding the
Speaker's test_combo failure (0% task success on its own, see above) with
whatever the Listener's own test_combo gap (13-24% in `eval_listener`,
against a ScriptedAgent partner that still sends correctly) adds on top. This
is the direct, empirical answer to "can the real task be solved": partially,
on combos seen during training; not at all on unseen ones -- exactly the
gap joint fine-tuning (the explicitly-out-of-scope stage 3) exists to close.

## Config

Every entry point is `python -m coop_comm.<module> --config <yaml> [--key value ...]`
(`coop_comm/config_utils.py`): a YAML file sets defaults, any `--key value` on
the command line overrides it. All configs live in `coop_comm/configs/`.

## Reproducing results

Results land in `runs/<exp_name>/<seed>/`: `log.csv` (PPO training curve),
`greedy_eval.csv` (periodic deterministic eval), `listener.pt` /
`ckpts/update_*.pt` (checkpoints), `eval/results.json` + `.png` plots. Rerun
with the same `--seed` and the same `excluded_combos`/level/split config for
exact reproducibility (everything is seeded: `numpy.random.default_rng`,
`torch.manual_seed`).

## Final full-scale results (3 seeds, 3M PPO steps each; see DESIGN_NOTES.md for the diagnosis)

Learning the *reactive* half of listening (receive `REQ_*` -> navigate to the
matching device) is a sparse/delayed-credit-assignment problem — the helpful
action itself earns no reward, only the partner's later barrier-pass (+1) and
eventual team completion (+10) do. In practice this did resolve with enough
budget: all 3 seeds cleared the curriculum's 0.8 threshold on L1, 2/3 reached
L3 (seed 2 stalled on L2). Greedy (deployed) success rate, mean +- std over
seeds, listener vs. a fixed ScriptedAgent partner (`eval_listener.py`):

| split | L1 | L2 | L3 |
|---|---|---|---|
| val | 0.71 +- 0.05 | 0.56 +- 0.09 | 0.49 +- 0.14 |
| test_layout | 0.75 +- 0.07 | 0.54 +- 0.08 | 0.44 +- 0.11 |
| test_combo | 0.53 +- 0.01 | 0.18 +- 0.02 | 0.23 +- 0.05 |

`response_accuracy` (0.68-0.90 in-distribution) and the `msg_in` ablation
(zero/shuffle both tank success -- e.g. val/L1 0.71 -> 0.06 zeroed) confirm
the listener is genuinely using the channel, not succeeding despite it.
`test_combo` is consistently the weak point (matches the Speaker's own
combo-generalization gap below) -- L1 survives mostly because half of L1
episodes don't need the excluded combo at all; L2/L3 collapse much harder.
