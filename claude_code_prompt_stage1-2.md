# Claude Code 프롬프트 — 협력 통신 MARL: 환경 구축 + 언어 체계(송신/이행) 학습

> 아래 전체를 Claude Code에 붙여 넣어 사용. 빈 디렉터리에서 시작하는 것을 가정.

---

## 0. 프로젝트 개요 (Context)

나는 ICEIC 2027 투고를 목표로 **"멀티 에이전트 강화학습에서 구조화된 통신을 통한 협력적 문제 해결"** 연구를 하고 있다. LLM은 사용하지 않는다.

연구는 3단계로 진행된다. **이번 작업 범위는 1·2단계와 이를 위한 환경 구축까지**다. 3단계는 구현하지 말고, 나중에 붙일 수 있도록 인터페이스만 열어 둬라.

1. **송신 학습 (Speaker)**: 에이전트가 자기 경로에서 막힌 장애물을 보면, 상대에게 필요한 도움을 정해진 어휘로 송신하도록 학습한다. (예: 빨간 문이 보임 → `REQ_PLATE_RED`)
2. **이행 학습 (Listener)**: 신호를 수신하면 자기 경로에서 해당 장치를 작동하도록 학습한다. (예: `REQ_PLATE_RED` 수신 → 내 경로의 빨간 발판으로 이동해 밟고 유지 → `DONE` 수신 시 이탈)
3. (향후) 학습된 송신·수신 모듈을 결합해 실제 과제를 해결.

## 1. 기술 스택 / 제약

- Python 3.10+, PyTorch, NumPy, Gymnasium spaces, **PettingZoo `ParallelEnv`** API
- CPU 노트북에서 각 학습이 **1~2시간 내**에 수렴 가능한 규모 (GPU 있으면 자동 사용)
- PPO는 CleanRL 스타일의 **단일 파일 구현** (외부 RL 프레임워크 의존 X — 통신 모듈 수정이 쉬워야 함)
- 설정은 YAML + argparse override, 로깅은 CSV + TensorBoard, 그림은 matplotlib
- 모든 실행은 `--seed`로 재현 가능해야 함. 결과는 `runs/<exp_name>/<seed>/`에 저장
- 테스트는 pytest

## 2. 디렉터리 구조

```
coop_comm/
  envs/
    dual_route_env.py      # PettingZoo ParallelEnv
    layout_generator.py    # 절차적 맵 생성 + 해결 가능성 검증
    oracle.py              # 정답 메시지 / 스크립트 정책
    render.py              # ASCII + matplotlib 프레임 → GIF
  comm/
    vocab.py               # 어휘 정의, 인코딩/디코딩
    channel.py             # 메시지 채널 (identity / 잡음 BSC형 symbol flip / drop)
  models/
    encoders.py            # 관측 인코더 (CNN for grid + MLP)
    speaker.py             # obs -> message logits
    listener.py            # obs + message -> action logits, value
  train/
    collect_speaker_data.py
    train_speaker.py       # 1단계 (지도학습)
    train_listener.py      # 2단계 (PPO)
    ppo.py
  eval/
    eval_speaker.py
    eval_listener.py
    plots.py
  configs/*.yaml
  tests/
  README.md
```

## 3. 환경 명세: `DualRouteEnv`

### 3.1 구조
- 에이전트 2명: `agent_0`(Route A), `agent_1`(Route B). **각자 자기 경로만 존재·관측**하며 서로의 경로로 이동할 수 없다.
- 각 Route는 독립 그리드(기본 7×15, config로 변경). 시작 S에서 출구 E까지 가는 주 경로가 있고, 옆으로 장치가 놓인 곁가지(alcove)가 있다.
- 오브젝트 (K개 색, 기본 K=3: RED/GREEN/BLUE):
  - `WALL`, `FLOOR`, `START`, `EXIT`
  - `DOOR_c`: 통과 불가. **상대 경로의 `PLATE_c`가 눌려 있는 동안에만** 열림 (유지형)
  - `OBSTACLE_c`: 통과 불가. **상대 경로의 `LEVER_c`가 토글되면** 열림/닫힘 (토글형)
  - `PLATE_c`: 에이전트가 위에 서 있는 동안 활성
  - `LEVER_c`: `INTERACT` 행동으로 토글
- **교차 결합**: 장치는 항상 상대 경로의 같은 색 장벽을 제어한다. 매핑 규칙(문↔발판, 장애물↔레버, 같은 색)은 고정이며 두 에이전트 모두 알고 있다고 가정.
- 한 장벽이 열렸는지 여부는 그 장벽이 있는 경로의 에이전트만 관측 가능 (상대 장치 상태는 관측 불가).

### 3.2 행동
`Discrete(6)`: `UP, DOWN, LEFT, RIGHT, STAY, INTERACT`

### 3.3 관측 (agent별 Dict space)
- `grid`: 자기 위치 중심 egocentric `V×V`(기본 V=7) 부분 관측, 채널별 one-hot (`C × V × V`). 채널: wall, floor, exit, door_c(K), obstacle_c(K), plate_c(K), lever_c(K), barrier_open 플래그, self
- `pos`: 정규화된 자기 좌표 (2,)
- `msg_in`: 직전 스텝에 상대로부터 받은 메시지 one-hot (|V_msg|,)
- `own_status`: 현재 내가 밟고 있는 발판/마지막으로 토글한 레버 one-hot (작게)
- 옵션 `--full_obs_partner` (oracle용): 상대 경로 전체 관측을 추가

### 3.4 통신
- 매 스텝 각 에이전트는 행동과 별도로 **메시지 토큰 1개**를 송신 (`step(actions, messages)` 형태가 PettingZoo와 안 맞으면 action을 `Dict{"move", "msg"}` 또는 MultiDiscrete로 구성할 것)
- 메시지는 `comm/channel.py`를 거쳐 다음 스텝 상대의 `msg_in`으로 전달 (1-step delay)
- 채널 옵션: `identity`, `symbol_flip(p)`, `drop(p)` — 나중에 잡음 실험에 사용

### 3.5 어휘 (`comm/vocab.py`)
```
NONE
REQ_PLATE_<c>   for c in colors   # "상대 경로의 c 발판을 밟아 달라" (내 앞에 DOOR_c)
REQ_LEVER_<c>   for c in colors   # "c 레버를 토글해 달라" (내 앞에 OBSTACLE_c)
DONE                              # "통과했다, 이제 놓아도 된다"
WAIT                              # 예약 (지금은 사용 안 해도 됨)
```
- 크기 = 2K + 3. 색 수 K가 바뀌어도 자동 생성되게.
- 추후 슬롯형(`[intent][device][color]`) 표현으로 바꿀 수 있도록 `Vocab` 클래스가 flat ↔ slot 변환을 지원하게 설계.

### 3.6 보상 (팀 공동 보상)
- 두 에이전트 모두 출구 도달: +10 (에피소드 종료)
- 한 에이전트 출구 도달: +2 (해당 에이전트는 이후 STAY 고정)
- 장벽 최초 통과: +1
- 스텝 패널티: −0.01, 최대 스텝 기본 200
- 보상 항목은 config로 on/off 및 가중치 조정 가능하게

### 3.7 레이아웃 생성기 (`layout_generator.py`)
- 각 Route에 장벽 0~N개(기본 1~2)를 주 경로 위에 배치하고, 대응 장치를 **상대 경로**의 곁가지에 배치
- **해결 가능성 검증 필수**: 두 경로의 joint state(위치, 장치 상태)로 BFS 하거나, 의존성 그래프에 순환 교착이 없음을 확인 후 스크립트 정책으로 실제 클리어되는지 시뮬레이션
- 시드 기반 생성. `train/val/test` 분할 지원:
  - `test_layout`: 학습에 없던 맵 배치
  - `test_combo`: 학습에서 제외한 (장벽 타입, 색) 조합 — 예: 학습 때 `OBSTACLE_BLUE`는 등장시키지 않음 (조합 일반화 평가용)
- 난이도 레벨: L1(한쪽만 장벽 1개), L2(양쪽 각 1개), L3(상호 의존: A→B→A 순서 필요)

### 3.8 Oracle (`oracle.py`)
- `oracle_message(state, agent)`: 정답 메시지 라벨
  - 내 주 경로상 **시야 내 첫 번째 닫힌 장벽**이 있으면 그에 대응하는 `REQ_*`
  - 상대가 내 요청으로 발판을 밟고 있고, 내가 방금 그 문을 통과했으면 `DONE` (통과 후 1~2스텝 유지)
  - 그 외 `NONE`
  - 규칙은 한 곳에 모아두고 docstring에 명확히 기술
- `ScriptedAgent`: BFS 기반으로 자기 경로를 진행하고, oracle 메시지를 송신하며, 수신 메시지에 정답 행동으로 반응. 파라미터 `epsilon`으로 무작위 행동 섞기 (데이터 다양성용)

### 3.9 렌더링
- ASCII 렌더 (두 경로 나란히 + 각 에이전트 송신 메시지 표시)
- matplotlib 프레임 → GIF 저장 (`eval` 시 옵션)

## 4. 1단계: 송신 학습 (Speaker)

### 데이터 수집 (`collect_speaker_data.py`)
- `ScriptedAgent(epsilon ∈ [0, 0.3])` 두 명으로 다수 에피소드 롤아웃, 각 스텝의 (agent obs, oracle 메시지 라벨) 저장 → `.npz`
- 클래스 불균형(NONE 다수) 리포트, 학습 시 class weight 또는 oversampling 옵션

### 모델 / 학습 (`speaker.py`, `train_speaker.py`)
- 인코더: 작은 CNN(grid) + MLP(pos, own_status) → 메시지 logits
- Cross-entropy (class-weighted). **인코더는 2·3단계에서 재사용 가능한 모듈로** 분리
- 옵션 `--gumbel`: Gumbel-Softmax 출력 헤드 (3단계 end-to-end 미세조정 대비)

### 평가 (`eval_speaker.py`)
- `val / test_layout / test_combo` 각각에서: 전체 accuracy, 클래스별 precision/recall/F1, confusion matrix 그림
- **온라인 평가**: ScriptedAgent의 메시지를 학습된 speaker로 대체해 실제 에피소드에서 (a) 메시지 정확도, (b) 수신 측이 ScriptedAgent일 때 과제 성공률 측정

## 5. 2단계: 이행 학습 (Listener)

### 설정 (`train_listener.py`)
- 상대 에이전트 = **ScriptedAgent (oracle 메시지 송신, ε=0)** 고정
- 학습 대상 에이전트 = Listener 정책: 입력 `grid, pos, own_status, msg_in` → `move` 행동 (송신은 이 단계에선 NONE 고정 또는 oracle로 대체 — config로 선택)
- 알고리즘: PPO (GRU 옵션 포함 — 발판 유지/DONE 대기에 메모리가 필요할 수 있음)
- 커리큘럼: L1 → L2 → L3 (성공률 임계값 도달 시 자동 진급, 기본 0.8)
- 강건성 옵션: 학습 중 채널 `symbol_flip(p)` 적용

### 평가 (`eval_listener.py`)
- 과제 성공률, 평균 스텝
- **응답 정확도**: `REQ_x` 수신 후 T스텝(기본 30) 내 올바른 장치 작동 비율
- **유지/해제 정확도**: 발판 요청 시 `DONE` 전 이탈 비율(낮을수록 좋음), `DONE` 후 이탈까지 걸린 스텝
- **오작동률**: 요청 없이(또는 다른 색 요청에) 장치를 작동한 비율
- **Ablation**: `msg_in`을 0으로/무작위 셔플 → 성능 하락폭 (메시지를 실제로 쓰는지 검증)
- `test_layout`, `test_combo`에서 일반화 평가

## 6. 3단계 대비 인터페이스 (구현 X)
- `Speaker`와 `Listener`가 같은 `ObsEncoder`를 공유/교체할 수 있는 구조
- 한 에이전트 = `CommAgent(encoder, speaker_head, policy_head)`로 조립 가능하게
- 체크포인트에서 speaker/listener를 각각 로드해 두 학습 에이전트끼리 붙이는 `eval_pair.py` 자리만 만들어 둘 것 (TODO)

## 7. 작업 순서와 체크포인트 — 각 단계 끝나면 멈추고 결과 보고

1. **환경 + 생성기 + oracle + ScriptedAgent + 렌더**
   - pytest: 생성된 맵 1000개 모두 해결 가능 / ScriptedAgent 2명이 L1~L3에서 ≥99% 클리어 / 통신 끊으면(`msg=NONE`) ScriptedAgent 성공률이 크게 떨어짐(환경이 실제로 통신을 요구하는지 확인) / 관측에 상대 경로 정보가 새지 않음
   - L3 예시 에피소드 GIF 1개 저장
   - 👉 여기서 멈추고 맵 예시(ASCII)와 테스트 결과를 보여줘
2. **1단계 Speaker**: 데이터 수집 → 학습 → 오프라인/온라인 평가. 목표 val accuracy ≥ 95%
   - 👉 멈추고 클래스별 F1, confusion matrix, test_combo 결과 보고
3. **2단계 Listener**: PPO + 커리큘럼 → 평가 + ablation
   - 👉 멈추고 학습 곡선, 응답 정확도, ablation 결과 보고
4. README: 설치, 각 단계 실행 명령, 주요 config 설명, 결과 재현 방법

## 8. 기타 요구사항
- 코드에 타입 힌트와 간결한 docstring. 매직 넘버 금지(전부 config)
- 시드 3개 이상으로 돌리는 스크립트 `scripts/run_seeds.sh`와 평균±표준편차 표/그림 생성
- 설계상 애매한 결정(예: DONE 타이밍 규칙, 장벽 열림 상태 관측 범위)은 임의로 정하되 `DESIGN_NOTES.md`에 근거와 함께 기록
- 학습이 수렴하지 않으면 하이퍼파라미터를 무작정 키우지 말고, 원인 진단(보상 스케일, 관측 누락, 클래스 불균형 등)을 먼저 보고할 것
