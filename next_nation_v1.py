from __future__ import annotations

import math
import sys
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

MAX_TURN = 200
START_GOLD = 500
START_WARRIORS = 3
MOVE_COST = 10
TRAIN_COST = 120
WORK_INCOME = 15
UPKEEP_PER_WARRIOR = 2
HQ_MAX_LEVEL = 5
BASE_MAX_LEVEL = 3
HQ_HEAL_COST = 1000
BASE_HEAL_COST = 500
LATE_GAME_THRESHOLD = 150
FORCE_ATTACK_TURNS = 20  # 이 턴 이상 공격이 없으면 무조건 공격
MAX_BASES = 3  # 확장할 거점(기지) 최대 개수


class HqLevelEntry(NamedTuple):
    upgrade_cost: int
    warrior_hp: int
    hp: int
    turret: int
    train_cap: int
    work_cap: int


class BaseLevelEntry(NamedTuple):
    cost: int
    hp: int
    turret: int
    work_cap: int


HQ_LEVELS: tuple[HqLevelEntry, ...] = (
    HqLevelEntry(0, 0, 0, 0, 0, 0),
    HqLevelEntry(0, 4, 10, 1, 1, 1),
    HqLevelEntry(600, 5, 15, 2, 1, 2),
    HqLevelEntry(1200, 6, 20, 2, 2, 3),
    HqLevelEntry(2400, 7, 25, 3, 2, 4),
    HqLevelEntry(3600, 8, 30, 3, 3, 5),
)

BASE_LEVELS: tuple[BaseLevelEntry, ...] = (
    BaseLevelEntry(0, 0, 0, 0),
    BaseLevelEntry(300, 6, 1, 1),
    BaseLevelEntry(600, 12, 1, 2),
    BaseLevelEntry(1000, 18, 2, 3),
)


class Side(Enum):
    LEFT = "A"
    RIGHT = "B"

    @property
    def opposite(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @classmethod
    def from_word(cls, w: str) -> Side:
        return cls.LEFT if w == "LEFT" else cls.RIGHT

    @classmethod
    def from_char(cls, c: str) -> Side:
        return cls.LEFT if c == "A" else cls.RIGHT


class BType(Enum):
    HQ = "HQ"
    BASE = "BASE"


class WState(Enum):
    STATIONARY = 0
    MOVING = 1


@dataclass(frozen=True)
class WarriorId:
    side: Side
    num: int

    def __str__(self) -> str:
        return f"{self.side.value}{self.num}"

    @classmethod
    def parse(cls, tok: str) -> WarriorId:
        assert tok and tok[0] in ("A", "B")
        return cls(Side.from_char(tok[0]), int(tok[1:]))


@dataclass
class Warrior:
    id: WarriorId
    region: int
    hp: int
    state: WState = WState.STATIONARY
    target: int = 0


@dataclass
class Building:
    region: int
    side: Side
    type: BType
    level: int = 1
    hp: int = 10

    def max_level(self) -> int:
        return HQ_MAX_LEVEL if self.type is BType.HQ else BASE_MAX_LEVEL

    def current_hp(self) -> int:
        if self.type is BType.HQ:
            return HQ_LEVELS[self.level].hp
        return BASE_LEVELS[self.level].hp

    def work_cap(self) -> int:
        if self.type is BType.HQ:
            return HQ_LEVELS[self.level].work_cap
        return BASE_LEVELS[self.level].work_cap

    def apply_upgrade(self) -> None:
        self.level += 1
        self.hp = self.current_hp()

    def upgrade_cost(self) -> int:
        if self.level < self.max_level():
            if self.type is BType.HQ:
                return HQ_LEVELS[self.level + 1].upgrade_cost
            else:
                return BASE_LEVELS[self.level + 1].cost
        return 0

    def repair_cost(self) -> int:
        return HQ_HEAL_COST if self.type is BType.HQ else BASE_HEAL_COST


@dataclass
class GameMap:
    N: int = 0
    K: int = 0
    x: list[int] = field(default_factory=list)
    y: list[int] = field(default_factory=list)
    strongholds: list[int] = field(default_factory=list)
    adj: list[list[int]] = field(default_factory=list)
    my_side: Side = Side.LEFT
    my_hq: int = 0
    opp_hq: int = 0

    def hq_of(self, s: Side) -> int:
        return 0 if s is Side.LEFT else self.N - 1


@dataclass
class GameState:
    gold: int = START_GOLD
    my_countdown: int = 5
    opp_countdown: int = 5
    warriors: list[Warrior] = field(default_factory=list)
    buildings: dict[int, Building] = field(default_factory=dict)
    expansion_targets: list[int] = field(default_factory=list)
    early_expansion_done: bool = False
    initial_train_done: bool = False

    prev_enemy_base_count: int = 0
    # 직전에 (5인 단일이든 4+4 분산이든) 공격을 감행했는지, 그리고 그 결과
    # 연속으로 몇 번 실패(=적 기지 수가 줄지 않음)했는지를 추적한다.
    # 공격 종류와 무관하게 갱신되어야 5인<->분산 모드 전환이 정상 동작한다.
    last_attack_launched: bool = False
    consecutive_failed_attacks: int = 0
    last_attack_turn: int = 0

    def find_building(self, region: int) -> Building | None:
        return self.buildings.get(region)

    def find_warrior(self, wid: WarriorId) -> Warrior | None:
        return next((w for w in self.warriors if w.id == wid), None)


@dataclass
class Actions:
    train_n: int = 0
    moves: list[tuple[WarriorId, int]] = field(default_factory=list)
    upgrades: list[int] = field(default_factory=list)


def make_base(region: int, s: Side) -> Building:
    return Building(region, s, BType.BASE, 1, BASE_LEVELS[1].hp)


# ----- path finding -----------------------------------------------
@dataclass
class Paths:
    nxt: list[list[int]]
    hop: list[list[int]]
    euc_dist: list[list[float]]


def calculate_paths(M: GameMap) -> Paths:
    N = M.N
    INF = float('inf')
    nxt = [[-1] * N for _ in range(N)]
    hop = [[0] * N for _ in range(N)]
    euc = [[INF] * N for _ in range(N)]
    prev = [[-1] * N for _ in range(N)]

    for src in range(N):
        dist = [INF] * N
        dist[src] = 0.0
        hop_src = [0] * N
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v in M.adj[u]:
                w = math.ceil(math.hypot(M.x[u] - M.x[v], M.y[u] - M.y[v]))
                nd = d + w
                if nd < dist[v] - 1e-9:
                    dist[v] = nd
                    hop_src[v] = hop_src[u] + 1
                    prev[src][v] = u
                    heapq.heappush(pq, (nd, v))
                elif abs(nd - dist[v]) < 1e-9:
                    if u < prev[src][v] or prev[src][v] == -1:
                        prev[src][v] = u

        euc[src] = dist[:]
        hop[src] = hop_src[:]

    for u in range(N):
        for v in range(N):
            if u == v:
                nxt[u][v] = u
                continue
            if euc[u][v] == INF:
                continue
            cur = v
            while prev[u][cur] != u:
                if prev[u][cur] == -1:
                    break
                cur = prev[u][cur]
            nxt[u][v] = cur

    return Paths(nxt, hop, euc)


# ----- I/O & game state update ------------------------------------
def readln() -> str:
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    return line.rstrip("\n")


def read_tokens() -> list[str]:
    return readln().split()


def parse_init() -> tuple[GameMap, GameState]:
    M = GameMap()

    t = read_tokens()
    assert len(t) >= 2 and t[0] == "READY"
    M.my_side = Side.from_word(t[1])

    t = read_tokens()
    M.N, M.K = int(t[0]), int(t[1])

    M.x = [int(v) for v in read_tokens()]
    M.y = [int(v) for v in read_tokens()]

    M.strongholds = sorted(int(v) for v in read_tokens())

    M.adj = [[] for _ in range(M.N)]
    for r in range(M.N):
        t = read_tokens()
        deg = int(t[0])
        M.adj[r] = sorted(int(v) for v in t[1:1 + deg])

    M.my_hq = M.hq_of(M.my_side)
    M.opp_hq = M.hq_of(M.my_side.opposite)

    S = GameState()
    opp = M.my_side.opposite
    for sfx in range(1, START_WARRIORS + 1):
        S.warriors.append(
            Warrior(WarriorId(M.my_side, sfx), M.my_hq, HQ_LEVELS[1].warrior_hp))
        S.warriors.append(
            Warrior(WarriorId(opp, sfx), M.opp_hq, HQ_LEVELS[1].warrior_hp))
    S.buildings[0] = Building(0, Side.LEFT, BType.HQ, 1, HQ_LEVELS[1].hp)
    S.buildings[M.N - 1] = Building(M.N - 1, Side.RIGHT, BType.HQ, 1, HQ_LEVELS[1].hp)

    S.prev_enemy_base_count = sum(1 for b in S.buildings.values() if b.side == opp and b.type != BType.HQ)

    print("OK", flush=True)
    return M, S


def read_turn_start() -> int | None:
    line = readln()
    if line == "FINISH":
        return None
    t = line.split()
    assert t and t[0] == "START"
    return int(t[2])


def read_turn_result(S: GameState, M: GameMap, submitted: Actions) -> None:
    move_cost_total = 0
    for wid, target in submitted.moves:
        b = S.find_building(target)
        cost = 0 if (b is not None and b.side is M.my_side) else MOVE_COST
        move_cost_total += cost
    train_cost_total = TRAIN_COST * submitted.train_n

    line = readln()
    if line == "FINISH":
        sys.exit(0)
    t = line.split()
    assert t and t[0] == "TURN"

    t = read_tokens()
    S.my_countdown = int(t[2])
    S.opp_countdown = int(t[4])

    t = read_tokens()
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        side = Side.from_char(r[0][0])
        region = int(r[1])
        b = S.find_building(region)
        if b is None:
            new_b = make_base(region, side)
            S.buildings[region] = new_b
            if side is M.my_side:
                S.gold -= BASE_LEVELS[1].cost
        else:
            max_level = HQ_MAX_LEVEL if b.type == BType.HQ else BASE_MAX_LEVEL
            if side is M.my_side:
                if b.level < max_level:
                    cost = b.upgrade_cost()
                    b.apply_upgrade()
                else:
                    cost = b.repair_cost()
                    b.hp = b.current_hp()
                S.gold -= cost
            else:
                if b.level < max_level:
                    b.apply_upgrade()
                else:
                    b.hp = b.current_hp()

    t = read_tokens()
    n = int(t[1])
    if n > 0:
        ids = read_tokens()
        for i in range(n):
            wid = WarriorId.parse(ids[i])
            hq_region = M.hq_of(wid.side)
            hq_b = S.find_building(hq_region)
            hq_level = hq_b.level if hq_b is not None else 1
            S.warriors.append(Warrior(wid, hq_region, HQ_LEVELS[hq_level].warrior_hp))

    t = read_tokens()
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        wid = WarriorId.parse(r[0])
        region = int(r[1])
        w = S.find_warrior(wid)
        if w is not None:
            w.region = region
            if wid.side is M.my_side:
                if w.state == WState.STATIONARY:
                    target = None
                    for mid, mtarget in submitted.moves:
                        if mid == wid:
                            target = mtarget
                            break
                    if target is not None:
                        w.state = WState.MOVING
                        w.target = target
                if w.state == WState.MOVING and w.region == w.target:
                    w.state = WState.STATIONARY

    t = read_tokens()
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        wid = WarriorId.parse(r[1])
        damage = int(r[2])
        w = S.find_warrior(wid)
        if w is not None:
            w.hp -= damage
    S.warriors = [w for w in S.warriors if w.hp > 0]

    t = read_tokens()
    n = int(t[1])
    for _ in range(n):
        r = read_tokens()
        region = int(r[1])
        dmg = int(r[2])
        b = S.find_building(region)
        if b is not None:
            b.hp -= dmg
    S.buildings = {reg: b for reg, b in S.buildings.items() if b.hp > 0}

    readln()

    S.gold -= move_cost_total
    S.gold -= train_cost_total

    income = 0
    for b in S.buildings.values():
        if b.side is not M.my_side:
            continue
        count = sum(1 for w in S.warriors if w.id.side is M.my_side and w.region == b.region)
        income += WORK_INCOME * min(count, b.work_cap())
    S.gold += income

    my_warriors = [w for w in S.warriors if w.id.side is M.my_side]
    my_warriors.sort(key=lambda w: w.id.num)
    for w in my_warriors:
        if S.gold >= UPKEEP_PER_WARRIOR:
            S.gold -= UPKEEP_PER_WARRIOR

    opp = M.my_side.opposite
    current_enemy_bases = sum(1 for b in S.buildings.values() if b.side == opp and b.type != BType.HQ)

    # 공격 종류(5인 단일 / 4+4 분산)에 상관없이, 직전에 공격을 감행했다면
    # 그 결과(적 기지 수가 실제로 줄었는지)에 따라 연속 실패 횟수를 갱신한다.
    if S.last_attack_launched:
        if current_enemy_bases < S.prev_enemy_base_count:
            S.consecutive_failed_attacks = 0
        else:
            S.consecutive_failed_attacks += 1
        S.last_attack_launched = False

    S.prev_enemy_base_count = current_enemy_bases


def emit(a: Actions) -> None:
    out: list[str] = ["COMMAND"]
    for wid, target in a.moves:
        out.append(f"MOVE {wid} {target}")
    for r in a.upgrades:
        out.append(f"UPGRADE {r}")
    if a.train_n > 0:
        out.append(f"TRAIN {a.train_n}")
    out.append("END")
    sys.stdout.write("\n".join(out) + "\n")
    sys.stdout.flush()


# =====================================================================
#                         DECIDE FUNCTION
# =====================================================================
def decide(S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
    a = Actions()
    my = M.my_side
    opp = my.opposite
    my_hq_reg = M.my_hq
    opp_hq_reg = M.opp_hq

    my_warriors = [w for w in S.warriors if w.id.side == my]
    my_buildings = [b for b in S.buildings.values() if b.side == my]
    if not my_buildings:
        return a

    my_hq = next(b for b in my_buildings if b.type == BType.HQ)
    enemy_warriors = [w for w in S.warriors if w.id.side == opp]

    my_warriors_by_region: dict[int, list[Warrior]] = defaultdict(list)
    for w in my_warriors:
        my_warriors_by_region[w.region].append(w)

    enemy_present_set = {ew.region for ew in enemy_warriors}
    enemy_count_by_region: dict[int, int] = defaultdict(int)
    for ew in enemy_warriors:
        enemy_count_by_region[ew.region] += 1

    own_regions_set = {b.region for b in my_buildings}

    def stationary_at(region: int) -> list[Warrior]:
        return [w for w in my_warriors_by_region.get(region, []) if w.state == WState.STATIONARY]

    def workers_now(region: int) -> int:
        return len(stationary_at(region))

    def can_move_now(w: Warrior) -> bool:
        return w.region not in enemy_present_set

    def is_own_building(region: int) -> bool:
        b = S.find_building(region)
        return b is not None and b.side == my

    total_cost = 0
    move_costs: list[int] = []
    upgrade_costs: list[int] = []

    def add_move(wid: WarriorId, target: int, cost: int) -> None:
        nonlocal total_cost
        a.moves.append((wid, target))
        move_costs.append(cost)
        total_cost += cost

    def add_upgrade(region: int, cost: int) -> None:
        nonlocal total_cost
        a.upgrades.append(region)
        upgrade_costs.append(cost)
        total_cost += cost

    def enforce_budget() -> None:
        nonlocal total_cost
        while total_cost > S.gold:
            if a.train_n > 0:
                a.train_n -= 1
                total_cost -= TRAIN_COST
            elif a.moves:
                a.moves.pop()
                total_cost -= move_costs.pop()
            elif a.upgrades:
                a.upgrades.pop()
                total_cost -= upgrade_costs.pop()
            else:
                a.train_n = 0
                a.moves.clear()
                a.upgrades.clear()
                total_cost = 0
                break

    moved_ids: set[WarriorId] = set()

    # ================================================================
    # 거점 확장 헬퍼 (초반 확장 모드 / 후반 거점 재건 공용)
    # ----------------------------------------------------------------
    # - 목표 거점에 적이 이미 와 있으면(=원정군이 조우해 소모전 중이면) 우리
    #   병력 수가 적보다 확실히 많아지도록 지원군을 추가로 보내, "1:1로 싸우다
    #   둘 다 죽는" 공멸을 막고 확장을 관철시킨다.
    # - 목표 거점을 향해 가던 병력이 죽어서 사라지면(주둔/이동 중인 아군이 0명)
    #   자동으로 다음 유휴 병력을 다시 보내 재시도한다.
    # - 목표를 적이 선점했다면 그 목표는 버리고 다른 거점으로 교체한다.
    # ================================================================
    def expand_bases(max_bases: int, reserve_starting_warriors: bool) -> None:
        unclaimed = [s for s in M.strongholds if S.find_building(s) is None]

        # 적이 차지했거나 이미 완성한 목표는 목록에서 제거
        S.expansion_targets = [
            t for t in S.expansion_targets
            if (b := S.find_building(t)) is None or b.side == my
        ]
        while len(S.expansion_targets) < max_bases and unclaimed:
            candidates = [s for s in unclaimed if s not in S.expansion_targets]
            if not candidates:
                break
            if len(S.expansion_targets) < 2:
                new_t = min(candidates, key=lambda s: P.hop[my_hq_reg][s])
            else:
                mid_candidates = [
                    s for s in candidates
                    if abs(P.hop[s][my_hq_reg] - P.hop[s][opp_hq_reg]) <= 2
                ]
                if mid_candidates:
                    new_t = min(mid_candidates, key=lambda s: P.hop[my_hq_reg][s])
                else:
                    new_t = min(candidates, key=lambda s: P.hop[my_hq_reg][s])
            S.expansion_targets.append(new_t)

        for i, target in enumerate(S.expansion_targets):
            if S.find_building(target) is not None:
                continue

            enemy_here = enemy_count_by_region[target]
            ours_here = workers_now(target)
            moving_here = sum(
                1 for w in my_warriors
                if w.state == WState.MOVING and w.target == target
            )

            if enemy_here > 0:
                # 이미 조우해 소모전 중: 병력 우위를 만들어 확실히 이긴다.
                need = enemy_here + 1 - (ours_here + moving_here)
                if need <= 0:
                    continue
            else:
                if moving_here > 0 or ours_here > 0:
                    continue
                need = 1

            idle: list[Warrior] = []
            for b in my_buildings:
                cap = b.work_cap()
                idle.extend(stationary_at(b.region)[cap:])
            idle = [w for w in idle if w.id not in moved_ids and can_move_now(w)]
            if reserve_starting_warriors and i >= 2:
                idle = [w for w in idle if w.id.num > START_WARRIORS]
            idle.sort(key=lambda w: P.hop[w.region][target])

            sent = 0
            for w in idle:
                if sent >= need:
                    break
                cost = MOVE_COST  # 목적지에 아직 아군 건물이 없으므로 항상 비용 발생
                if total_cost + cost > S.gold:
                    continue
                add_move(w.id, target, cost)
                moved_ids.add(w.id)
                sent += 1

        for s in S.expansion_targets:
            if (workers_now(s) > 0 and not enemy_count_by_region[s]
                    and S.find_building(s) is None):
                cost = BASE_LEVELS[1].cost
                if total_cost + cost <= S.gold:
                    add_upgrade(s, cost)

    # ================================================================
    # [1순위] 적 2명 이상이 우리 진영 깊숙이 침투 → 총력 방어 모드
    # ================================================================
    rush_detected = False
    for region, cnt in enemy_count_by_region.items():
        if cnt >= 2 and P.hop[region][my_hq_reg] + 1 < P.hop[region][opp_hq_reg]:
            rush_detected = True
            break

    if rush_detected:
        attacked_own = [b for b in my_buildings if enemy_count_by_region[b.region] >= 2]
        if attacked_own:
            hq_attacked = next((b for b in attacked_own if b.type == BType.HQ), None)
            if hq_attacked:
                defense_target = hq_attacked.region
            else:
                defense_target = min(attacked_own, key=lambda b: P.hop[b.region][my_hq_reg]).region
        else:
            closest_enemy_region = min(
                (r for r, cnt in enemy_count_by_region.items() if cnt >= 2),
                key=lambda r: P.hop[r][my_hq_reg]
            )
            defense_target = min(my_buildings, key=lambda b: P.hop[b.region][closest_enemy_region]).region

        for b in my_buildings:
            for w in stationary_at(b.region):
                if w.region == defense_target:
                    continue
                if w.id not in moved_ids and can_move_now(w):
                    cost = 0 if is_own_building(defense_target) else MOVE_COST
                    if total_cost + cost <= S.gold:
                        add_move(w.id, defense_target, cost)
                        moved_ids.add(w.id)

        for w in my_warriors:
            if w.region not in own_regions_set and w.state == WState.STATIONARY:
                if w.id not in moved_ids and can_move_now(w):
                    cost = 0 if is_own_building(defense_target) else MOVE_COST
                    if total_cost + cost <= S.gold:
                        add_move(w.id, defense_target, cost)
                        moved_ids.add(w.id)

        # 기지 업그레이드는 본부가 5레벨(최대 레벨)을 찍은 후에만 의미가 있으므로,
        # 방어 상황에서도 본부가 아직 5레벨 미만이면 기지는 업그레이드하지 않는다
        # (수리는 upgrade_cost()가 0이 되는 만렙 상태에서만 별도 처리되므로 여기 영향 없음).
        target_b = S.find_building(defense_target)
        if target_b and target_b.level < target_b.max_level():
            is_base = target_b.type is BType.BASE
            if not is_base or my_hq.level >= HQ_MAX_LEVEL:
                cost = target_b.upgrade_cost()
                if (total_cost + cost <= S.gold and
                    workers_now(defense_target) >= 1 and
                    defense_target not in enemy_present_set):
                    add_upgrade(defense_target, cost)

        train_cap = HQ_LEVELS[my_hq.level].train_cap
        max_train = min(train_cap, (S.gold - total_cost) // TRAIN_COST)
        if max_train > 0:
            a.train_n = max_train
            total_cost += max_train * TRAIN_COST

        enforce_budget()
        return a

    # ================================================================
    # [2순위] 평시 거점 방어: 공격받는 건물에 잉여 병력 지원
    # ================================================================
    attacked_buildings = [b for b in my_buildings if enemy_count_by_region[b.region] > 0]
    if attacked_buildings:
        attacked_buildings.sort(key=lambda b: (b.type != BType.HQ, -b.level))
        for target_b in attacked_buildings:
            target_region = target_b.region
            enemy_cnt = enemy_count_by_region[target_region]
            is_hq = target_b.type == BType.HQ

            if workers_now(target_region) >= enemy_cnt:
                continue

            if is_hq:
                pool = []
                for b in my_buildings:
                    pool.extend(stationary_at(b.region))
                pool = [w for w in pool if w.id not in moved_ids and can_move_now(w)]
                pool.sort(key=lambda w: P.hop[w.region][target_region])
                for w in pool:
                    if w.region == target_region:
                        continue
                    if total_cost + 0 > S.gold:
                        continue
                    add_move(w.id, target_region, 0)
                    moved_ids.add(w.id)
            else:
                needed = enemy_cnt - workers_now(target_region)
                if needed <= 0:
                    continue
                pool = []
                for b in my_buildings:
                    cap = b.work_cap()
                    st = stationary_at(b.region)
                    pool.extend(st[cap:])
                pool = [w for w in pool if w.id not in moved_ids and can_move_now(w)]
                pool.sort(key=lambda w: P.hop[w.region][target_region])
                sent = 0
                for w in pool:
                    if sent >= needed:
                        break
                    if w.region == target_region:
                        continue
                    if total_cost + 0 > S.gold:
                        continue
                    add_move(w.id, target_region, 0)
                    moved_ids.add(w.id)
                    sent += 1

    # ================================================================
    # [3순위] 초반 확장 모드 (거점 최대 MAX_BASES개, 초기 훈련 1회)
    # ================================================================
    my_bases = [b for b in my_buildings if b.type == BType.BASE]
    if not S.early_expansion_done and len(my_bases) >= MAX_BASES:
        S.early_expansion_done = True

    if not S.early_expansion_done:
        expand_bases(MAX_BASES, reserve_starting_warriors=True)

        a.train_n = 0
        if not S.initial_train_done and len(my_bases) >= 2 and S.gold - total_cost >= TRAIN_COST:
            a.train_n = 1
            total_cost += TRAIN_COST
            S.initial_train_done = True

        enforce_budget()
        return a

    # ================================================================
    # [4순위] 후반 운영 모드
    # ================================================================
    frontline_region = min(my_buildings, key=lambda b: P.hop[b.region][opp_hq_reg]).region

    # 업그레이드: 본부가 5레벨(HQ_MAX_LEVEL) 미만이면 오직 본부만 업그레이드한다.
    # 기지 업그레이드(레벨업)는 본부가 5레벨을 찍은 뒤에만 진행한다.
    # (기지를 새로 짓는 것 자체는 이 제한과 무관하며 거점 재건 로직에서 처리한다.)
    if my_hq.level < HQ_MAX_LEVEL:
        cost = HQ_LEVELS[my_hq.level + 1].upgrade_cost
        if (total_cost + cost <= S.gold and workers_now(my_hq_reg) > 0
                and not enemy_count_by_region[my_hq_reg] and my_hq_reg not in set(a.upgrades)):
            add_upgrade(my_hq_reg, cost)
    else:
        for b in my_bases:
            if b.level < BASE_MAX_LEVEL:
                cost = BASE_LEVELS[b.level + 1].cost
                if (total_cost + cost <= S.gold and workers_now(b.region) >= b.work_cap()
                        and not enemy_count_by_region[b.region] and b.region not in set(a.upgrades)):
                    add_upgrade(b.region, cost)
                    break

    # 노동 인력 배치
    for b in my_buildings:
        cap = b.work_cap()
        cur = workers_now(b.region)
        deficit = cap - cur
        if deficit <= 0:
            continue
        for other in my_buildings:
            if other.region == b.region:
                continue
            other_cap = other.work_cap()
            surplus = stationary_at(other.region)[other_cap:]
            if not surplus:
                continue
            surplus.sort(key=lambda w: P.hop[w.region][b.region])
            for w in surplus:
                if w.id in moved_ids or not can_move_now(w):
                    continue
                add_move(w.id, b.region, 0)
                moved_ids.add(w.id)
                deficit -= 1
                if deficit == 0:
                    break
            if deficit == 0:
                break

    # ★★★ 공격 목표 선정 ★★★
    def pick_best_target(from_region: int, attackers: list[Warrior], force: bool = False) -> int | None:
        enemy_strongholds = [
            s for s in M.strongholds
            if (b := S.find_building(s)) is not None and b.side == opp and b.type != BType.HQ
        ]
        # 적 BASE가 없으면 무조건 적 HQ 반환
        if not enemy_strongholds:
            return opp_hq_reg

        enemy_strongholds.sort(key=lambda s: (
            P.hop[from_region][s],
            enemy_count_by_region[s]
        ))
        for target in enemy_strongholds:
            defenders = enemy_count_by_region[target]
            if force or len(attackers) > defenders:
                # target이 적 HQ가 아닐 때만 nxt 체크 (적 HQ는 항상 허용)
                if target != opp_hq_reg and P.nxt[from_region][target] == opp_hq_reg:
                    continue
                return target
        # 강제 공격이면 첫 번째 적 BASE라도 반환
        if force and enemy_strongholds:
            return enemy_strongholds[0]
        return None

    def pick_dual_targets(attackers: list[Warrior]) -> list[int]:
        enemy_strongholds = [
            s for s in M.strongholds
            if (b := S.find_building(s)) is not None and b.side == opp and b.type != BType.HQ
        ]
        if len(enemy_strongholds) < 2:
            return []

        t1 = min(enemy_strongholds, key=lambda s: (
            P.hop[frontline_region][s],
            enemy_count_by_region[s]
        ))
        candidates = [s for s in enemy_strongholds if s != t1]
        candidates.sort(key=lambda s: -P.hop[opp_hq_reg][s])
        for s in candidates:
            if P.hop[t1][s] >= 2:
                return [t1, s]
        if candidates:
            return [t1, candidates[0]]
        return [t1]

    # ---- 최전방 공격 부대 확보 ----
    fb = S.find_building(frontline_region)
    attackers: list[Warrior] = []
    force_attack = (turn - S.last_attack_turn) >= FORCE_ATTACK_TURNS and S.last_attack_turn > 0
    # 적 BASE(HQ 제외)의 개수. 2개 이상 남아있어야 "양방향(4+4) 분산 공격"이 의미가 있다.
    # 0개(HQ만 남음)이거나 1개만 남았을 때는 자동으로 단일 목표 공격으로 돌아간다.
    enemy_bases_left = sum(1 for b in S.buildings.values() if b.side == opp and b.type != BType.HQ)

    if fb:
        fb_stationary = stationary_at(frontline_region)
        fb_cap = fb.work_cap()
        surplus = [w for w in fb_stationary[fb_cap:] if w.id not in moved_ids]

        if len(surplus) >= 5:
            attackers = surplus
        elif len(fb_stationary) >= 8:
            # 8명 이상 모였으면 전부 공격 부대로 사용 (기존에는 8명으로 캡되어
            # 9명, 10명이 모여도 초과 인원이 버려지던 버그를 수정)
            attackers = [w for w in fb_stationary if w.id not in moved_ids]
        elif len(fb_stationary) >= 5 and force_attack:
            attackers = [w for w in fb_stationary if w.id not in moved_ids]

    # ---- 분산 공격(8명, 4+4) or 단일 공격(5명) ----
    # 최근 공격(5인 단일이든 4+4 분산이든)이 2회 연속으로 적 거점을 뚫지 못했고,
    # 나눠 칠 만한 적 거점이 2개 이상 남아있을 때만 분산 공격 모드로 전환한다.
    force_split_attack = (S.consecutive_failed_attacks >= 2) and enemy_bases_left >= 2

    if force_split_attack and len(attackers) >= 8:
        dual_targets = pick_dual_targets(attackers)
        if len(dual_targets) == 2:
            t1, t2 = dual_targets
            # t1과 t2 중 어느 쪽에 더 가까운지를 기준으로 전체 인원을 나눈다.
            # (기존에는 4/4로 고정 분할하여 8명을 초과하는 병력이 버려졌음)
            attackers_sorted = sorted(
                attackers,
                key=lambda w: P.hop[w.region][t1] - P.hop[w.region][t2]
            )
            half = max(4, len(attackers_sorted) // 2)
            team1 = attackers_sorted[:half]
            team2 = attackers_sorted[half:]
            if not team2:
                # 병력이 정확히 8명일 때 안전장치
                team1 = attackers_sorted[:4]
                team2 = attackers_sorted[4:]

            for w in team1:
                cost = 0 if is_own_building(t1) else MOVE_COST
                if total_cost + cost <= S.gold and w.id not in moved_ids:
                    add_move(w.id, t1, cost)
                    moved_ids.add(w.id)
            for w in team2:
                cost = 0 if is_own_building(t2) else MOVE_COST
                if total_cost + cost <= S.gold and w.id not in moved_ids:
                    add_move(w.id, t2, cost)
                    moved_ids.add(w.id)

            # 분산 공격도 "공격을 감행했다"로 기록해야 성공/실패 추적이 정상 동작한다.
            # (기존에는 여기서 바로 return 해버려서 이 턴에 거점 재건/훈련 등
            #  나머지 로직이 통째로 스킵되고, 단일 공격 턴과 동작이 달라졌다.
            #  이제는 return 없이 아래 로직을 계속 이어서 처리한다.)
            S.last_attack_launched = True
            S.last_attack_turn = turn

    if len(attackers) >= 5 and not force_split_attack:
        # 적 거점이 없으면(enemy_bases_left == 0) pick_best_target이 자동으로
        # 적 본부(opp_hq_reg)를 반환하므로, 본부만 남았을 때도 이 경로로
        # 5명이 모이는 대로 자연스럽게 본부 러시를 계속한다.
        target = pick_best_target(frontline_region, attackers, force_attack)
        if target is not None:
            cost_per = 0 if is_own_building(target) else MOVE_COST
            total_move_cost = cost_per * len(attackers)
            if total_cost + total_move_cost <= S.gold:
                for w in attackers:
                    if w.id not in moved_ids:
                        add_move(w.id, target, cost_per)
                        moved_ids.add(w.id)
                S.last_attack_launched = True
                S.last_attack_turn = turn

    # 유랑 병력 처리
    for region in list(set(w.region for w in my_warriors if w.state == WState.STATIONARY and w.region not in own_regions_set)):
        local = [w for w in stationary_at(region) if w.id not in moved_ids]
        if not local:
            continue
        building_here = S.find_building(region)
        enemy_here = enemy_count_by_region[region]
        if building_here is None and enemy_here == 0:
            next_target = pick_best_target(region, local, force_attack)
            if next_target is None:
                retreat_cost = 0 if is_own_building(frontline_region) else MOVE_COST
                if total_cost + retreat_cost * len(local) <= S.gold:
                    for w in local:
                        if can_move_now(w) and w.id not in moved_ids:
                            add_move(w.id, frontline_region, retreat_cost)
                            moved_ids.add(w.id)
            else:
                cost_per = 0 if is_own_building(next_target) else MOVE_COST
                total_move_cost = cost_per * len(local)
                if total_cost + total_move_cost <= S.gold:
                    for w in local:
                        if w.id not in moved_ids:
                            add_move(w.id, next_target, cost_per)
                            moved_ids.add(w.id)

    # 잉여 병력 최전방으로
    for b in my_buildings:
        region = b.region
        if region == frontline_region:
            continue
        cap = b.work_cap()
        surplus = stationary_at(region)[cap:]
        for w in surplus:
            if w.id in moved_ids or not can_move_now(w):
                continue
            add_move(w.id, frontline_region, 0)
            moved_ids.add(w.id)

    # 거점 재건: 전멸/파괴 등으로 확보한 거점이 MAX_BASES에 못 미치면
    # expand_bases가 계속 재시도한다 (적과 조우해 소모전 중이면 지원군도 보낸다).
    if len(my_bases) < MAX_BASES:
        expand_bases(MAX_BASES, reserve_starting_warriors=True)

    # 훈련
    if turn >= LATE_GAME_THRESHOLD:
        a.train_n = 0
    else:
        train_cap = HQ_LEVELS[my_hq.level].train_cap
        max_train = min(train_cap, (S.gold - total_cost) // TRAIN_COST)
        if max_train > 0:
            a.train_n = max_train
            total_cost += max_train * TRAIN_COST

    enforce_budget()
    return a


def main() -> None:
    M, S = parse_init()
    P = calculate_paths(M)

    while (turn := read_turn_start()) is not None:
        a = decide(S, M, P, turn)
        emit(a)
        read_turn_result(S, M, a)


if __name__ == "__main__":
    main()