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
        """다음 레벨로 업그레이드하는 비용. 최대 레벨이면 0 반환 (수리 불가)"""
        if self.level < self.max_level():
            if self.type is BType.HQ:
                return HQ_LEVELS[self.level + 1].upgrade_cost
            else:
                return BASE_LEVELS[self.level + 1].cost
        return 0


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
    buildings: dict[int, Building] = field(default_factory=dict)  # region -> Building

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


# ----- path finding (타이 브레이크 수정) -----------------------------------
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


# ----- I/O & game state update --------------------------------------
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

    # ---- UPGRADE ----
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
            if side is M.my_side:
                max_level = HQ_MAX_LEVEL if b.type == BType.HQ else BASE_MAX_LEVEL
                if b.level < max_level:
                    cost = b.upgrade_cost()
                    b.apply_upgrade()
                else:
                    # 우리는 수리 명령을 내리지 않으므로 이 분기는 도달하지 않음
                    pass
                S.gold -= cost
            else:
                max_level = HQ_MAX_LEVEL if b.type == BType.HQ else BASE_MAX_LEVEL
                if b.level < max_level:
                    b.apply_upgrade()
                else:
                    b.hp = b.current_hp()  # 적이 수리한 경우 hp 회복

    # ---- TRAIN ----
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

    # ---- MOVE ----
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

    # ---- DAMAGE ----
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

    # ---- SIEGE ----
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

    # ---- END ----
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


# ========== 의사 결정 ==========
def decide(S: GameState, M: GameMap, P: Paths, turn: int) -> Actions:
    a = Actions()
    my = M.my_side
    opp = my.opposite
    my_hq_reg = M.my_hq
    opp_hq_reg = M.opp_hq

    my_warriors = [w for w in S.warriors if w.id.side == my]
    my_buildings = [b for b in S.buildings.values() if b.side == my]
    my_hq = next(b for b in my_buildings if b.type == BType.HQ)
    my_bases = [b for b in my_buildings if b.type == BType.BASE]
    enemy_warriors = [w for w in S.warriors if w.id.side == opp]
    enemy_hq = next(b for b in S.buildings.values() if b.side == opp and b.type == BType.HQ)

    my_warriors_by_region: dict[int, list[Warrior]] = defaultdict(list)
    for w in my_warriors:
        my_warriors_by_region[w.region].append(w)

    enemy_present_set = {ew.region for ew in enemy_warriors}

    def friendly_present(region: int) -> bool:
        return region in my_warriors_by_region

    def stationary_at(region: int) -> list[Warrior]:
        return [w for w in my_warriors_by_region.get(region, []) if w.state == WState.STATIONARY]

    def workers_now(region: int) -> int:
        return len(stationary_at(region))

    def enemy_present(region: int) -> bool:
        return region in enemy_present_set

    def enemy_count(region: int) -> int:
        return sum(1 for ew in enemy_warriors if ew.region == region)

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

    def pop_move() -> None:
        nonlocal total_cost
        if a.moves:
            a.moves.pop()
            total_cost -= move_costs.pop()

    def pop_upgrade() -> None:
        nonlocal total_cost
        if a.upgrades:
            a.upgrades.pop()
            total_cost -= upgrade_costs.pop()

    def enforce_budget() -> None:
        nonlocal total_cost
        while total_cost > S.gold:
            if a.train_n > 0:
                a.train_n -= 1
                total_cost -= TRAIN_COST
            elif a.moves:
                pop_move()
            elif a.upgrades:
                pop_upgrade()
            else:
                a.train_n = 0
                a.moves.clear()
                a.upgrades.clear()
                total_cost = 0
                break

    gold = S.gold
    moved_ids: set[WarriorId] = set()
    upgrade_set: set[int] = set()

    my_military = sum(w.hp for w in my_warriors)
    enemy_military = sum(w.hp for w in enemy_warriors)
    dist_between_hqs = P.hop[my_hq_reg][opp_hq_reg]
    THREAT_DIST = max(5, min(12, dist_between_hqs // 3))

    if my_warriors:
        main_army_region = max(my_warriors_by_region.keys(), key=lambda r: len(my_warriors_by_region[r]))
        our_main_dist_to_opp_hq = P.hop[main_army_region][opp_hq_reg]
    else:
        our_main_dist_to_opp_hq = 999

    enemy_front_dist = min((P.hop[ew.region][my_hq_reg] for ew in enemy_warriors), default=999)

    hq_defenders = sum(1 for w in my_warriors if w.state == WState.STATIONARY and w.region == my_hq_reg)
    enemy_at_hq = enemy_present(my_hq_reg)
    enemy_hq_count = enemy_count(my_hq_reg)

    # 적이 우리 건물이 있는 어느 구역에라도 있으면 위협으로 간주
    enemy_at_any_building = any(enemy_present(b.region) for b in my_buildings)

    imminent_threat = False
    if enemy_military > 0:
        if enemy_military > my_military * 1.2 and enemy_front_dist <= THREAT_DIST:
            imminent_threat = True
        if enemy_at_hq or enemy_front_dist <= 2 or enemy_at_any_building:
            imminent_threat = True

    strategy = "ECON"
    if imminent_threat:
        if not enemy_at_hq and our_main_dist_to_opp_hq < enemy_front_dist and len(my_warriors) >= 4:
            strategy = "BASE_RACE"
        else:
            strategy = "DEFEND"

    # ---------- BASE_RACE 모드 ----------
    if strategy == "BASE_RACE":
        for b in my_buildings:
            cap = b.work_cap()
            pool = stationary_at(b.region)
            surplus = pool[cap:]
            for w in surplus:
                if w.id in moved_ids or not can_move_now(w):
                    continue
                if w.region == opp_hq_reg:
                    continue
                if gold >= MOVE_COST:
                    add_move(w.id, opp_hq_reg, MOVE_COST)
                    moved_ids.add(w.id)
                    gold -= MOVE_COST
        train_cap = HQ_LEVELS[my_hq.level].train_cap
        max_train = min(train_cap, gold // TRAIN_COST)
        if max_train > 0:
            a.train_n = max_train
            total_cost += max_train * TRAIN_COST
            gold -= max_train * TRAIN_COST
        enforce_budget()
        return a

    # ---------- DEFEND 모드 (본부 우선, 거점별 방어) ----------
    if strategy == "DEFEND":
        # 적이 침입한 우리 건물 목록
        threatened = [b for b in my_buildings if enemy_present(b.region)]
        
        # 본부가 위협받는지 확인
        hq_threatened = any(b.type == BType.HQ for b in threatened)
        
        if hq_threatened:
            # 본부 방어: 모든 건물의 모든 정지 병력을 본부로 보낸다 (일꾼도 예외 없이)
            for b in my_buildings:
                if b.region == my_hq_reg:
                    continue  # 이미 본부에 있는 병력은 그대로
                # 이 건물에 있는 정지 병력 중 이동 명령이 없는 모든 병력
                all_stationary = [w for w in stationary_at(b.region) if w.id not in moved_ids]
                for w in all_stationary:
                    if not can_move_now(w):
                        continue  # 적이 있는 곳이면 못 움직임
                    add_move(w.id, my_hq_reg, 0)  # 아군 건물로 이동 무료
                    moved_ids.add(w.id)
        else:
            # 본부가 아닌 거점 방어: 위협받는 거점 순서대로 처리
            # (가장 위협이 큰 순서 = 적 수가 많은 순서)
            threatened.sort(key=lambda b: enemy_count(b.region), reverse=True)
            for target_b in threatened:
                target_region = target_b.region
                enemy_cnt = enemy_count(target_region)
                # 현재 이 지역에 있는 방어 병력 (이동 명령 없는 정지 병력)
                current_defenders = [w for w in stationary_at(target_region) if w.id not in moved_ids]
                # 적보다 1명 이상 많아지도록 필요한 병력 수
                deficit = max(0, enemy_cnt - len(current_defenders) + 1)
                if deficit <= 0:
                    continue
                
                # 다른 건물들에서 잉여 병력(work_cap 초과)만 차출
                other_buildings = [b for b in my_buildings if b.region != target_region]
                other_buildings.sort(key=lambda b: P.hop[b.region][target_region])
                for other_b in other_buildings:
                    # 이미 위협받는 건물에서는 병력을 빼지 않음
                    if any(tb.region == other_b.region for tb in threatened):
                        continue
                    cap = other_b.work_cap()
                    pool = [w for w in stationary_at(other_b.region) if w.id not in moved_ids]
                    surplus = pool[cap:]  # work_cap 초과분
                    for w in surplus:
                        if deficit <= 0:
                            break
                        if w.id in moved_ids:
                            continue
                        if not can_move_now(w):
                            continue
                        add_move(w.id, target_region, 0)  # 아군 건물로 이동 무료
                        moved_ids.add(w.id)
                        deficit -= 1
                    if deficit <= 0:
                        break

        train_cap = HQ_LEVELS[my_hq.level].train_cap
        max_train = min(train_cap, gold // TRAIN_COST)
        if max_train > 0:
            a.train_n = max_train
            total_cost += max_train * TRAIN_COST
            gold -= max_train * TRAIN_COST
        enforce_budget()
        return a

    # ---------- ECON 모드 (평시) ----------
    # 본부 업그레이드 최우선
    if my_hq.level < 5:
        cost_next = HQ_LEVELS[my_hq.level + 1].upgrade_cost
        if gold >= cost_next and friendly_present(my_hq_reg) and not enemy_present(my_hq_reg):
            add_upgrade(my_hq_reg, cost_next)
            enforce_budget()
            return a

    late_game = turn >= LATE_GAME_THRESHOLD
    focus_hq = late_game and my_hq.level < 5

    should_upgrade_hq_now = False
    if my_hq.level < 5:
        if enemy_hq.level > my_hq.level or my_military > enemy_military:
            if gold >= HQ_LEVELS[my_hq.level + 1].upgrade_cost and friendly_present(my_hq_reg) and not enemy_present(my_hq_reg):
                should_upgrade_hq_now = True

    if focus_hq and not should_upgrade_hq_now and my_hq.level < 5:
        if gold >= HQ_LEVELS[my_hq.level + 1].upgrade_cost and friendly_present(my_hq_reg) and not enemy_present(my_hq_reg):
            should_upgrade_hq_now = True

    if should_upgrade_hq_now and my_hq_reg not in upgrade_set:
        cost = HQ_LEVELS[my_hq.level + 1].upgrade_cost
        add_upgrade(my_hq_reg, cost)
        upgrade_set.add(my_hq_reg)
        gold -= cost
        enforce_budget()
        return a

    unclaimed = [s for s in M.strongholds if S.find_building(s) is None]
    all_claimed = len(unclaimed) == 0

    # 1. 새 기지 건설
    for s in unclaimed:
        if friendly_present(s) and not enemy_present(s) and gold >= BASE_LEVELS[1].cost:
            if s not in upgrade_set:
                add_upgrade(s, BASE_LEVELS[1].cost)
                upgrade_set.add(s)
                gold -= BASE_LEVELS[1].cost

    # 2. 업그레이드 (기지) – HQ 5레벨 전까지는 기지 2레벨 금지
    if all_claimed:
        if my_hq.level < 5 and gold >= HQ_LEVELS[my_hq.level + 1].upgrade_cost:
            if friendly_present(my_hq_reg) and not enemy_present(my_hq_reg) and my_hq_reg not in upgrade_set:
                cost = HQ_LEVELS[my_hq.level + 1].upgrade_cost
                add_upgrade(my_hq_reg, cost)
                upgrade_set.add(my_hq_reg)
                gold -= cost
        elif my_hq.level >= 5 and any(b.level < 2 for b in my_bases):
            for b in my_bases:
                if b.level < 2 and workers_now(b.region) >= b.work_cap() and gold >= BASE_LEVELS[b.level + 1].cost:
                    if friendly_present(b.region) and not enemy_present(b.region) and b.region not in upgrade_set:
                        cost = BASE_LEVELS[b.level + 1].cost
                        add_upgrade(b.region, cost)
                        upgrade_set.add(b.region)
                        gold -= cost
                        break
        elif my_hq.level >= 5 and any(b.level < 3 for b in my_bases):
            for b in my_bases:
                if b.level < 3 and workers_now(b.region) >= b.work_cap() and gold >= BASE_LEVELS[b.level + 1].cost:
                    if friendly_present(b.region) and not enemy_present(b.region) and b.region not in upgrade_set:
                        cost = BASE_LEVELS[b.level + 1].cost
                        add_upgrade(b.region, cost)
                        upgrade_set.add(b.region)
                        gold -= cost
                        break

    # 3. 수리 로직 완전 제거

    # 4. 일꾼 배치 (기존 로직과 동일)
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
            other_cur = workers_now(other.region)
            surplus = max(0, other_cur - other_cap)
            if surplus == 0:
                continue
            idle_pool = stationary_at(other.region)[other_cap:]
            idle_pool.sort(key=lambda w: P.hop[w.region][b.region])
            for w in idle_pool:
                if w.id in moved_ids or enemy_present(other.region):
                    continue
                if not can_move_now(w):
                    continue
                add_move(w.id, b.region, 0)  # 아군 건물로 이동은 무료
                moved_ids.add(w.id)
                deficit -= 1
                if deficit == 0:
                    break
            if deficit == 0:
                break

    # 5. 확장 (중립 거점)
    idle = []
    for b in my_buildings:
        pool = stationary_at(b.region)
        idle.extend(pool[b.work_cap():])
    idle = [w for w in idle if w.id not in moved_ids]

    moving_to_unclaimed = any(w.state == WState.MOVING and w.target in unclaimed for w in my_warriors)

    if not moving_to_unclaimed and idle and unclaimed and gold >= MOVE_COST:
        still_unclaimed = [s for s in unclaimed if workers_now(s) == 0 and s not in upgrade_set]
        if still_unclaimed:
            still_unclaimed.sort(key=lambda s: P.hop[my_hq_reg][s])
            target = still_unclaimed[0]
            if not enemy_present(target):
                available = [w for w in idle if can_move_now(w)]
                if available:
                    w = min(available, key=lambda w: P.hop[w.region][target])
                    add_move(w.id, target, MOVE_COST)
                    moved_ids.add(w.id)
                    gold -= MOVE_COST

    # ========== 6. 공격 (5명 초과 & 1.5배 우위 시 전원 출격) ==========
    # 공격 목표: 적 점령 stronghold + 적 본부
    enemy_targets = [s for s in M.strongholds if not is_own_building(s)]
    if not enemy_targets:
        enemy_targets = [opp_hq_reg]

    if enemy_targets:
        # 적 본부에 가까운 순서로 아군 건물 정렬 (가장 전방부터 공격 시도)
        front_buildings = sorted(my_buildings, key=lambda b: P.hop[b.region][opp_hq_reg])
        for b in front_buildings:
            region = b.region
            # 현재 이 지역에서 이동 명령이 없고 정지해 있는 병력만 수집
            local_garrison = [w for w in stationary_at(region) if w.id not in moved_ids and can_move_now(w)]
            
            # 조건 1: 5명보다 많아야 함 (6명 이상)
            if len(local_garrison) <= 5:
                continue
                
            # 이 지역에서 가장 가까운 적 거점 탐색
            target_sh = min(enemy_targets, key=lambda s: P.hop[region][s])
            defenders = enemy_count(target_sh)
            
            # 조건 2: 수비병보다 1.5배 많아야 함
            if len(local_garrison) > defenders * 1.5:
                cost_per_move = 0 if is_own_building(target_sh) else MOVE_COST
                for w in local_garrison:
                    if gold >= cost_per_move:
                        add_move(w.id, target_sh, cost_per_move)
                        moved_ids.add(w.id)
                        gold -= cost_per_move
                    else:
                        break  # 골드 부족 시 중단

    # 7. 잉여 병력 최전방 집결 (공격에 사용되지 않은 work_cap 초과 병력)
    if my_bases:
        frontline_reg = min(my_bases, key=lambda b: P.hop[b.region][opp_hq_reg]).region
    else:
        frontline_reg = my_hq_reg

    for b in my_buildings:
        cap = b.work_cap()
        pool = stationary_at(b.region)
        # work_cap 초과 병력 중 moved_ids가 아닌 것만
        surplus = [w for w in pool[cap:] if w.id not in moved_ids]
        for w in surplus:
            if w.region == frontline_reg or not can_move_now(w):
                continue
            cost = 0 if is_own_building(frontline_reg) else MOVE_COST
            if gold >= cost:
                add_move(w.id, frontline_reg, cost)
                moved_ids.add(w.id)
                gold -= cost

    # 8. 훈련 (기존 로직 유지)
    if focus_hq:
        a.train_n = 0
    else:
        train_cap = HQ_LEVELS[my_hq.level].train_cap
        idle = []
        for b in my_buildings:
            pool = stationary_at(b.region)
            idle.extend(pool[b.work_cap():])
        idle = [w for w in idle if w.id not in moved_ids]
        can_start_new = unclaimed and idle and gold >= MOVE_COST

        if not can_start_new:
            total_work_cap = sum(b.work_cap() for b in my_buildings)
            desired = total_work_cap * 3
            future_expansion_cost = BASE_LEVELS[1].cost if unclaimed else 0
            if len(my_warriors) < desired and gold >= TRAIN_COST + future_expansion_cost:
                train_n = min(train_cap, (gold - future_expansion_cost) // TRAIN_COST)
                if train_n > 0:
                    a.train_n = train_n
                    total_cost += train_n * TRAIN_COST
                    gold -= train_n * TRAIN_COST

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