from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

POINT_LABELS = ["0", "15", "30", "40"]


@dataclass
class ScoreSnapshot:
    set_games_a: int = 0
    set_games_b: int = 0
    point_a: str = "0"
    point_b: str = "0"
    sets_a: int = 0
    sets_b: int = 0
    tiebreak: bool = False
    tiebreak_a: int = 0
    tiebreak_b: int = 0
    server_id: str = "A1"
    team_a_end: str = "near"
    completed_games: int = 0

    def to_dict(self):
        return asdict(self)


class MatchScorer:
    """Deterministic tennis scoring from point-winner sequence.

    This is deliberately independent from video/AI. Point winners are the source
    of truth; score, server order and end changes are regenerated from scratch.
    """

    def __init__(
        self,
        server_rotation: list[str] | None = None,
        no_ad: bool = False,
        tiebreak_at_6_all: bool = True,
        team_a_start_end: str = "near",
        change_ends: bool = True,
    ):
        self.server_rotation = server_rotation or ["A1", "B1"]
        if not self.server_rotation:
            raise ValueError("server_rotation cannot be empty")
        self.no_ad = no_ad
        self.tiebreak_at_6_all = tiebreak_at_6_all
        self.team_a_start_near = team_a_start_end != "far"
        self.change_ends = change_ends  # casual matches often stay on one end all set
        self.reset()

    def reset(self):
        self.sets = [0, 0]
        self.games = [0, 0]
        self.points = [0, 0]
        self.adv = None
        self.tiebreak = False
        self.tb = [0, 0]
        self.completed_games = 0
        self.game_server_index = 0
        self.tb_points_played = 0
        self.team_a_near = self.team_a_start_near

    @staticmethod
    def _team_index(winner: str | int) -> int:
        if winner in ("A", "TEAM_A", 0):
            return 0
        if winner in ("B", "TEAM_B", 1):
            return 1
        raise ValueError(f"Unknown winner: {winner}")

    @staticmethod
    def _server_team(server_id: str) -> int:
        return 0 if str(server_id).upper().startswith("A") else 1

    def _current_server_id(self) -> str:
        if not self.tiebreak:
            return self.server_rotation[self.game_server_index % len(self.server_rotation)]

        # Tie-break: first server serves one point; subsequent servers serve two.
        # game_server_index identifies the player whose service game would be next.
        n = self.tb_points_played
        if n == 0:
            offset = 0
        else:
            offset = 1 + ((n - 1) // 2)
        return self.server_rotation[(self.game_server_index + offset) % len(self.server_rotation)]

    def _point_labels(self):
        if self.tiebreak:
            return str(self.tb[0]), str(self.tb[1])
        if self.points[0] >= 3 and self.points[1] >= 3:
            if self.adv == 0:
                return "AD", "40"
            if self.adv == 1:
                return "40", "AD"
            return "40", "40"
        return POINT_LABELS[min(self.points[0], 3)], POINT_LABELS[min(self.points[1], 3)]

    def snapshot(self) -> ScoreSnapshot:
        pa, pb = self._point_labels()
        return ScoreSnapshot(
            set_games_a=self.games[0],
            set_games_b=self.games[1],
            point_a=pa,
            point_b=pb,
            sets_a=self.sets[0],
            sets_b=self.sets[1],
            tiebreak=self.tiebreak,
            tiebreak_a=self.tb[0],
            tiebreak_b=self.tb[1],
            server_id=self._current_server_id(),
            team_a_end="near" if self.team_a_near else "far",
            completed_games=self.completed_games,
        )

    def _change_ends_after_game_if_needed(self):
        # Global completed-game parity reproduces normal end changes across sets.
        if self.change_ends and self.completed_games % 2 == 1:
            self.team_a_near = not self.team_a_near

    def _start_next_game(self):
        self.points = [0, 0]
        self.adv = None
        self.game_server_index = (self.game_server_index + 1) % len(self.server_rotation)

    def _award_game(self, team: int):
        self.games[team] += 1
        self.completed_games += 1
        self._change_ends_after_game_if_needed()

        # Set won before tiebreak.
        if self.games[team] >= 6 and self.games[team] - self.games[1-team] >= 2:
            self.sets[team] += 1
            self.games = [0, 0]
            self._start_next_game()
            return

        # Enter a 6-all tiebreak. The next server is the player whose service game
        # would have been next, so advance service rotation once.
        if self.tiebreak_at_6_all and self.games == [6, 6]:
            self.points = [0, 0]
            self.adv = None
            self.game_server_index = (self.game_server_index + 1) % len(self.server_rotation)
            self.tiebreak = True
            self.tb = [0, 0]
            self.tb_points_played = 0
            return

        self._start_next_game()

    def _award_regular_point(self, team: int):
        other = 1 - team

        if self.no_ad:
            if self.points[team] >= 3 and self.points[other] >= 3:
                self._award_game(team)
                return
            self.points[team] += 1
            if self.points[team] >= 4:
                self._award_game(team)
            return

        # Deuce / advantage handling.
        if self.points[0] >= 3 and self.points[1] >= 3:
            if self.adv is None:
                self.adv = team
            elif self.adv == team:
                self._award_game(team)
            else:
                self.adv = None
            return

        self.points[team] += 1
        if self.points[team] >= 4 and self.points[team] - self.points[other] >= 2:
            self._award_game(team)

    def _award_tiebreak_point(self, team: int):
        self.tb[team] += 1
        self.tb_points_played += 1

        # Change ends every six tiebreak points.
        if self.change_ends and self.tb_points_played % 6 == 0:
            self.team_a_near = not self.team_a_near

        if self.tb[team] >= 7 and self.tb[team] - self.tb[1-team] >= 2:
            self.games[team] += 1  # 7-6 set score
            self.completed_games += 1
            self.sets[team] += 1
            self.games = [0, 0]
            self.tiebreak = False
            self.tb = [0, 0]
            self.tb_points_played = 0
            # Next set: server after last tiebreak server rotation position is
            # awkward to infer perfectly from only game_server_index. For MVP,
            # advance one service slot from the tiebreak starter; UI can correct
            # server rotation if a resumed/multi-file match disagrees.
            self.game_server_index = (self.game_server_index + 1) % len(self.server_rotation)
            self.points = [0, 0]
            self.adv = None

    def award_point(self, winner: str | int):
        team = self._team_index(winner)
        if self.tiebreak:
            self._award_tiebreak_point(team)
        else:
            self._award_regular_point(team)

    def replay(self, winners: Iterable[str | int]):
        self.reset()
        rows = []
        for i, winner in enumerate(winners, 1):
            before = self.snapshot().to_dict()
            self.award_point(winner)
            after = self.snapshot().to_dict()
            rows.append({"index": i, "winner": winner, "before": before, "after": after})
        return rows


def build_score_rows(points: list[dict], settings: dict) -> list[dict]:
    rotation = settings.get("server_rotation") or ["A1", "B1"]
    scorer = MatchScorer(
        server_rotation=rotation,
        no_ad=bool(settings.get("no_ad", False)),
        tiebreak_at_6_all=bool(settings.get("tiebreak_at_6_all", True)),
        team_a_start_end=settings.get("team_a_start_end") or "near",
        change_ends=settings.get("change_ends", True) is not False,
    )
    rows = []
    for point in points:
        before = scorer.snapshot().to_dict()
        winner = point.get("winner")
        if winner in ("A", "B", "TEAM_A", "TEAM_B"):
            scorer.award_point(winner)
        after = scorer.snapshot().to_dict()
        rows.append({"point_id": point.get("id"), "before": before, "after": after})
    return rows
