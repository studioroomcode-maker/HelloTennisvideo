from core.scoring import MatchScorer

def test_love_game_and_end_change():
    s = MatchScorer(["A1", "B1"])
    for _ in range(4): s.award_point("A")
    assert s.games == [1, 0]
    assert s.snapshot().server_id == "B1"
    assert s.snapshot().team_a_end == "far"

def test_deuce_advantage():
    s = MatchScorer(["A1", "B1"])
    for w in ["A","A","A","B","B","B","A","B","B","B"]:
        s.award_point(w)
    assert s.games == [0, 1]

def test_tiebreak_set():
    s = MatchScorer(["A1", "B1"])
    for g in range(12):
        team = "A" if g % 2 == 0 else "B"
        for _ in range(4): s.award_point(team)
    assert s.tiebreak
    for _ in range(7): s.award_point("A")
    assert s.sets == [1, 0]


def test_team_a_can_start_at_far_end():
    s = MatchScorer(["A1", "B1"], team_a_start_end="far")
    assert s.snapshot().team_a_end == "far"
    for _ in range(4): s.award_point("A")
    assert s.snapshot().team_a_end == "near"


def test_change_ends_can_be_disabled():
    from core.scoring import MatchScorer
    sc = MatchScorer(server_rotation=["A1", "B1"], team_a_start_end="far", change_ends=False)
    for _ in range(4):
        sc.award_point("A")
    assert sc.snapshot().completed_games == 1
    assert sc.snapshot().team_a_end == "far"
    sc2 = MatchScorer(server_rotation=["A1", "B1"], team_a_start_end="far")
    for _ in range(4):
        sc2.award_point("A")
    assert sc2.snapshot().team_a_end == "near"
