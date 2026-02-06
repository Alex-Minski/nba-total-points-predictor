from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd

def parse_score(ss: str) -> Optional[Tuple[int, int]]:
    try:
        a, b = ss.split("-")
        return int(a), int(b)
    except Exception:
        return None

@dataclass
class RollingConfig:
    window: int = 10
    min_games: int = 5

def build_rolling_features(games: pd.DataFrame, cfg: RollingConfig) -> pd.DataFrame:
    """Build pre-game rolling features for each row (game).

    games must be sorted ascending by date.
    Required columns: date, home_id, away_id, home_pts, away_pts
    """
    # Per-team history buffers
    history: Dict[int, List[Tuple[int,int,int]]] = {}  # team_id -> list of (pts_for, pts_against, total)

    rows = []
    for _, g in games.iterrows():
        home = int(g.home_id); away = int(g.away_id)
        h_pts = int(g.home_pts); a_pts = int(g.away_pts)
        total = h_pts + a_pts

        def last_stats(team_id: int):
            h = history.get(team_id, [])
            tail = h[-cfg.window:]
            if len(tail) < cfg.min_games:
                return None
            pf = sum(x[0] for x in tail) / len(tail)
            pa = sum(x[1] for x in tail) / len(tail)
            tot = sum(x[2] for x in tail) / len(tail)
            return pf, pa, tot, len(tail)

        home_stats = last_stats(home)
        away_stats = last_stats(away)

        if home_stats and away_stats:
            h_pf, h_pa, h_tot, h_n = home_stats
            a_pf, a_pa, a_tot, a_n = away_stats

            rows.append({
                "date": g.date,
                "home_id": home,
                "away_id": away,
                "home_name": g.home_name,
                "away_name": g.away_name,
                "home_pf_roll": h_pf,
                "home_pa_roll": h_pa,
                "home_tot_roll": h_tot,
                "home_n": h_n,
                "away_pf_roll": a_pf,
                "away_pa_roll": a_pa,
                "away_tot_roll": a_tot,
                "away_n": a_n,
                # simple matchup features
                "exp_total_mean": (h_tot + a_tot) / 2.0,
                "exp_total_mix": (h_pf + a_pf + h_pa + a_pa) / 2.0,
                "y_total": total,
            })

        # update histories AFTER using the game (prevents leakage)
        history.setdefault(home, []).append((h_pts, a_pts, total))
        history.setdefault(away, []).append((a_pts, h_pts, total))

    return pd.DataFrame(rows)
