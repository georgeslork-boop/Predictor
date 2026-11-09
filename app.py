import json
import re
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

LEAGUES = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Serie A": "Serie_A",
    "Bundesliga": "Bundesliga",
}
HOME_ADVANTAGE = 0.20
MAX_GOALS = 10
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def current_season():
    t = date.today()
    return t.year if t.month >= 8 else t.year - 1


def fetch_dates(league, season):
    """Επιστρέφει τη λίστα αγώνων από την Understat (δύο μέθοδοι)."""
    page = f"https://understat.com/league/{league}/{season}"
    problems = []

    # Μέθοδος 1: εσωτερικό JSON endpoint
    try:
        r = requests.get(
            f"https://understat.com/getLeagueData/{league}/{season}",
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": UA,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Accept-Encoding": "gzip, deflate",
                "Referer": page,
            },
            timeout=20,
        )
        try:
            return r.json()["dates"]
        except Exception:
            problems.append(
                f"endpoint status={r.status_code} body={r.text[:100]!r}"
            )
    except Exception as e:
        problems.append(f"endpoint error={e}")

    # Μέθοδος 2: δεδομένα μέσα στη σελίδα HTML
    try:
        r = requests.get(
            page,
            headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"},
            timeout=20,
        )
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", r.text)
        if m:
            raw = m.group(1).encode("ascii", "backslashreplace").decode("unicode_escape")
            return json.loads(raw)
        problems.append(f"html status={r.status_code} body={r.text[:100]!r}")
    except Exception as e:
        problems.append(f"html error={e}")

    raise RuntimeError(" || ".join(problems))


@st.cache_data(ttl=3600, show_spinner="Φόρτωση δεδομένων από Understat...")
def load_matches(league_code):
    season = current_season()
    rows = []
    errors = []
    for s in (season, season - 1):
        try:
            matches = fetch_dates(league_code, s)
        except Exception as e:
            errors.append(f"[{s}] {e}")
            continue
        for r in matches:
            if not r.get("isResult"):
                continue
            rows.append(
                {
                    "date": r["datetime"],
                    "season": s,
                    "home": r["h"]["title"],
                    "away": r["a"]["title"],
                    "hxg": float(r["xG"]["h"]),
                    "axg": float(r["xG"]["a"]),
                    "hg": int(r["goals"]["h"]),
                    "ag": int(r["goals"]["a"]),
                }
            )
    if not rows:
        raise RuntimeError("Δεν επιστράφηκαν δεδομένα. " + " | ".join(errors))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def team_games(df, team, n, same_venue, is_home):
    d = df[df.home == team]
    h = pd.DataFrame(
        {
            "Ημερομηνία": d.date.dt.date,
            "Αντίπαλος": d.away,
            "Έδρα": "Εντός",
            "xG υπέρ": d.hxg,
            "xG κατά": d.axg,
            "Γκολ": d.hg.astype(str) + "-" + d.ag.astype(str),
        }
    )
    d = df[df.away == team]
    a = pd.DataFrame(
        {
            "Ημερομηνία": d.date.dt.date,
            "Αντίπαλος": d.home,
            "Έδρα": "Εκτός",
            "xG υπέρ": d.axg,
            "xG κατά": d.hxg,
            "Γκολ": d.hg.astype(str) + "-" + d.ag.astype(str),
        }
    )
    if same_venue:
        g = h if is_home else a
    else:
        g = pd.concat([h, a])
    return g.sort_values("Ημερομηνία", ascending=False).head(n)


def predict(hg, ag):
    home_xg = (hg["xG υπέρ"].mean() + ag["xG κατά"].mean()) / 2 + HOME_ADVANTAGE
    away_xg = (ag["xG υπέρ"].mean() + hg["xG κατά"].mean()) / 2

    goals = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(goals, home_xg), poisson.pmf(goals, away_xg))
    p1 = np.tril(m, -1).sum()
    px = np.trace(m)
    p2 = np.triu(m, 1).sum()
    t = p1 + px + p2
    return home_xg, away_xg, p1 / t, px / t, p2 / t, m / t


st.set_page_config(page_title="Πρόβλεψη Αγώνα", page_icon="⚽")
st.title("⚽ Πρόβλεψη Αγώνα (xG & Poisson)")

with st.sidebar:
    n = st.slider("Τελευταία παιχνίδια", 3, 15, 10)
    same_venue = st.checkbox(
        "Μόνο εντός έδρας (γηπεδούχος) / εκτός έδρας (φιλοξενούμενος)", value=False
    )

league_name = st.selectbox("Πρωτάθλημα", list(LEAGUES))

try:
    df = load_matches(LEAGUES[league_name])
except Exception as e:
    st.error(f"Αποτυχία φόρτωσης δεδομένων: {e}")
    st.stop()

cur = df[df.season == current_season()]
src = cur if not cur.empty else df
teams = sorted(set(src.home) | set(src.away))

c1, c2 = st.columns(2)
home_team = c1.selectbox("Γηπεδούχος", teams, index=0)
away_team = c2.selectbox("Φιλοξενούμενος", teams, index=1)

if st.button("Πρόβλεψη", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Διάλεξε διαφορετικές ομάδες.")
        st.stop()

    hg = team_games(df, home_team, n, same_venue, True)
    ag = team_games(df, away_team, n, same_venue, False)
    if hg.empty or ag.empty:
        st.error("Δεν βρέθηκαν αρκετά παιχνίδια για μία από τις ομάδες.")
        st.stop()

    hxg, axg, p1, px, p2, m = predict(hg, ag)

    st.subheader(f"{home_team} vs {away_team}")
    st.markdown("#### Εκτιμώμενα xG")
    a, b = st.columns(2)
    a.metric(home_team, f"{hxg:.2f}")
    b.metric(away_team, f"{axg:.2f}")

    st.markdown("#### Πιθανότητες (1X2) & δίκαιες αποδόσεις")
    out = pd.DataFrame(
        {
            "Σημείο": ["1", "X", "2"],
            "Αποτέλεσμα": [f"Νίκη {home_team}", "Ισοπαλία", f"Νίκη {away_team}"],
            "Πιθανότητα": [f"{p * 100:.1f}%" for p in (p1, px, p2)],
            "Δίκαιη απόδοση": [f"{1 / max(p, 1e-9):.2f}" for p in (p1, px, p2)],
        }
    )
    st.dataframe(out, hide_index=True, use_container_width=True)

    top = sorted(
        ((i, j, m[i, j]) for i in range(6) for j in range(6)),
        key=lambda x: x[2],
        reverse=True,
    )[:5]
    with st.expander("Πιο πιθανά ακριβή σκορ"):
        st.table(
            pd.DataFrame(
                {
                    "Σκορ": [f"{i}-{j}" for i, j, _ in top],
                    "Πιθανότητα": [f"{p * 100:.1f}%" for _, _, p in top],
                }
            )
        )

    with st.expander(f"Τελευταία παιχνίδια: {home_team}"):
        st.dataframe(hg, hide_index=True, use_container_width=True)
    with st.expander(f"Τελευταία παιχνίδια: {away_team}"):
        st.dataframe(ag, hide_index=True, use_container_width=True)
