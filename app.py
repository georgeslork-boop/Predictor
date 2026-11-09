import io
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson

LEAGUES = {
    "Premier League": ("EPL", "E0"),
    "La Liga": ("La_liga", "SP1"),
    "Serie A": ("Serie_A", "I1"),
    "Bundesliga": ("Bundesliga", "D1"),
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


# ------------------------------------------------------------ Understat ----
def fetch_understat(league, season):
    page = f"https://understat.com/league/{league}/{season}"
    problems = []
    try:
        r = requests.get(
            f"https://understat.com/getLeagueData/{league}/{season}",
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "User-Agent": UA,
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": page,
            },
            timeout=20,
        )
        try:
            return r.json()["dates"]
        except Exception:
            problems.append(f"endpoint status={r.status_code}")
    except Exception as e:
        problems.append(f"endpoint error={e}")
    try:
        r = requests.get(page, headers={"User-Agent": UA}, timeout=20)
        m = re.search(r"datesData\s*=\s*JSON\.parse\('(.*?)'\)", r.text)
        if m:
            raw = m.group(1).encode("ascii", "backslashreplace").decode("unicode_escape")
            return json.loads(raw)
        problems.append(f"html status={r.status_code}, no data found")
    except Exception as e:
        problems.append(f"html error={e}")
    raise RuntimeError(" / ".join(problems))


def rows_from_understat(league, season, notes):
    rows = []
    try:
        matches = fetch_understat(league, season)
    except Exception as e:
        notes.append(f"Understat {season}: {e}")
        return rows
    for r in matches:
        if not r.get("isResult"):
            continue
        rows.append(
            {
                "date": r["datetime"],
                "season": season,
                "home": r["h"]["title"],
                "away": r["a"]["title"],
                "hxg": float(r["xG"]["h"]),
                "axg": float(r["xG"]["a"]),
                "hg": int(r["goals"]["h"]),
                "ag": int(r["goals"]["a"]),
            }
        )
    return rows


# ------------------------------------------------------- football-data ----
def shots_to_xg(shots, on_target):
    """Χονδρική εκτίμηση xG από σουτ (δεν είναι πραγματικό xG)."""
    return 0.22 * on_target + 0.04 * (shots - on_target)


def rows_from_footballdata(code, season, notes):
    rows = []
    tag = f"{season % 100:02d}{(season + 1) % 100:02d}"
    url = f"https://www.football-data.co.uk/mmz4281/{tag}/{code}.csv"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        r.raise_for_status()
        d = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
        need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST"]
        d = d.dropna(subset=need)
    except Exception as e:
        notes.append(f"football-data {season}: {e}")
        return rows
    for _, x in d.iterrows():
        rows.append(
            {
                "date": pd.to_datetime(x["Date"], dayfirst=True, errors="coerce"),
                "season": season,
                "home": x["HomeTeam"],
                "away": x["AwayTeam"],
                "hxg": shots_to_xg(x["HS"], x["HST"]),
                "axg": shots_to_xg(x["AS"], x["AST"]),
                "hg": int(x["FTHG"]),
                "ag": int(x["FTAG"]),
            }
        )
    return rows


@st.cache_data(ttl=3600, show_spinner="Φόρτωση δεδομένων...")
def load_matches(understat_code, fd_code):
    season = current_season()
    notes = []
    rows = []
    for s in (season, season - 1):
        rows += rows_from_understat(understat_code, s, notes)
    source = "understat"
    if not rows:
        source = "footballdata"
        for s in (season, season - 1):
            rows += rows_from_footballdata(fd_code, s, notes)
    if not rows:
        raise RuntimeError("Καμία πηγή δεν απάντησε. " + " | ".join(notes))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date")
    return df, source, notes


# --------------------------------------------------------------- Μοντέλο ----
def team_games(df, team, n, same_venue, is_home):
    d = df[df.home == team]
    h = pd.DataFrame(
        {
            "Ημερομηνία": d.date.dt.date,
            "Αντίπαλος": d.away,
            "Έδρα": "Εντός",
            "xG υπέρ": d.hxg.round(2),
            "xG κατά": d.axg.round(2),
            "Γκολ": d.hg.astype(str) + "-" + d.ag.astype(str),
        }
    )
    d = df[df.away == team]
    a = pd.DataFrame(
        {
            "Ημερομηνία": d.date.dt.date,
            "Αντίπαλος": d.home,
            "Έδρα": "Εκτός",
            "xG υπέρ": d.axg.round(2),
            "xG κατά": d.hxg.round(2),
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


# ------------------------------------------------------------------- UI ----
st.set_page_config(page_title="Πρόβλεψη Αγώνα", page_icon="⚽")
st.title("⚽ Πρόβλεψη Αγώνα (xG & Poisson)")

with st.sidebar:
    n = st.slider("Τελευταία παιχνίδια", 3, 15, 10)
    same_venue = st.checkbox(
        "Μόνο εντός έδρας (γηπεδούχος) / εκτός έδρας (φιλοξενούμενος)", value=False
    )

league_name = st.selectbox("Πρωτάθλημα", list(LEAGUES))
u_code, fd_code = LEAGUES[league_name]

try:
    df, source, notes = load_matches(u_code, fd_code)
except Exception as e:
    st.error(f"Αποτυχία φόρτωσης δεδομένων: {e}")
    st.stop()

if source == "footballdata":
    st.warning(
        "Η Understat δεν απάντησε. Τα xG είναι ΕΚΤΙΜΗΣΗ από τα σουτ "
        "(football-data.co.uk), όχι πραγματικό xG."
    )
    with st.expander("Τεχνικές λεπτομέρειες"):
        st.write(notes)
else:
    st.caption("Πηγή δεδομένων: Understat (πραγματικά xG)")

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
