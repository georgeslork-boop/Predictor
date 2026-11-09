import io
import json
import re
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import nbinom, poisson

LEAGUES = {
    "Premier League": ("EPL", "E0"),
    "La Liga": ("La_liga", "SP1"),
    "Serie A": ("Serie_A", "I1"),
    "Bundesliga": ("Bundesliga", "D1"),
}
MAX_GOALS = 10
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
ODDS_SETS = [
    ("AvgH", "AvgD", "AvgA"),
    ("BbAvH", "BbAvD", "BbAvA"),
    ("B365H", "B365D", "B365A"),
]


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


def num(x, key):
    return float(pd.to_numeric(x.get(key, np.nan), errors="coerce"))


def get_odds(x):
    for hk, dk, ak in ODDS_SETS:
        vals = [num(x, hk), num(x, dk), num(x, ak)]
        if all(pd.notna(v) and v > 1 for v in vals):
            return vals
    return [np.nan, np.nan, np.nan]


def cards_of(x, yk, rk):
    y = num(x, yk)
    if pd.isna(y):
        return np.nan
    r = num(x, rk)
    return y + (0 if pd.isna(r) else r)


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
        oh, od, oa = get_odds(x)
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
                "hc": num(x, "HC"),
                "ac": num(x, "AC"),
                "hcards": cards_of(x, "HY", "HR"),
                "acards": cards_of(x, "AY", "AR"),
                "oh": oh,
                "od": od,
                "oa": oa,
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
    for c in ["hc", "ac", "hcards", "acards", "oh", "od", "oa"]:
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, source, notes


# --------------------------------------------------------------- Μοντέλο ----
def last_games(df, team, n):
    """Για εμφάνιση: τα τελευταία n παιχνίδια της ομάδας."""
    d = df[df.home == team]
    h = pd.DataFrame(
        {
            "Ημερομηνία": d.date,
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
            "Ημερομηνία": d.date,
            "Αντίπαλος": d.home,
            "Έδρα": "Εκτός",
            "xG υπέρ": d.axg,
            "xG κατά": d.hxg,
            "Γκολ": d.hg.astype(str) + "-" + d.ag.astype(str),
        }
    )
    g = pd.concat([h, a]).sort_values("Ημερομηνία", ascending=False)
    return g.head(n)


def team_for_against(df, team, n, hcol, acol):
    h = df[df.home == team][["date", hcol, acol]]
    h = h.rename(columns={hcol: "f", acol: "a"})
    a = df[df.away == team][["date", acol, hcol]]
    a = a.rename(columns={acol: "f", hcol: "a"})
    g = pd.concat([h, a]).dropna().sort_values("date", ascending=False)
    return g.head(n)


def strengths(g, base, half_life, shrink_k):
    """Δύναμη επίθεσης/άμυνας σε σχέση με τον μέσο όρο (1.0 = μέσος όρος)."""
    n = len(g)
    w = 0.5 ** (np.arange(n) / half_life)  # πιο πρόσφατα = μεγαλύτερο βάρος
    att = np.average(g["f"], weights=w) / base
    dfn = np.average(g["a"], weights=w) / base
    att = (n * att + shrink_k) / (n + shrink_k)
    dfn = (n * dfn + shrink_k) / (n + shrink_k)
    return att, dfn


def expected_pair(prior, home, away, n, hcol, acol, half_life, shrink_k):
    prior = prior.dropna(subset=[hcol, acol])
    if prior.empty:
        return None
    gh = team_for_against(prior, home, n, hcol, acol)
    ga = team_for_against(prior, away, n, hcol, acol)
    if len(gh) < 3 or len(ga) < 3:
        return None
    mh, ma = prior[hcol].mean(), prior[acol].mean()
    base = (mh + ma) / 2
    if base <= 0:
        return None
    hfa, afa = mh / base, ma / base
    ah, dh = strengths(gh, base, half_life, shrink_k)
    aa, da = strengths(ga, base, half_life, shrink_k)
    return {
        "lam": base * ah * da * hfa,
        "mu": base * aa * dh * afa,
        "ah": ah, "dh": dh, "aa": aa, "da": da,
        "base": base, "hfa": hfa, "afa": afa,
    }


def score_matrix(lam, mu, rho):
    """Πίνακας σκορ Poisson με διόρθωση Dixon-Coles στα χαμηλά σκορ."""
    g = np.arange(MAX_GOALS + 1)
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


def predict_goals(prior, home, away, n, half_life, shrink_k, rho, display=False):
    e = expected_pair(prior, home, away, n, "hxg", "axg", half_life, shrink_k)
    if e is None:
        return None
    m = score_matrix(e["lam"], e["mu"], rho)
    out = {
        "home_xg": e["lam"],
        "away_xg": e["mu"],
        "p1": np.tril(m, -1).sum(),
        "px": np.trace(m),
        "p2": np.triu(m, 1).sum(),
        "m": m,
        "e": e,
    }
    if display:
        out["hg"] = last_games(prior, home, n)
        out["ag"] = last_games(prior, away, n)
    return out


def dispersion(prior, hcol, acol):
    t = (prior[hcol] + prior[acol]).dropna()
    if len(t) < 30 or t.mean() <= 0:
        return None
    return t.var() / t.mean()


def predict_total_stat(prior, home, away, n, hcol, acol, half_life, shrink_k):
    e = expected_pair(prior, home, away, n, hcol, acol, half_life, shrink_k)
    if e is None:
        return None
    return {
        "mean": e["lam"] + e["mu"],
        "home": e["lam"],
        "away": e["mu"],
        "ratio": dispersion(prior, hcol, acol),
    }


def prob_over(mean, ratio, line):
    k = int(np.floor(line))
    if ratio is None or ratio <= 1.02:
        return float(1 - poisson.cdf(k, mean))
    p = 1 / ratio
    r = mean / (ratio - 1)
    return float(1 - nbinom.cdf(k, r, p))


# --------------------------------------------------------------- Αγορές ----
def centered_lines(mean, k=3):
    c = np.floor(mean) + 0.5
    return [c + d for d in range(-k, k + 1) if c + d > 0]


def goal_market_rows(res, home, away):
    m = res["m"]
    tot = np.zeros(2 * MAX_GOALS + 1)
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            tot[i + j] += m[i, j]
    p1, px, p2 = res["p1"], res["px"], res["p2"]

    result_rows = [
        (f"Νίκη {home} (1)", p1),
        ("Ισοπαλία (Χ)", px),
        (f"Νίκη {away} (2)", p2),
        (f"{home} ή ισοπαλία (1Χ)", p1 + px),
        (f"Ισοπαλία ή {away} (Χ2)", px + p2),
        (f"{home} ή {away} (12)", p1 + p2),
    ]

    goal_rows = []
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        o = tot[int(line) + 1:].sum()
        goal_rows.append((f"Πάνω από {line} γκολ", o))
        goal_rows.append((f"Κάτω από {line} γκολ", 1 - o))
    for t in range(0, 5):
        goal_rows.append((f"Ακριβώς {t} γκολ", tot[t]))
    btts = 1 - m[0, :].sum() - m[:, 0].sum() + m[0, 0]
    goal_rows.append(("Και οι δύο σκοράρουν: ΝΑΙ", btts))
    goal_rows.append(("Και οι δύο σκοράρουν: ΟΧΙ", 1 - btts))
    goal_rows.append((f"{home} σκοράρει (πάνω από 0.5)", 1 - m[0, :].sum()))
    goal_rows.append((f"{home} πάνω από 1.5 γκολ", m[2:, :].sum()))
    goal_rows.append((f"{away} σκοράρει (πάνω από 0.5)", 1 - m[:, 0].sum()))
    goal_rows.append((f"{away} πάνω από 1.5 γκολ", m[:, 2:].sum()))
    return result_rows, goal_rows


def stat_market_rows(stat, name):
    rows = []
    for line in centered_lines(stat["mean"]):
        o = prob_over(stat["mean"], stat["ratio"], line)
        rows.append((f"Πάνω από {line} {name}", o))
        rows.append((f"Κάτω από {line} {name}", 1 - o))
    return rows


def render_market_tab(rows, cat, key):
    df = pd.DataFrame(
        {
            "Επιλογή": [r[0] for r in rows],
            "Πιθανότητα %": [round(r[1] * 100, 1) for r in rows],
            "Δίκαιη απόδοση": [round(1 / max(r[1], 1e-9), 2) for r in rows],
            "Απόδοση στοιχηματικού": [np.nan] * len(rows),
        }
    )
    edited = st.data_editor(
        df,
        key=key,
        hide_index=True,
        use_container_width=True,
        disabled=["Επιλογή", "Πιθανότητα %", "Δίκαιη απόδοση"],
        column_config={
            "Απόδοση στοιχηματικού": st.column_config.NumberColumn(
                min_value=1.01, max_value=1000.0, step=0.01, format="%.2f"
            )
        },
    )
    out = edited.copy()
    out["p"] = [r[1] for r in rows]
    out["Αγορά"] = cat
    return out


# -------------------------------------------------------------- Backtest ----
def metrics(probs, ys):
    probs, ys = np.array(probs), np.array(ys)
    onehot = np.eye(3)[ys]
    acc = (probs.argmax(1) == ys).mean()
    brier = ((probs - onehot) ** 2).sum(1).mean()
    logloss = -np.log(np.clip(probs[np.arange(len(ys)), ys], 1e-9, 1)).mean()
    return acc, brier, logloss


def run_backtest(df, n_test, n, half_life, shrink_k, rho):
    mp, bp, qp, od, has, ys = [], [], [], [], [], []
    for _, r in df.tail(n_test).iterrows():
        prior = df[df["date"] < r["date"]]
        res = predict_goals(prior, r["home"], r["away"], n, half_life, shrink_k, rho)
        if res is None:
            continue
        mp.append([res["p1"], res["px"], res["p2"]])
        bp.append(
            [
                (prior.hg > prior.ag).mean(),
                (prior.hg == prior.ag).mean(),
                (prior.hg < prior.ag).mean(),
            ]
        )
        ys.append(0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2))
        o = [r["oh"], r["od"], r["oa"]]
        if all(pd.notna(v) and v > 1 for v in o):
            q = 1 / np.array(o, dtype=float)
            qp.append(q / q.sum())
            od.append(o)
            has.append(True)
        else:
            qp.append([np.nan] * 3)
            od.append([np.nan] * 3)
            has.append(False)
    return (
        np.array(mp), np.array(bp), np.array(qp),
        np.array(od, dtype=float), np.array(has, dtype=bool), np.array(ys),
    )


# ------------------------------------------------------------------- UI ----
st.set_page_config(page_title="Πρόβλεψη Αγώνα", page_icon="⚽")
st.title("⚽ Πρόβλεψη Αγώνα (xG & Poisson)")

with st.sidebar:
    st.header("Ρυθμίσεις μοντέλου")
    n = st.slider("Τελευταία παιχνίδια ανά ομάδα", 5, 20, 10)
    half_life = st.slider(
        "Ημι-ζωή βαρών (παιχνίδια)", 2, 20, 6,
        help="Μικρότερο = μετράνε πολύ περισσότερο τα πιο πρόσφατα παιχνίδια.",
    )
    shrink_k = st.slider(
        "Συρρίκνωση προς τον μέσο όρο", 0, 10, 4,
        help="Μεγαλύτερο = λιγότερο επηρεάζεται το μοντέλο από λίγα παιχνίδια.",
    )
    rho = st.slider(
        "Dixon-Coles ρ", -0.30, 0.0, -0.10, 0.01,
        help="Διόρθωση για 0-0, 1-0, 0-1, 1-1. Το 0 την απενεργοποιεί.",
    )
    min_edge = (
        st.slider(
            "Ελάχιστο πλεονέκτημα (edge) για ένδειξη value", 1, 20, 5,
            help="Edge = πιθανότητα μοντέλου × απόδοση − 1.",
        )
        / 100
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
    st.caption(
        "Πηγή δεδομένων: Understat (πραγματικά xG). Κόρνερ, κάρτες και "
        "αποδόσεις αγοράς δεν είναι διαθέσιμα από αυτή την πηγή."
    )

cur = df[df.season == current_season()]
src = cur if not cur.empty else df
teams = sorted(set(src.home) | set(src.away))

c1, c2 = st.columns(2)
home_team = c1.selectbox("Γηπεδούχος", teams, index=0)
away_team = c2.selectbox("Φιλοξενούμενος", teams, index=1)

if "pred_id" not in st.session_state:
    st.session_state["pred_id"] = 0

if st.button("Πρόβλεψη", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Διάλεξε διαφορετικές ομάδες.")
    else:
        goals = predict_goals(
            df, home_team, away_team, n, half_life, shrink_k, rho, display=True
        )
        if goals is None:
            st.error("Δεν βρέθηκαν αρκετά παιχνίδια για μία από τις ομάδες.")
        else:
            st.session_state["pred_id"] += 1
            st.session_state["pred"] = {
                "id": st.session_state["pred_id"],
                "home": home_team,
                "away": away_team,
                "goals": goals,
                "corners": predict_total_stat(
                    df, home_team, away_team, n, "hc", "ac", half_life, shrink_k
                ),
                "cards": predict_total_stat(
                    df, home_team, away_team, n, "hcards", "acards", half_life, shrink_k
                ),
            }

pred = st.session_state.get("pred")
if pred:
    home, away, res = pred["home"], pred["away"], pred["goals"]
    e = res["e"]
    p1, px, p2 = res["p1"], res["px"], res["p2"]

    st.subheader(f"{home} vs {away}")
    st.markdown("#### Εκτιμώμενα xG")
    a, b = st.columns(2)
    a.metric(home, f"{res['home_xg']:.2f}")
    b.metric(away, f"{res['away_xg']:.2f}")

    st.markdown("#### Πιθανότητες (1X2) & δίκαιες αποδόσεις")
    out = pd.DataFrame(
        {
            "Σημείο": ["1", "X", "2"],
            "Αποτέλεσμα": [f"Νίκη {home}", "Ισοπαλία", f"Νίκη {away}"],
            "Πιθανότητα": [f"{p * 100:.1f}%" for p in (p1, px, p2)],
            "Δίκαιη απόδοση": [f"{1 / max(p, 1e-9):.2f}" for p in (p1, px, p2)],
        }
    )
    st.dataframe(out, hide_index=True, use_container_width=True)

    m = res["m"]
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

    with st.expander("Λεπτομέρειες μοντέλου"):
        st.write(
            f"Μέσο xG ανά ομάδα/παιχνίδι στο πρωτάθλημα: **{e['base']:.2f}**  \n"
            f"Συντελεστής γηπεδούχου: **{e['hfa']:.2f}**, "
            f"φιλοξενούμενου: **{e['afa']:.2f}**"
        )
        st.table(
            pd.DataFrame(
                {
                    "Ομάδα": [home, away],
                    "Επίθεση (1.00 = μέσος όρος)": [f"{e['ah']:.2f}", f"{e['aa']:.2f}"],
                    "Άμυνα (κάτω από 1 = καλή)": [f"{e['dh']:.2f}", f"{e['da']:.2f}"],
                }
            )
        )

    for team, g in ((home, res["hg"]), (away, res["ag"])):
        with st.expander(f"Τελευταία παιχνίδια: {team}"):
            show = g.copy()
            show["Ημερομηνία"] = show["Ημερομηνία"].dt.date
            show["xG υπέρ"] = show["xG υπέρ"].round(2)
            show["xG κατά"] = show["xG κατά"].round(2)
            st.dataframe(show, hide_index=True, use_container_width=True)

    # ------------------------------------------------- Αγορές & value ----
    st.divider()
    st.subheader("Αγορές & value")
    st.caption(
        "Γράψε στη στήλη «Απόδοση στοιχηματικού» την απόδοση που βλέπεις στο "
        "site σου. Θα εμφανιστεί παρακάτω αν η επιλογή αξίζει. "
        "(Για «πάνω από 7 κόρνερ» ψάξε το «Πάνω από 7.5».)"
    )

    result_rows, goal_rows = goal_market_rows(res, home, away)
    tabs = st.tabs(["Αποτέλεσμα", "Γκολ", "Κόρνερ", "Κάρτες"])
    edited_all = []
    all_rows = []
    kid = pred["id"]

    with tabs[0]:
        edited_all.append(render_market_tab(result_rows, "Αποτέλεσμα", f"ed_res_{kid}"))
        all_rows += [("Αποτέλεσμα",) + r for r in result_rows]
    with tabs[1]:
        edited_all.append(render_market_tab(goal_rows, "Γκολ", f"ed_goal_{kid}"))
        all_rows += [("Γκολ",) + r for r in goal_rows]
    with tabs[2]:
        if pred["corners"] is None:
            st.info("Δεν υπάρχουν δεδομένα κόρνερ από αυτή την πηγή.")
        else:
            st.write(
                f"Αναμενόμενα κόρνερ: **{pred['corners']['mean']:.1f}** "
                f"({home} {pred['corners']['home']:.1f}, {away} {pred['corners']['away']:.1f})"
            )
            rows = stat_market_rows(pred["corners"], "κόρνερ")
            edited_all.append(render_market_tab(rows, "Κόρνερ", f"ed_cor_{kid}"))
            all_rows += [("Κόρνερ",) + r for r in rows]
    with tabs[3]:
        if pred["cards"] is None:
            st.info("Δεν υπάρχουν δεδομένα καρτών από αυτή την πηγή.")
        else:
            st.write(
                f"Αναμενόμενες κάρτες (κίτρινες + κόκκινες): **{pred['cards']['mean']:.1f}**"
            )
            rows = stat_market_rows(pred["cards"], "κάρτες")
            edited_all.append(render_market_tab(rows, "Κάρτες", f"ed_card_{kid}"))
            all_rows += [("Κάρτες",) + r for r in rows]

    st.markdown("#### Αποτέλεσμα ελέγχου value")
    allv = pd.concat(edited_all)
    allv = allv[allv["Απόδοση στοιχηματικού"].notna()]
    if allv.empty:
        st.info("Γράψε αποδόσεις στους πίνακες πάνω για να δεις ποιες επιλογές αξίζουν.")
    else:
        o = allv["Απόδοση στοιχηματικού"].astype(float).values
        p = allv["p"].astype(float).values
        edge = p * o - 1
        kelly = np.where(o > 1, edge / (o - 1), 0)
        stake = np.where(edge >= min_edge, np.minimum(0.25 * kelly, 0.02), 0)
        vt = pd.DataFrame(
            {
                "Αγορά": allv["Αγορά"].values,
                "Επιλογή": allv["Επιλογή"].values,
                "Πιθανότητα": [f"{x * 100:.1f}%" for x in p],
                "Απόδοση": [f"{x:.2f}" for x in o],
                "Edge": [f"{x * 100:+.1f}%" for x in edge],
                "Ένδειξη": ["✅ Αξίζει" if x >= min_edge else "❌ Όχι" for x in edge],
                "% bankroll": [f"{x * 100:.1f}%" for x in stake],
                "_e": edge,
            }
        ).sort_values("_e", ascending=False).drop(columns="_e")
        st.dataframe(vt, hide_index=True, use_container_width=True)
        st.caption(
            "Edge = πιθανότητα μοντέλου × απόδοση − 1. Το «% bankroll» είναι ένα "
            "τέταρτο του κριτηρίου Kelly με ανώτατο όριο 2%. Είναι ένδειξη, όχι "
            "εγγύηση: αν το μοντέλο υπερτιμά μια επιλογή, το «value» είναι φαντασία."
        )

    with st.expander("Επιλογές με τη μεγαλύτερη πιθανότητα (χωρίς απόδοση)"):
        best = sorted(
            [r for r in all_rows if 0.55 <= r[2] <= 0.95],
            key=lambda r: r[2],
            reverse=True,
        )[:10]
        st.table(
            pd.DataFrame(
                {
                    "Αγορά": [r[0] for r in best],
                    "Επιλογή": [r[1] for r in best],
                    "Πιθανότητα": [f"{r[2] * 100:.1f}%" for r in best],
                    "Δίκαιη απόδοση": [f"{1 / r[2]:.2f}" for r in best],
                }
            )
        )
        st.caption(
            "Μεγάλη πιθανότητα δεν σημαίνει καλό στοίχημα. Αν ο στοιχηματικός "
            "δίνει απόδοση κάτω από τη δίκαιη, δεν αξίζει."
        )

# -------------------------------------------------------------- Backtest ----
st.divider()
st.subheader("Backtest")
st.caption(
    "Τρέχει το μοντέλο σε παλιά παιχνίδια χρησιμοποιώντας μόνο δεδομένα που "
    "υπήρχαν πριν από κάθε αγώνα και συγκρίνει με το αποτέλεσμα, την απλή βάση "
    "και (όπου υπάρχουν) τις αποδόσεις της αγοράς."
)
n_test = st.slider("Πόσα πρόσφατα παιχνίδια να ελεγχθούν", 30, 300, 100)

if st.button("Τρέξε Backtest", use_container_width=True):
    with st.spinner("Υπολογισμός..."):
        mp, bp, qp, od, has, ys = run_backtest(df, n_test, n, half_life, shrink_k, rho)
    if len(ys) < 10:
        st.error("Πολύ λίγα παιχνίδια για αξιόπιστο έλεγχο.")
    else:
        use_market = has.sum() >= 30
        if use_market:
            mp, bp, qp, od, ys = mp[has], bp[has], qp[has], od[has], ys[has]
        ma, mb, ml = metrics(mp, ys)
        ba, bb, bl = metrics(bp, ys)
        table = {
            "Μέτρο": [
                "Σωστό αποτέλεσμα",
                "Brier (χαμηλότερο = καλύτερο)",
                "Log loss (χαμηλότερο = καλύτερο)",
            ],
            "Μοντέλο": [f"{ma * 100:.1f}%", f"{mb:.3f}", f"{ml:.3f}"],
        }
        if use_market:
            qa, qb, ql = metrics(qp, ys)
            table["Αγορά (στοιχηματικοί)"] = [f"{qa * 100:.1f}%", f"{qb:.3f}", f"{ql:.3f}"]
        table["Απλή βάση*"] = [f"{ba * 100:.1f}%", f"{bb:.3f}", f"{bl:.3f}"]

        st.write(f"Ελέγχθηκαν **{len(ys)}** παιχνίδια.")
        st.table(pd.DataFrame(table))
        st.caption(
            "*Η απλή βάση προβλέπει πάντα τα ποσοστά νίκης/ισοπαλίας/ήττας του "
            "πρωταθλήματος, χωρίς να κοιτάει τις ομάδες. Η «αγορά» είναι οι μέσες "
            "αποδόσεις στοιχηματικών, με αφαιρεμένο το περιθώριο."
        )
        if mb < bb:
            st.success("Το μοντέλο πετυχαίνει καλύτερα από την απλή βάση.")
        else:
            st.warning("Το μοντέλο δεν ξεπερνά την απλή βάση. Δοκίμασε άλλες ρυθμίσεις.")
        if use_market:
            if mb < qb:
                st.success("Το μοντέλο πετυχαίνει καλύτερα και από την αγορά.")
            else:
                st.info(
                    "Η αγορά πετυχαίνει καλύτερα από το μοντέλο, όπως συνήθως "
                    "συμβαίνει. Το «value» που βλέπεις στην εφαρμογή είναι τότε "
                    "κυρίως λάθος του μοντέλου και όχι ευκαιρία."
                )

            st.markdown("#### Προσομοίωση value στοιχημάτων")
            bets = wins = 0
            profit = 0.0
            for pr, oo, y in zip(mp, od, ys):
                edges = pr * oo - 1
                k = int(edges.argmax())
                if edges[k] >= min_edge:
                    bets += 1
                    if k == y:
                        wins += 1
                        profit += oo[k] - 1
                    else:
                        profit -= 1
            if bets == 0:
                st.write("Καμία ένδειξη value με αυτό το όριο edge.")
            else:
                st.table(
                    pd.DataFrame(
                        {
                            "Στοιχήματα": [bets],
                            "Επιτυχία": [f"{wins / bets * 100:.1f}%"],
                            "Κέρδος (μονάδες)": [f"{profit:+.1f}"],
                            "ROI": [f"{profit / bets * 100:+.1f}%"],
                        }
                    )
                )
                st.caption(
                    "Στοιχηματίζει 1 μονάδα στο 1X2 με το μεγαλύτερο edge όταν "
                    "ξεπερνά το όριο του sidebar, με μέσες αποδόσεις αγοράς. Με "
                    "λίγες δεκάδες στοιχήματα, ένα θετικό ROI είναι κυρίως τύχη. "
                    "Για σύγκριση, στοιχηματισμός χωρίς κριτήριο χάνει συνήθως "
                    "το περιθώριο του στοιχηματικού."
                )
        else:
            st.info(
                "Δεν υπάρχουν αρκετές αποδόσεις αγοράς σε αυτή την πηγή δεδομένων "
                "για σύγκριση."
            )

        st.markdown("#### Βαθμονόμηση (πόσο ρεαλιστικές είναι οι πιθανότητες)")
        actual = np.eye(3)[ys].mean(0)
        st.table(
            pd.DataFrame(
                {
                    "Αποτέλεσμα": ["1", "X", "2"],
                    "Μέση πρόβλεψη": [f"{v * 100:.1f}%" for v in mp.mean(0)],
                    "Πραγματική συχνότητα": [f"{v * 100:.1f}%" for v in actual],
                }
            )
        )
