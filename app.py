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
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df, source, notes


# --------------------------------------------------------------- Μοντέλο ----
def last_games(df, team, n):
    """Τα τελευταία n παιχνίδια της ομάδας (εντός και εκτός), νεότερα πρώτα."""
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


def league_constants(prior):
    """Μέσο xG ανά ομάδα/παιχνίδι και πλεονέκτημα έδρας από τα δεδομένα."""
    mh, ma = prior.hxg.mean(), prior.axg.mean()
    base = (mh + ma) / 2
    return base, mh / base, ma / base


def strengths(games, base, half_life, shrink_k):
    """Δύναμη επίθεσης/άμυνας σε σχέση με τον μέσο όρο (1.0 = μέσος όρος)."""
    n = len(games)
    w = 0.5 ** (np.arange(n) / half_life)  # πιο πρόσφατα = μεγαλύτερο βάρος
    att = np.average(games["xG υπέρ"], weights=w) / base
    dfn = np.average(games["xG κατά"], weights=w) / base
    # Συρρίκνωση προς το 1.0 όταν το δείγμα είναι μικρό
    att = (n * att + shrink_k) / (n + shrink_k)
    dfn = (n * dfn + shrink_k) / (n + shrink_k)
    return att, dfn


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


def predict(prior, home, away, n, half_life, shrink_k, rho):
    hg = last_games(prior, home, n)
    ag = last_games(prior, away, n)
    if len(hg) < 3 or len(ag) < 3 or prior.empty:
        return None
    base, hfa, afa = league_constants(prior)
    ah, dh = strengths(hg, base, half_life, shrink_k)
    aa, da = strengths(ag, base, half_life, shrink_k)
    lam = base * ah * da * hfa
    mu = base * aa * dh * afa
    m = score_matrix(lam, mu, rho)
    return {
        "home_xg": lam,
        "away_xg": mu,
        "p1": np.tril(m, -1).sum(),
        "px": np.trace(m),
        "p2": np.triu(m, 1).sum(),
        "m": m,
        "hg": hg,
        "ag": ag,
        "ah": ah, "dh": dh, "aa": aa, "da": da,
        "base": base, "hfa": hfa, "afa": afa,
    }


# -------------------------------------------------------------- Backtest ----
def metrics(probs, ys):
    probs, ys = np.array(probs), np.array(ys)
    onehot = np.eye(3)[ys]
    acc = (probs.argmax(1) == ys).mean()
    brier = ((probs - onehot) ** 2).sum(1).mean()
    logloss = -np.log(np.clip(probs[np.arange(len(ys)), ys], 1e-9, 1)).mean()
    return acc, brier, logloss


def run_backtest(df, n_test, n, half_life, shrink_k, rho):
    model_p, base_p, ys = [], [], []
    for _, r in df.tail(n_test).iterrows():
        prior = df[df["date"] < r["date"]]
        res = predict(prior, r["home"], r["away"], n, half_life, shrink_k, rho)
        if res is None:
            continue
        model_p.append([res["p1"], res["px"], res["p2"]])
        base_p.append(
            [
                (prior.hg > prior.ag).mean(),
                (prior.hg == prior.ag).mean(),
                (prior.hg < prior.ag).mean(),
            ]
        )
        ys.append(0 if r["hg"] > r["ag"] else (1 if r["hg"] == r["ag"] else 2))
    return np.array(model_p), np.array(base_p), np.array(ys)


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

    res = predict(df, home_team, away_team, n, half_life, shrink_k, rho)
    if res is None:
        st.error("Δεν βρέθηκαν αρκετά παιχνίδια για μία από τις ομάδες.")
        st.stop()

    p1, px, p2 = res["p1"], res["px"], res["p2"]

    st.subheader(f"{home_team} vs {away_team}")
    st.markdown("#### Εκτιμώμενα xG")
    a, b = st.columns(2)
    a.metric(home_team, f"{res['home_xg']:.2f}")
    b.metric(away_team, f"{res['away_xg']:.2f}")

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
            f"Μέσο xG ανά ομάδα/παιχνίδι στο πρωτάθλημα: **{res['base']:.2f}**  \n"
            f"Συντελεστής γηπεδούχου: **{res['hfa']:.2f}**, "
            f"φιλοξενούμενου: **{res['afa']:.2f}**"
        )
        st.table(
            pd.DataFrame(
                {
                    "Ομάδα": [home_team, away_team],
                    "Επίθεση (1.00 = μέσος όρος)": [f"{res['ah']:.2f}", f"{res['aa']:.2f}"],
                    "Άμυνα (κάτω από 1 = καλή)": [f"{res['dh']:.2f}", f"{res['da']:.2f}"],
                }
            )
        )

    for team, g in ((home_team, res["hg"]), (away_team, res["ag"])):
        with st.expander(f"Τελευταία παιχνίδια: {team}"):
            show = g.copy()
            show["Ημερομηνία"] = show["Ημερομηνία"].dt.date
            show["xG υπέρ"] = show["xG υπέρ"].round(2)
            show["xG κατά"] = show["xG κατά"].round(2)
            st.dataframe(show, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Backtest")
st.caption(
    "Τρέχει το μοντέλο σε παλιά παιχνίδια χρησιμοποιώντας μόνο δεδομένα που "
    "υπήρχαν πριν από κάθε αγώνα και συγκρίνει με το αποτέλεσμα."
)
n_test = st.slider("Πόσα πρόσφατα παιχνίδια να ελεγχθούν", 30, 300, 100)

if st.button("Τρέξε Backtest", use_container_width=True):
    with st.spinner("Υπολογισμός..."):
        mp, bp, ys = run_backtest(df, n_test, n, half_life, shrink_k, rho)
    if len(ys) < 10:
        st.error("Πολύ λίγα παιχνίδια για αξιόπιστο έλεγχο.")
        st.stop()

    ma, mb, ml = metrics(mp, ys)
    ba, bb, bl = metrics(bp, ys)
    st.write(f"Ελέγχθηκαν **{len(ys)}** παιχνίδια.")
    st.table(
        pd.DataFrame(
            {
                "Μέτρο": ["Σωστό αποτέλεσμα", "Brier (χαμηλότερο = καλύτερο)", "Log loss (χαμηλότερο = καλύτερο)"],
                "Μοντέλο": [f"{ma * 100:.1f}%", f"{mb:.3f}", f"{ml:.3f}"],
                "Απλή βάση*": [f"{ba * 100:.1f}%", f"{bb:.3f}", f"{bl:.3f}"],
            }
        )
    )
    st.caption(
        "*Η απλή βάση προβλέπει πάντα τα ποσοστά νίκης/ισοπαλίας/ήττας του "
        "πρωταθλήματος, χωρίς να κοιτάει τις ομάδες."
    )
    if mb < bb:
        st.success("Το μοντέλο πετυχαίνει καλύτερα από την απλή βάση.")
    else:
        st.warning("Το μοντέλο δεν ξεπερνά την απλή βάση. Δοκίμασε άλλες ρυθμίσεις.")

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
