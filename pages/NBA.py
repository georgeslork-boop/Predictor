from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import norm

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


# ------------------------------------------------------------- Δεδομένα ----
def fetch_range(start, end):
    dates = f"{start:%Y%m%d}" if start == end else f"{start:%Y%m%d}-{end:%Y%m%d}"
    r = requests.get(
        BASE_URL,
        params={"dates": dates, "limit": 200},
        headers={"User-Agent": UA},
        timeout=25,
    )
    r.raise_for_status()
    return r.json().get("events", [])


def fetch_chunk(span):
    """Ζητά ένα διάστημα. Αν αποτύχει, δοκιμάζει ημέρα-ημέρα."""
    start, end = span
    try:
        return fetch_range(start, end), None
    except Exception as e:
        err = str(e)
    events, fails = [], 0
    d = start
    while d <= end:
        try:
            events += fetch_range(d, d)
        except Exception:
            fails += 1
        d += timedelta(days=1)
    return events, (f"{start}: {err}" if fails else None)


@st.cache_data(ttl=1800, show_spinner="Φόρτωση αγώνων NBA (η πρώτη φορά κρατά λίγο)...")
def load_games():
    today = date.today()
    season = today.year if today.month >= 10 else today.year - 1
    cur = date(season - 1, 10, 1)  # από την προηγούμενη σεζόν

    spans = []
    while cur <= today:
        end = min(cur + timedelta(days=6), today)
        if not (cur.month in (7, 8, 9) and end.month in (7, 8, 9)):
            spans.append((cur, end))
        cur = end + timedelta(days=1)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_chunk, spans))

    rows, notes, seen = [], [], set()
    for events, note in results:
        if note:
            notes.append(note)
        for ev in events:
            try:
                if ev["id"] in seen or not ev["status"]["type"]["completed"]:
                    continue
                if ev.get("season", {}).get("type") not in (2, 3, 5, None):
                    continue
                comp = ev["competitions"][0]
                h = next(c for c in comp["competitors"] if c["homeAway"] == "home")
                a = next(c for c in comp["competitors"] if c["homeAway"] == "away")
                hp, ap = int(float(h["score"])), int(float(a["score"]))
                if hp == 0 and ap == 0:
                    continue
                dt = (
                    pd.to_datetime(ev["date"], utc=True)
                    .tz_convert("America/New_York")
                    .tz_localize(None)
                )
                seen.add(ev["id"])
                rows.append(
                    {
                        "date": dt,
                        "day": dt.normalize(),
                        "home": h["team"]["displayName"],
                        "away": a["team"]["displayName"],
                        "hp": hp,
                        "ap": ap,
                    }
                )
            except Exception:
                continue
    if not rows:
        raise RuntimeError("Δεν επιστράφηκαν αγώνες. " + " | ".join(notes[:3]))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    counts = pd.concat([df.home, df.away]).value_counts()
    good = counts[counts >= 10].index
    df = df[df.home.isin(good) & df.away.isin(good)].reset_index(drop=True)
    return df, notes


# --------------------------------------------------------------- Μοντέλο ----
def team_games(prior, team, n):
    h = prior[prior.home == team][["date", "hp", "ap"]].rename(
        columns={"hp": "f", "ap": "a"}
    )
    a = prior[prior.away == team][["date", "ap", "hp"]].rename(
        columns={"ap": "f", "hp": "a"}
    )
    return pd.concat([h, a]).sort_values("date", ascending=False).head(n)


def strengths(g, base, half_life, shrink_k):
    n = len(g)
    w = 0.5 ** (np.arange(n) / half_life)
    att = np.average(g["f"], weights=w) / base
    dfn = np.average(g["a"], weights=w) / base
    att = (n * att + shrink_k) / (n + shrink_k)
    dfn = (n * dfn + shrink_k) / (n + shrink_k)
    return att, dfn


def rest_days(prior, team, day):
    g = prior[(prior.home == team) | (prior.away == team)]
    if g.empty:
        return None
    return (day - g.day.max()).days


def predict(prior, home, away, day, p):
    gh = team_games(prior, home, p["n"])
    ga = team_games(prior, away, p["n"])
    if len(gh) < 5 or len(ga) < 5:
        return None
    mh, ma = prior.hp.mean(), prior.ap.mean()
    base = (mh + ma) / 2
    hfa, afa = mh / base, ma / base
    ah, dh = strengths(gh, base, p["half_life"], p["shrink_k"])
    aa, da = strengths(ga, base, p["half_life"], p["shrink_k"])
    lam = base * ah * da * hfa
    mu = base * aa * dh * afa

    rh, ra = rest_days(prior, home, day), rest_days(prior, away, day)
    b2b_h = rh is not None and rh <= 1
    b2b_a = ra is not None and ra <= 1
    adj = (-p["b2b_pen"] if b2b_h else 0) + (p["b2b_pen"] if b2b_a else 0)
    lam += adj / 2
    mu -= adj / 2

    M, T = lam - mu, lam + mu
    p_home = 1 - norm.cdf((0.5 - M) / p["sd_m"])
    return {
        "lam": lam, "mu": mu, "M": M, "T": T, "p_home": p_home,
        "b2b_h": b2b_h, "b2b_a": b2b_a,
        "base": base, "hfa": hfa, "afa": afa,
        "ah": ah, "dh": dh, "aa": aa, "da": da,
    }


# --------------------------------------------------------------- Αγορές ----
def half_lines(x, k):
    c = np.floor(x) + 0.5
    return [float(c + d) for d in range(-k, k + 1)]


def market_rows(res, home, away, p):
    M, T, lam, mu = res["M"], res["T"], res["lam"], res["mu"]
    sd_m, sd_t = p["sd_m"], p["sd_t"]
    sd_team = np.sqrt((sd_t**2 + sd_m**2) / 4)

    ph = res["p_home"]
    ml = [(f"Νίκη {home}", ph), (f"Νίκη {away}", 1 - ph)]

    spread = []
    for L in half_lines(M, 4):
        h = -L
        pc = 1 - norm.cdf((L - M) / sd_m)
        spread.append((f"{home} {h:+.1f}", pc))
        spread.append((f"{away} {-h:+.1f}", 1 - pc))

    totals = []
    for L in half_lines(T, 4):
        po = 1 - norm.cdf((L - T) / sd_t)
        totals.append((f"Σύνολο πάνω από {L}", po))
        totals.append((f"Σύνολο κάτω από {L}", 1 - po))

    team_tot = []
    for name, mean in ((home, lam), (away, mu)):
        for L in half_lines(mean, 2):
            po = 1 - norm.cdf((L - mean) / sd_team)
            team_tot.append((f"{name} πάνω από {L}", po))
            team_tot.append((f"{name} κάτω από {L}", 1 - po))
    return ml, spread, totals, team_tot


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
def bin_metrics(p, y):
    p, y = np.array(p, dtype=float), np.array(y, dtype=float)
    acc = ((p > 0.5) == (y == 1)).mean()
    brier = ((p - y) ** 2).mean()
    ll = -np.mean(
        y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))
    )
    return acc, brier, ll


def run_backtest(df, n_test, p):
    recs = []
    for _, r in df.tail(n_test).iterrows():
        prior = df[df.date < r["date"]]
        res = predict(prior, r["home"], r["away"], r["day"], p)
        if res is None:
            continue
        recs.append(
            {
                "p": res["p_home"],
                "y": int(r["hp"] > r["ap"]),
                "M": res["M"],
                "T": res["T"],
                "margin": r["hp"] - r["ap"],
                "total": r["hp"] + r["ap"],
                "bp": (prior.hp > prior.ap).mean(),
                "bM": (prior.hp - prior.ap).mean(),
                "bT": (prior.hp + prior.ap).mean(),
            }
        )
    return pd.DataFrame(recs)


# ------------------------------------------------------------------- UI ----
st.set_page_config(page_title="NBA Πρόβλεψη", page_icon="🏀")
st.title("🏀 NBA: Πρόβλεψη Αγώνα")

with st.sidebar:
    st.header("Ρυθμίσεις μοντέλου")
    n = st.slider("Τελευταία παιχνίδια ανά ομάδα", 5, 40, 20)
    half_life = st.slider(
        "Ημι-ζωή βαρών (παιχνίδια)", 3, 40, 12,
        help="Μικρότερο = μετράνε περισσότερο τα πιο πρόσφατα παιχνίδια.",
    )
    shrink_k = st.slider(
        "Συρρίκνωση προς τον μέσο όρο", 0, 10, 4,
        help="Μεγαλύτερο = λιγότερο επηρεάζεται το μοντέλο από λίγα παιχνίδια.",
    )
    b2b_pen = st.slider(
        "Ποινή back-to-back (πόντοι)", 0.0, 4.0, 1.0, 0.1,
        help="Πόσους πόντους χάνει μια ομάδα που παίζει δεύτερο βράδυ στη σειρά.",
    )
    sd_m = st.slider(
        "Τυπική απόκλιση διαφοράς (πόντοι)", 8.0, 16.0, 12.0, 0.5,
        help="Το Backtest προτείνει τιμή από τα δεδομένα.",
    )
    sd_t = st.slider(
        "Τυπική απόκλιση συνόλου (πόντοι)", 12.0, 24.0, 18.0, 0.5,
        help="Το Backtest προτείνει τιμή από τα δεδομένα.",
    )
    min_edge = (
        st.slider("Ελάχιστο edge για ένδειξη value", 1, 20, 5,
                  help="Edge = πιθανότητα μοντέλου × απόδοση − 1.")
        / 100
    )

params = {
    "n": n, "half_life": half_life, "shrink_k": shrink_k,
    "b2b_pen": b2b_pen, "sd_m": sd_m, "sd_t": sd_t,
}

try:
    df, notes = load_games()
except Exception as e:
    st.error(f"Αποτυχία φόρτωσης δεδομένων: {e}")
    st.stop()

st.caption(
    f"Πηγή: ESPN. {len(df)} αγώνες (προηγούμενη και τρέχουσα σεζόν). "
    "Το μοντέλο ΔΕΝ γνωρίζει τραυματισμούς ή ποιοι παίκτες απουσιάζουν."
)
if notes:
    with st.expander(f"Προειδοποίηση: {len(notes)} διαστήματα δεν φορτώθηκαν πλήρως"):
        st.write(notes[:5])

teams = sorted(set(df.home) | set(df.away))
c1, c2 = st.columns(2)
home_team = c1.selectbox("Γηπεδούχος", teams, index=0)
away_team = c2.selectbox("Φιλοξενούμενος", teams, index=1)
eastern_today = pd.Timestamp.now(tz="America/New_York").date()
game_date = st.date_input("Ημερομηνία αγώνα", value=eastern_today)

if "nba_pid" not in st.session_state:
    st.session_state["nba_pid"] = 0

if st.button("Πρόβλεψη", type="primary", use_container_width=True):
    if home_team == away_team:
        st.error("Διάλεξε διαφορετικές ομάδες.")
    else:
        res = predict(df, home_team, away_team, pd.Timestamp(game_date), params)
        if res is None:
            st.error("Δεν βρέθηκαν αρκετά παιχνίδια για μία από τις ομάδες.")
        else:
            st.session_state["nba_pid"] += 1
            st.session_state["nba_pred"] = {
                "id": st.session_state["nba_pid"],
                "home": home_team,
                "away": away_team,
                "res": res,
            }

pred = st.session_state.get("nba_pred")
if pred:
    home, away, res = pred["home"], pred["away"], pred["res"]
    ph = res["p_home"]

    st.subheader(f"{home} vs {away}")
    st.markdown("#### Αναμενόμενο σκορ")
    a, b = st.columns(2)
    a.metric(home, f"{res['lam']:.1f}")
    b.metric(away, f"{res['mu']:.1f}")
    st.write(
        f"Αναμενόμενη διαφορά: **{home} {res['M']:+.1f}** · "
        f"Αναμενόμενο σύνολο: **{res['T']:.1f}** πόντοι"
    )
    if res["b2b_h"] or res["b2b_a"]:
        who = [t for t, f in ((home, res["b2b_h"]), (away, res["b2b_a"])) if f]
        st.info("Back-to-back: " + ", ".join(who))

    st.markdown("#### Νικητής & δίκαιες αποδόσεις")
    st.dataframe(
        pd.DataFrame(
            {
                "Ομάδα": [home, away],
                "Πιθανότητα": [f"{ph * 100:.1f}%", f"{(1 - ph) * 100:.1f}%"],
                "Δίκαιη απόδοση": [
                    f"{1 / max(ph, 1e-9):.2f}",
                    f"{1 / max(1 - ph, 1e-9):.2f}",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Λεπτομέρειες μοντέλου"):
        st.write(
            f"Μέσοι πόντοι ανά ομάδα στο πρωτάθλημα: **{res['base']:.1f}**  \n"
            f"Συντελεστής γηπεδούχου: **{res['hfa']:.3f}**, "
            f"φιλοξενούμενου: **{res['afa']:.3f}**"
        )
        st.table(
            pd.DataFrame(
                {
                    "Ομάδα": [home, away],
                    "Επίθεση (1.00 = μέσος όρος)": [f"{res['ah']:.3f}", f"{res['aa']:.3f}"],
                    "Άμυνα (κάτω από 1 = καλή)": [f"{res['dh']:.3f}", f"{res['da']:.3f}"],
                }
            )
        )

    for team in (home, away):
        with st.expander(f"Τελευταία παιχνίδια: {team}"):
            g = df[(df.home == team) | (df.away == team)].tail(10).iloc[::-1]
            show = pd.DataFrame(
                {
                    "Ημερομηνία": g.day.dt.date,
                    "Αντίπαλος": np.where(g.home == team, g.away, g.home),
                    "Έδρα": np.where(g.home == team, "Εντός", "Εκτός"),
                    "Σκορ (ομάδα-αντίπαλος)": np.where(
                        g.home == team,
                        g.hp.astype(str) + "-" + g.ap.astype(str),
                        g.ap.astype(str) + "-" + g.hp.astype(str),
                    ),
                }
            )
            st.dataframe(show, hide_index=True, use_container_width=True)

    # ------------------------------------------------- Αγορές & value ----
    st.divider()
    st.subheader("Αγορές & value")
    st.caption(
        "Γράψε στη στήλη «Απόδοση στοιχηματικού» την απόδοση που βλέπεις στο "
        "site σου. Παρακάτω θα φανεί αν η επιλογή αξίζει."
    )

    ml, spread, totals, team_tot = market_rows(res, home, away, params)
    tabs = st.tabs(["Νικητής", "Handicap", "Σύνολο πόντων", "Πόντοι ομάδας"])
    kid = pred["id"]
    edited_all, all_rows = [], []
    for tab, rows, cat, key in (
        (tabs[0], ml, "Νικητής", f"nba_ml_{kid}"),
        (tabs[1], spread, "Handicap", f"nba_sp_{kid}"),
        (tabs[2], totals, "Σύνολο", f"nba_tt_{kid}"),
        (tabs[3], team_tot, "Πόντοι ομάδας", f"nba_tm_{kid}"),
    ):
        with tab:
            edited_all.append(render_market_tab(rows, cat, key))
            all_rows += [(cat,) + r for r in rows]

    st.markdown("#### Αποτέλεσμα ελέγχου value")
    allv = pd.concat(edited_all)
    allv = allv[allv["Απόδοση στοιχηματικού"].notna()]
    if allv.empty:
        st.info("Γράψε αποδόσεις στους πίνακες πάνω για να δεις ποιες επιλογές αξίζουν.")
    else:
        o = allv["Απόδοση στοιχηματικού"].astype(float).values
        pr = allv["p"].astype(float).values
        edge = pr * o - 1
        kelly = np.where(o > 1, edge / (o - 1), 0)
        stake = np.where(edge >= min_edge, np.minimum(0.25 * kelly, 0.02), 0)
        vt = pd.DataFrame(
            {
                "Αγορά": allv["Αγορά"].values,
                "Επιλογή": allv["Επιλογή"].values,
                "Πιθανότητα": [f"{x * 100:.1f}%" for x in pr],
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
            "εγγύηση. Αν το μοντέλο ξέρει λιγότερα από τον στοιχηματικό (π.χ. για "
            "τραυματισμούς), το «value» είναι συνήθως λάθος του μοντέλου."
        )

# -------------------------------------------------------------- Backtest ----
st.divider()
st.subheader("Backtest")
st.caption(
    "Τρέχει το μοντέλο σε παλιούς αγώνες χρησιμοποιώντας μόνο δεδομένα που "
    "υπήρχαν πριν από κάθε αγώνα, και το συγκρίνει με απλές βάσεις."
)
n_test = st.slider("Πόσοι πρόσφατοι αγώνες να ελεγχθούν", 50, 600, 300)

if st.button("Τρέξε Backtest", use_container_width=True):
    with st.spinner("Υπολογισμός..."):
        bt = run_backtest(df, n_test, params)
    if len(bt) < 30:
        st.error("Πολύ λίγοι αγώνες για αξιόπιστο έλεγχο.")
    else:
        acc, brier, ll = bin_metrics(bt.p, bt.y)
        bacc, bbrier, bll = bin_metrics(bt.bp, bt.y)
        st.write(f"Ελέγχθηκαν **{len(bt)}** αγώνες.")
        st.table(
            pd.DataFrame(
                {
                    "Μέτρο": [
                        "Σωστός νικητής",
                        "Brier (χαμηλότερο = καλύτερο)",
                        "Log loss (χαμηλότερο = καλύτερο)",
                        "Μέσο λάθος διαφοράς (πόντοι)",
                        "Μέσο λάθος συνόλου (πόντοι)",
                    ],
                    "Μοντέλο": [
                        f"{acc * 100:.1f}%", f"{brier:.3f}", f"{ll:.3f}",
                        f"{(bt.M - bt.margin).abs().mean():.2f}",
                        f"{(bt['T'] - bt.total).abs().mean():.2f}",
                    ],
                    "Απλή βάση*": [
                        f"{bacc * 100:.1f}%", f"{bbrier:.3f}", f"{bll:.3f}",
                        f"{(bt.bM - bt.margin).abs().mean():.2f}",
                        f"{(bt.bT - bt.total).abs().mean():.2f}",
                    ],
                }
            )
        )
        st.caption(
            "*Η απλή βάση προβλέπει πάντα τον μέσο όρο του πρωταθλήματος "
            "(ποσοστό νίκης γηπεδούχου, μέση διαφορά, μέσο σύνολο), χωρίς να "
            "κοιτάει τις ομάδες."
        )
        if brier < bbrier:
            st.success("Το μοντέλο πετυχαίνει καλύτερα από την απλή βάση.")
        else:
            st.warning("Το μοντέλο δεν ξεπερνά την απλή βάση. Δοκίμασε άλλες ρυθμίσεις.")

        rs_m = (bt.margin - bt.M).std()
        rs_t = (bt.total - bt["T"]).std()
        st.markdown("#### Προτεινόμενες τυπικές αποκλίσεις")
        st.write(
            f"Από τα δεδομένα: διαφορά ≈ **{rs_m:.1f}** πόντοι, "
            f"σύνολο ≈ **{rs_t:.1f}** πόντοι. Βάλε αυτές τις τιμές στα "
            "sliders αριστερά, για να βγαίνουν σωστές οι πιθανότητες "
            "στις αγορές handicap και συνόλου."
        )

        st.markdown("#### Βαθμονόμηση νίκης γηπεδούχου")
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        bt["bin"] = pd.cut(bt.p, bins)
        cal = bt.groupby("bin", observed=True).agg(
            Αγώνες=("y", "size"), Πρόβλεψη=("p", "mean"), Πραγματικά=("y", "mean")
        )
        cal["Πρόβλεψη"] = (cal["Πρόβλεψη"] * 100).round(1).astype(str) + "%"
        cal["Πραγματικά"] = (cal["Πραγματικά"] * 100).round(1).astype(str) + "%"
        cal.index = cal.index.astype(str)
        st.table(cal)
