import difflib
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import nbinom, norm, poisson

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
STATS = ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA", "STL"]
COMBOS = {
    "PRA": ["PTS", "REB", "AST"],
    "PR": ["PTS", "REB"],
    "PA": ["PTS", "AST"],
    "RA": ["REB", "AST"],
}
NORMAL_STATS = {"PTS", "PRA", "PR", "PA", "RA"}
NUM_COLS = ["Γραμμή", "Over απόδοση", "Under απόδοση", "Διόρθωση μέσου"]


def get_json(url, params=None):
    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=25)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------- Δεδομένα ----
@st.cache_data(ttl=86400)
def load_teams():
    j = get_json(f"{SITE}/teams")
    teams = j["sports"][0]["leagues"][0]["teams"]
    return {t["team"]["displayName"]: str(t["team"]["id"]) for t in teams}


@st.cache_data(ttl=900)
def load_injuries():
    j = get_json(f"{SITE}/injuries")
    res = {}
    for t in j.get("injuries", []):
        name = t.get("displayName", "")
        lst = []
        for i in t.get("injuries", []):
            a = i.get("athlete", {}) or {}
            det = i.get("details", {}) or {}
            detail = f"{det.get('type', '')} {det.get('detail', '')}".strip()
            if not detail:
                detail = (i.get("type", {}) or {}).get("description", "") or ""
            lst.append(
                {
                    "id": str(a.get("id", "")),
                    "name": a.get("displayName", "?"),
                    "status": i.get("status", "") or "",
                    "detail": detail,
                    "comment": i.get("shortComment") or i.get("longComment") or "",
                    "date": str(i.get("date", ""))[:10],
                }
            )
        res[name] = lst
    return res


@st.cache_data(ttl=21600)
def load_roster(team_id):
    j = get_json(f"{SITE}/teams/{team_id}/roster")
    out = []
    for a in j.get("athletes", []):
        items = a.get("items") if isinstance(a, dict) and "items" in a else [a]
        for p in items:
            pid = p.get("id")
            nm = p.get("displayName") or p.get("fullName")
            if pid and nm:
                out.append({"id": str(pid), "name": nm})
    return out


@st.cache_data(ttl=900)
def load_news(team_id):
    j = get_json(f"{SITE}/news", params={"team": team_id})
    out = []
    for a in j.get("articles", [])[:8]:
        out.append(
            {
                "headline": a.get("headline", ""),
                "published": str(a.get("published", ""))[:10],
                "url": ((a.get("links") or {}).get("web") or {}).get("href", ""),
            }
        )
    return out


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def raw_gamelog(aid):
    try:
        j = get_json(f"{WEB}/athletes/{aid}/gamelog")
    except Exception as e:
        return pd.DataFrame(), f"σφάλμα: {e}"
    labels = j.get("labels") or j.get("names") or []
    meta = j.get("events", {}) or {}
    rows = []
    for stype in j.get("seasonTypes", []):
        if "Preseason" in str(stype.get("displayName", "")):
            continue
        for cat in stype.get("categories", []):
            for ev in cat.get("events", []):
                eid = str(ev.get("eventId"))
                rec = dict(zip(labels, ev.get("stats", [])))
                m = meta.get(eid, {}) or {}
                three = str(rec.get("3PT", "")).split("-")[0]
                rows.append(
                    {
                        "date": pd.to_datetime(m.get("gameDate"), utc=True, errors="coerce"),
                        "opp": ((m.get("opponent") or {}).get("abbreviation", "")),
                        "MIN": fnum(rec.get("MIN")),
                        "PTS": fnum(rec.get("PTS")),
                        "REB": fnum(rec.get("REB")),
                        "AST": fnum(rec.get("AST")),
                        "STL": fnum(rec.get("STL")),
                        "3PM": fnum(three),
                    }
                )
    if not rows:
        return pd.DataFrame(), f"κανένα παιχνίδι (κλειδιά: {list(j.keys())[:8]})"
    df = pd.DataFrame(rows).dropna(subset=["MIN"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df, ""


@st.cache_data(ttl=1800, show_spinner="Φόρτωση παιχνιδιών παικτών...")
def load_gamelogs(ids):
    ids = list(ids)
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(raw_gamelog, ids))
    return dict(zip(ids, results))


# --------------------------------------------------------------- Μοντέλο ----
def stat_series(df, stat):
    if stat in COMBOS:
        return df[COMBOS[stat]].sum(axis=1, min_count=1)
    return df[stat]


def analyze(vals, line, adj, stat, half_life, shrink):
    v = np.asarray(vals, dtype=float) + adj
    n = len(v)
    w = 0.5 ** (np.arange(n) / half_life)
    w = w / w.sum()
    m = float((w * v).sum())
    var = float((w * (v - m) ** 2).sum())

    over, under = w[v > line].sum(), w[v < line].sum()
    p_emp = over / (over + under) if over + under > 0 else 0.5

    if stat in NORMAL_STATS:
        sd = max(np.sqrt(var), 1.0)
        po = 1 - norm.cdf((np.floor(line) + 0.5 - m) / sd)
        pu = norm.cdf((np.ceil(line) - 0.5 - m) / sd)
    else:
        mm = max(m, 0.05)
        if var > mm * 1.05:
            dist = nbinom(mm * mm / (var - mm), mm / var)
        else:
            dist = poisson(mm)
        po = 1 - dist.cdf(np.floor(line))
        pu = dist.cdf(np.ceil(line) - 1)
    p_par = po / (po + pu) if po + pu > 0 else 0.5

    p = 0.5 * p_emp + 0.5 * p_par
    p = (n * p + shrink * 0.5) / (n + shrink)
    return p, m


EMOJI = {"out": "🔴", "doubtful": "🟠", "questionable": "🟡", "day-to-day": "🟡", "probable": "🟢"}


def status_label(s):
    s = (s or "").strip()
    return f"{EMOJI.get(s.lower(), '⚪')} {s}" if s else "Υγιής"


# ---------------------------------------------------------- Εισαγωγή ----
def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def to_decimal(x):
    """Δεκαδική απόδοση. Δέχεται και αμερικάνικες (π.χ. -110, +150)."""
    try:
        v = float(str(x).replace(",", ".").replace("+", "").strip())
    except Exception:
        return np.nan
    if v >= 100:
        return 1 + v / 100
    if v <= -100:
        return 1 + 100 / abs(v)
    return v if v > 1 else np.nan


def parse_import(text, labels):
    norm_map = {_norm(lb): lb for lb in labels}
    rows, problems = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"[;\t|]", line)]
        if len(parts) < 3:
            problems.append(f"Δεν κατάλαβα τη γραμμή: {line}")
            continue
        name, stat = parts[0], parts[1].upper()
        if stat not in STATS:
            problems.append(f"Άγνωστο στατιστικό «{parts[1]}» στη γραμμή: {line}")
            continue
        try:
            ln = float(parts[2].replace(",", "."))
        except Exception:
            problems.append(f"Μη έγκυρη γραμμή στο: {line}")
            continue
        key = _norm(name)
        label = norm_map.get(key)
        if label is None:
            close = difflib.get_close_matches(key, list(norm_map), n=1, cutoff=0.75)
            label = norm_map[close[0]] if close else None
        if label is None:
            problems.append(f"Ο παίκτης «{name}» δεν βρέθηκε στις δύο ομάδες.")
        rows.append(
            {
                "Παίκτης": label,
                "Στατιστικό": stat,
                "Γραμμή": ln,
                "Over απόδοση": to_decimal(parts[3]) if len(parts) > 3 else np.nan,
                "Under απόδοση": to_decimal(parts[4]) if len(parts) > 4 else np.nan,
                "Διόρθωση μέσου": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, problems


def empty_template():
    return pd.DataFrame(
        {
            "Παίκτης": [None],
            "Στατιστικό": ["PTS"],
            "Γραμμή": [np.nan],
            "Over απόδοση": [np.nan],
            "Under απόδοση": [np.nan],
            "Διόρθωση μέσου": [0.0],
        }
    )


# ------------------------------------------------------------------- UI ----
st.set_page_config(page_title="NBA Props", page_icon="🏀")
st.title("🏀 NBA: Τραυματισμοί, νέα & player props")

with st.sidebar:
    st.header("Ρυθμίσεις")
    n_games = st.slider("Τελευταία παιχνίδια παίκτη", 5, 40, 15)
    half_life = st.slider(
        "Ημι-ζωή βαρών (παιχνίδια)", 2, 30, 8,
        help="Μικρότερο = μετράνε περισσότερο τα πιο πρόσφατα παιχνίδια.",
    )
    min_minutes = st.slider(
        "Ελάχιστα λεπτά για να μετρήσει ένα παιχνίδι", 0, 30, 10,
        help="Αποκλείει παιχνίδια όπου ο παίκτης έπαιξε ελάχιστα (τραυματισμός, garbage time).",
    )
    shrink = st.slider(
        "Συρρίκνωση προς 50%", 0, 10, 3,
        help="Μεγαλύτερο = πιο συντηρητικές πιθανότητες. 3 = σαν να υπήρχαν 3 επιπλέον παιχνίδια στο 50%.",
    )
    min_edge = (
        st.slider("Ελάχιστο edge για ένδειξη value", 1, 25, 8,
                  help="Edge = πιθανότητα μοντέλου × απόδοση − 1.")
        / 100
    )

try:
    teams = load_teams()
except Exception as e:
    st.error(f"Αποτυχία φόρτωσης ομάδων: {e}")
    st.stop()
names = sorted(teams)

c1, c2 = st.columns(2)
t1 = c1.selectbox("Ομάδα 1", names, index=0)
t2 = c2.selectbox("Ομάδα 2", names, index=1)
if t1 == t2:
    st.error("Διάλεξε διαφορετικές ομάδες.")
    st.stop()

# ------------------------------------------------------- Τραυματισμοί ----
st.subheader("Τραυματισμοί")
try:
    inj = load_injuries()
except Exception as e:
    inj = {}
    st.warning(f"Δεν φορτώθηκαν οι τραυματισμοί: {e}")

status_by_name = {}
inj_rows = []
for t in (t1, t2):
    for i in inj.get(t, []):
        status_by_name[i["name"]] = i["status"]
        inj_rows.append((t, i))

form_ids = tuple(sorted({i["id"] for _, i in inj_rows if i["id"]}))
form_logs = load_gamelogs(form_ids) if form_ids else {}


def form_text(pid):
    df, _ = form_logs.get(pid, (pd.DataFrame(), ""))
    if df.empty:
        return "—"
    g = df[df.MIN >= 10].head(10)
    if g.empty:
        return "—"
    return (
        f"{g.MIN.mean():.0f} λ · {g.PTS.mean():.1f} π · "
        f"{g.REB.mean():.1f} ρ · {g.AST.mean():.1f} α"
    )


if not inj_rows:
    st.info("Δεν βρέθηκαν καταγεγραμμένοι τραυματισμοί για αυτές τις ομάδες.")
else:
    st.dataframe(
        pd.DataFrame(
            {
                "Ομάδα": [t for t, _ in inj_rows],
                "Παίκτης": [i["name"] for _, i in inj_rows],
                "Κατάσταση": [status_label(i["status"]) for _, i in inj_rows],
                "Τραυματισμός": [i["detail"] for _, i in inj_rows],
                "Μ.Ο. τελευταίων 10 (λεπτά·πόντοι·ριμπ·ασίστ)": [
                    form_text(i["id"]) for _, i in inj_rows
                ],
                "Σχόλιο": [i["comment"] for _, i in inj_rows],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Η στήλη με τα λεπτά και τους πόντους δείχνει πόσο σημαντικός είναι ο παίκτης που λείπει. "
        "Όταν λείπει βασικός παίκτης, οι συμπαίκτες του συνήθως παίρνουν περισσότερα σουτ. "
        "Εφάρμοσέ το με τη στήλη «Διόρθωση μέσου» παρακάτω."
    )

# --------------------------------------------------------------- Νέα ----
st.subheader("Τελευταία νέα")
for t in (t1, t2):
    with st.expander(t):
        try:
            news = load_news(teams[t])
        except Exception as e:
            news = []
            st.write(f"Δεν φορτώθηκαν νέα: {e}")
        if not news:
            st.write("Δεν βρέθηκαν νέα.")
        for a in news:
            if a["url"]:
                st.markdown(f"- {a['published']} · [{a['headline']}]({a['url']})")
            else:
                st.markdown(f"- {a['published']} · {a['headline']}")

# --------------------------------------------------------------- Props ----
st.divider()
st.subheader("Player props & value")
st.caption(
    "Γράψε τη γραμμή και την απόδοση που βλέπεις στο site σου, ή κάνε εισαγωγή από κείμενο "
    "παρακάτω. PRA = πόντοι+ριμπάουντ+ασίστ, PR = πόντοι+ριμπάουντ, PA = πόντοι+ασίστ, "
    "RA = ριμπάουντ+ασίστ, 3PM = τρίποντα που βάζει. "
    "Η «Διόρθωση μέσου» προστίθεται στον αναμενόμενο αριθμό (π.χ. +3 πόντοι αν λείπει ο βασικός συμπαίκτης)."
)

try:
    roster = load_roster(teams[t1]) + load_roster(teams[t2])
except Exception as e:
    st.error(f"Δεν φορτώθηκαν οι ρόστερ: {e}")
    st.stop()
label_to_id = {p["name"]: p["id"] for p in roster}
labels = sorted(label_to_id)

# Κατάσταση πίνακα (επιβιώνει των ανανεώσεων, μηδενίζεται όταν αλλάζουν ομάδες)
pair = f"{t1}|{t2}"
if st.session_state.get("props_pair") != pair:
    st.session_state["props_pair"] = pair
    st.session_state["props_df"] = empty_template()
    st.session_state["props_ver"] = st.session_state.get("props_ver", 0) + 1

with st.expander("Εισαγωγή από κείμενο (π.χ. από φωτογραφία μέσω Claude)"):
    st.caption(
        "Μία γραμμή ανά επιλογή, με ερωτηματικά: Παίκτης; Στατιστικό; Γραμμή; Over; Under. "
        "Παράδειγμα: Jayson Tatum; PTS; 27.5; 1.87; 1.93. "
        "Δέχεται και αμερικάνικες αποδόσεις (π.χ. -110)."
    )
    txt = st.text_area("Επικόλλησε εδώ", height=160, key="import_text")
    if st.button("Εισαγωγή στον πίνακα"):
        new_df, problems = parse_import(txt, labels)
        if new_df.empty:
            st.error("Δεν διαβάστηκε καμία γραμμή.")
        else:
            st.session_state["props_df"] = new_df
            st.session_state["props_ver"] += 1
            st.success(f"Εισήχθησαν {len(new_df)} γραμμές.")
        if problems:
            st.warning("Έλεγξε τα παρακάτω:")
            for p in problems:
                st.write("- " + p)

edited = st.data_editor(
    st.session_state["props_df"],
    key=f"props_{pair}_{st.session_state['props_ver']}",
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "Παίκτης": st.column_config.SelectboxColumn("Παίκτης", options=labels),
        "Στατιστικό": st.column_config.SelectboxColumn("Στατιστικό", options=STATS),
        "Γραμμή": st.column_config.NumberColumn("Γραμμή", min_value=0.0, step=0.5, format="%.1f"),
        "Over απόδοση": st.column_config.NumberColumn("Over απόδοση", min_value=1.01, step=0.01, format="%.2f"),
        "Under απόδοση": st.column_config.NumberColumn("Under απόδοση", min_value=1.01, step=0.01, format="%.2f"),
        "Διόρθωση μέσου": st.column_config.NumberColumn("Διόρθωση μέσου", step=0.5, format="%.1f"),
    },
)

valid = edited.dropna(subset=["Παίκτης", "Στατιστικό", "Γραμμή"])
if valid.empty:
    st.info("Διάλεξε παίκτη, στατιστικό και γραμμή (πάτα το + για περισσότερες γραμμές).")
else:
    ids = tuple(sorted({label_to_id[p] for p in valid["Παίκτης"] if p in label_to_id}))
    logs = load_gamelogs(ids)
    out = []
    for _, r in valid.iterrows():
        pid = label_to_id.get(r["Παίκτης"])
        df, note = logs.get(pid, (pd.DataFrame(), "άγνωστος παίκτης"))
        g = df[df.MIN >= min_minutes].head(n_games) if not df.empty else df
        if g.empty or len(g) < 6:
            out.append({"Παίκτης": r["Παίκτης"], "Στατ.": r["Στατιστικό"],
                        "Σχόλιο": f"Λίγα παιχνίδια ({len(g)}) {note}"})
            continue
        vals = stat_series(g, r["Στατιστικό"]).dropna().values
        if len(vals) < 6:
            out.append({"Παίκτης": r["Παίκτης"], "Στατ.": r["Στατιστικό"],
                        "Σχόλιο": "Λίγα δεδομένα για αυτό το στατιστικό"})
            continue
        adj = 0.0 if pd.isna(r["Διόρθωση μέσου"]) else float(r["Διόρθωση μέσου"])
        line = float(r["Γραμμή"])
        p_over, mean = analyze(vals, line, adj, r["Στατιστικό"], half_life, shrink)
        l10 = (vals[:10] + adj > line).mean() * 100
        l5 = vals[:5].mean() + adj

        oo, ou = r.get("Over απόδοση"), r.get("Under απόδοση")
        cands = []
        if pd.notna(oo):
            cands.append(("Over", p_over, float(oo)))
        if pd.notna(ou):
            cands.append(("Under", 1 - p_over, float(ou)))
        status = status_by_name.get(r["Παίκτης"], "")

        row = {
            "Παίκτης": r["Παίκτης"],
            "Στατ.": r["Στατιστικό"],
            "Γραμμή": line,
            "Μέσος": round(mean, 1),
            "Τελ.5": round(l5, 1),
            "Over %": f"{p_over * 100:.1f}%",
            "Over τελ.10": f"{l10:.0f}%",
            "Κατάσταση": status_label(status),
            "Παιχνίδια": len(vals),
        }
        if cands:
            side, p, o = max(cands, key=lambda c: c[1] * c[2] - 1)
            edge = p * o - 1
            kelly = edge / (o - 1) if o > 1 else 0
            ok = edge >= min_edge
            if status.lower() == "out":
                flag = "⚠️ Εκτός"
            else:
                flag = "✅ Αξίζει" if ok else "❌ Όχι"
            row.update(
                {
                    "Πλευρά": side,
                    "Πιθ. πλευράς": f"{p * 100:.1f}%",
                    "Απόδοση": f"{o:.2f}",
                    "Edge": f"{edge * 100:+.1f}%",
                    "Ένδειξη": flag,
                    "% bankroll": f"{min(0.25 * kelly, 0.01) * 100:.1f}%" if ok else "0.0%",
                    "_e": edge,
                }
            )
        else:
            row.update({"Δίκαιη Over": f"{1 / max(p_over, 1e-9):.2f}",
                        "Δίκαιη Under": f"{1 / max(1 - p_over, 1e-9):.2f}"})
        out.append(row)

    res = pd.DataFrame(out)
    if "_e" in res.columns:
        res = res.sort_values("_e", ascending=False, na_position="last").drop(columns="_e")
    st.dataframe(res, hide_index=True, use_container_width=True)
    st.caption(
        "Η πιθανότητα προκύπτει από τα τελευταία παιχνίδια του παίκτη (μίγμα πραγματικής "
        "συχνότητας και στατιστικής κατανομής, με βάρος στα πρόσφατα). "
        "ΔΕΝ λαμβάνει υπόψη αντίπαλο, ρυθμό παιχνιδιού ή απουσίες συμπαικτών "
        "(για αυτά χρησιμοποίησε τη «Διόρθωση μέσου»). Edge = πιθανότητα × απόδοση − 1. "
        "Το «% bankroll» είναι ένα τέταρτο Kelly με όριο 1%. Δεν έχει γίνει backtest για props, "
        "οπότε το «value» εδώ είναι ένδειξη και όχι εγγύηση."
    )

# ----------------------------------------------- Έλεγχος δεδομένων παίκτη ----
with st.expander("Τελευταία παιχνίδια παίκτη (έλεγχος δεδομένων)"):
    who = st.selectbox("Παίκτης", labels, key="chk_player")
    d, note = load_gamelogs((label_to_id[who],))[label_to_id[who]]
    if d.empty:
        st.write(f"Δεν διαβάστηκαν παιχνίδια. {note}")
    else:
        show = d.head(10).copy()
        show["date"] = show["date"].dt.date
        st.dataframe(
            show.rename(columns={"date": "Ημερομηνία", "opp": "Αντίπαλος"}),
            hide_index=True,
            use_container_width=True,
        )
