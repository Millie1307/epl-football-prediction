
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from datetime import date

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ English Premier League Match Predictor")
st.write("Machine Learning prediction of Home Win, Draw, or Away Win.")

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("football_model10.joblib")
    features = joblib.load("football_features.joblib")
    return model, features

@st.cache_data
def load_data():
    data = pd.read_csv("epl_data.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    return data

@st.cache_data
def load_metadata():
    with open("football_metadata.json", "r") as f:
        return json.load(f)

model, features_loaded = load_model()
epl = load_data()
metadata = load_metadata()

# ============================================================
# PREDICTION FEATURE ENGINE
# ============================================================

def create_prediction_features(data, home_team, away_team, prediction_date):

    prediction_date = pd.Timestamp(prediction_date)

    historical = data[data["Date"] < prediction_date].copy()
    historical = historical.sort_values("Date").reset_index(drop=True)

    if home_team not in set(data["HomeTeam"]) | set(data["AwayTeam"]):
        raise ValueError(f"{home_team} not found in historical data.")

    if away_team not in set(data["HomeTeam"]) | set(data["AwayTeam"]):
        raise ValueError(f"{away_team} not found in historical data.")

    # --------------------------------------------------------
    # CURRENT SEASON
    # --------------------------------------------------------

    season_rows = historical[
        (historical["HomeTeam"] == home_team) |
        (historical["AwayTeam"] == home_team) |
        (historical["HomeTeam"] == away_team) |
        (historical["AwayTeam"] == away_team)
    ]

    if len(season_rows) == 0:
        raise ValueError("Not enough historical data.")

    current_season = season_rows.iloc[-1]["Season"]

    season_data = historical[
        historical["Season"] == current_season
    ].copy()

    # --------------------------------------------------------
    # RECENT FORM
    # --------------------------------------------------------

    def get_form(team):

        matches = historical[
            (historical["HomeTeam"] == team) |
            (historical["AwayTeam"] == team)
        ].tail(5)

        points = 0
        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0

        for _, match in matches.iterrows():

            if match["HomeTeam"] == team:
                gf = match["FTHG"]
                ga = match["FTAG"]
                result = match["FTR"]

                if result == "H":
                    wins += 1
                    points += 3
                elif result == "D":
                    draws += 1
                    points += 1
                else:
                    losses += 1

            else:
                gf = match["FTAG"]
                ga = match["FTHG"]
                result = match["FTR"]

                if result == "A":
                    wins += 1
                    points += 3
                elif result == "D":
                    draws += 1
                    points += 1
                else:
                    losses += 1

            goals_for += gf
            goals_against += ga

        return {
            "FormPoints": points,
            "FormWins": wins,
            "FormDraws": draws,
            "FormLosses": losses,
            "FormGoalsFor": goals_for,
            "FormGoalsAgainst": goals_against
        }

    home_form = get_form(home_team)
    away_form = get_form(away_team)

    # --------------------------------------------------------
    # LEAGUE STANDINGS
    # --------------------------------------------------------

    teams = set(season_data["HomeTeam"]) | set(season_data["AwayTeam"])

    table = {}

    for team in teams:
        table[team] = {
            "points": 0,
            "gf": 0,
            "ga": 0
        }

    for _, match in season_data.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        hg = match["FTHG"]
        ag = match["FTAG"]

        table[home]["gf"] += hg
        table[home]["ga"] += ag

        table[away]["gf"] += ag
        table[away]["ga"] += hg

        if match["FTR"] == "H":
            table[home]["points"] += 3
        elif match["FTR"] == "A":
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    standings = []

    for team, stats in table.items():
        standings.append({
            "Team": team,
            "Points": stats["points"],
            "GoalDifference": stats["gf"] - stats["ga"],
            "GoalsFor": stats["gf"]
        })

    standings = pd.DataFrame(standings)

    standings = standings.sort_values(
        ["Points", "GoalDifference", "GoalsFor"],
        ascending=False
    ).reset_index(drop=True)

    standings["Position"] = range(1, len(standings) + 1)

    def get_position(team):
        row = standings[standings["Team"] == team]

        if len(row) == 0:
            return 20

        return int(row.iloc[0]["Position"])

    home_position = get_position(home_team)
    away_position = get_position(away_team)

    home_stats = standings[standings["Team"] == home_team]

    if len(home_stats):
        home_points = home_stats.iloc[0]["Points"]
        home_gd = home_stats.iloc[0]["GoalDifference"]
        home_gf = home_stats.iloc[0]["GoalsFor"]
    else:
        home_points = home_gd = home_gf = 0

    away_stats = standings[standings["Team"] == away_team]

    if len(away_stats):
        away_points = away_stats.iloc[0]["Points"]
        away_gd = away_stats.iloc[0]["GoalDifference"]
        away_gf = away_stats.iloc[0]["GoalsFor"]
    else:
        away_points = away_gd = away_gf = 0

    # --------------------------------------------------------
    # ELO
    # --------------------------------------------------------

    elo = {}

    for team in set(historical["HomeTeam"]) | set(historical["AwayTeam"]):
        elo[team] = 1500

    K = 20
    HOME_ADVANTAGE = 60

    for _, match in historical.iterrows():

        home = match["HomeTeam"]
        away = match["AwayTeam"]

        home_elo = elo.get(home, 1500)
        away_elo = elo.get(away, 1500)

        expected_home = 1 / (
            1 + 10 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400)
        )

        if match["FTR"] == "H":
            actual_home = 1
        elif match["FTR"] == "D":
            actual_home = 0.5
        else:
            actual_home = 0

        elo[home] = home_elo + K * (actual_home - expected_home)
        elo[away] = away_elo + K * ((1 - actual_home) - (1 - expected_home))

    home_elo = elo.get(home_team, 1500)
    away_elo = elo.get(away_team, 1500)

    elo_diff = home_elo - away_elo
    adjusted_elo_diff = elo_diff + HOME_ADVANTAGE
    home_elo_advantage = home_elo + HOME_ADVANTAGE

    # --------------------------------------------------------
    # REST DAYS
    # --------------------------------------------------------

    def get_rest_days(team):

        matches = historical[
            (historical["HomeTeam"] == team) |
            (historical["AwayTeam"] == team)
        ]

        if len(matches) == 0:
            return 7

        last_date = matches.iloc[-1]["Date"]

        rest = (prediction_date - last_date).days

        return min(max(rest, 2), 14)

    home_rest = get_rest_days(home_team)
    away_rest = get_rest_days(away_team)

    # --------------------------------------------------------
    # ATTACK / DEFENCE STRENGTH
    # --------------------------------------------------------

    def get_strength(team):

        matches = historical[
            (historical["HomeTeam"] == team) |
            (historical["AwayTeam"] == team)
        ].tail(5)

        if len(matches) == 0:
            return 0, 0

        gf = []
        ga = []

        for _, match in matches.iterrows():

            if match["HomeTeam"] == team:
                gf.append(match["FTHG"])
                ga.append(match["FTAG"])
            else:
                gf.append(match["FTAG"])
                ga.append(match["FTHG"])

        return np.mean(gf), np.mean(ga)

    home_attack, home_defense = get_strength(home_team)
    away_attack, away_defense = get_strength(away_team)

    # --------------------------------------------------------
    # BUILD FEATURE ROW
    # --------------------------------------------------------

    row = {
        "Home_FormPoints": home_form["FormPoints"],
        "Home_FormWins": home_form["FormWins"],
        "Home_FormDraws": home_form["FormDraws"],
        "Home_FormLosses": home_form["FormLosses"],
        "Home_FormGoalsFor": home_form["FormGoalsFor"],
        "Home_FormGoalsAgainst": home_form["FormGoalsAgainst"],

        "Away_FormPoints": away_form["FormPoints"],
        "Away_FormWins": away_form["FormWins"],
        "Away_FormDraws": away_form["FormDraws"],
        "Away_FormLosses": away_form["FormLosses"],
        "Away_FormGoalsFor": away_form["FormGoalsFor"],
        "Away_FormGoalsAgainst": away_form["FormGoalsAgainst"],

        "Form_Wins_Diff": (
            home_form["FormWins"] - away_form["FormWins"]
        ),

        "Form_GoalsFor_Diff": (
            home_form["FormGoalsFor"] - away_form["FormGoalsFor"]
        ),

        "Home_Position": home_position,
        "Away_Position": away_position,
        "Position_Diff": home_position - away_position,

        "Points_Diff": home_points - away_points,
        "GoalDifference_Diff": home_gd - away_gd,

        "League_GoalsFor_Diff": home_gf - away_gf,

        "Home_Elo": home_elo,
        "Away_Elo": away_elo,
        "Elo_Diff": elo_diff,

        "Home_RestDays": home_rest,
        "Away_RestDays": away_rest,
        "RestDays_Diff": home_rest - away_rest,

        "Home_Attack": home_attack,
        "Home_Defense": home_defense,
        "Away_Attack": away_attack,
        "Away_Defense": away_defense,

        "Attack_Diff": home_attack - away_attack,
        "Defense_Diff": home_defense - away_defense
    }

    features = pd.DataFrame([row])

    # Make sure feature order matches training
    features = features.reindex(columns=features_loaded)

    return features


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Match Details")

teams = sorted(
    set(epl["HomeTeam"].unique()) |
    set(epl["AwayTeam"].unique())
)

home_team = st.sidebar.selectbox(
    "Home Team",
    teams,
    index=teams.index("Liverpool") if "Liverpool" in teams else 0
)

away_options = [team for team in teams if team != home_team]

away_team = st.sidebar.selectbox(
    "Away Team",
    away_options
)

prediction_date = st.sidebar.date_input(
    "Prediction Date",
    value=date.today()
)

predict_button = st.sidebar.button(
    "Predict Match",
    type="primary"
)

# ============================================================
# MAIN APP
# ============================================================

st.subheader("Model Information")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Test Accuracy",
    f"{metadata['accuracy']:.1%}"
)

col2.metric(
    "ROC-AUC",
    f"{metadata['roc_auc']:.3f}"
)

col3.metric(
    "Training Matches",
    metadata["training_matches"]
)

col4.metric(
    "Features",
    metadata["number_of_features"]
)

st.divider()

if predict_button:

    try:

        features = create_prediction_features(
            epl,
            home_team,
            away_team,
            prediction_date
        )

        probabilities = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]

        probability_dict = dict(
            zip(model.classes_, probabilities)
        )

        st.subheader(
            f"{home_team} vs {away_team}"
        )

        if prediction == "H":
            result_text = f"🏠 {home_team} Win"
        elif prediction == "A":
            result_text = f"✈️ {away_team} Win"
        else:
            result_text = "🤝 Draw"

        st.success(
            f"### Prediction: {result_text}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            f"🏠 {home_team}",
            f"{probability_dict['H']:.1%}"
        )

        col2.metric(
            "🤝 Draw",
            f"{probability_dict['D']:.1%}"
        )

        col3.metric(
            f"✈️ {away_team}",
            f"{probability_dict['A']:.1%}"
        )

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Outcome": [
                f"{home_team} Win",
                "Draw",
                f"{away_team} Win"
            ],
            "Probability": [
                probability_dict["H"],
                probability_dict["D"],
                probability_dict["A"]
            ]
        })

        st.bar_chart(
            probability_df.set_index("Outcome")
        )

        with st.expander("View Model Features"):
            st.dataframe(features)

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

else:

    st.info(
        "Select the teams and prediction date, then click "
        "**Predict Match**."
    )

st.divider()

st.caption(
    "Model: Logistic Regression (Model 10) | "
    "Historical EPL data: 2019/20–2025/26"
)
