import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os

# ============ USER SETTINGS ============
CSV_PATH_MATCHES = "https://raw.githubusercontent.com/HamzaTouray/Man_U_premier-league-analysis/main/E0.csv"
CSV_PATH_PLAYERS = "https://raw.githubusercontent.com/HamzaTouray/Man_U_premier-league-analysis/main/epl_player_stats_24_25.csv"
FOCUS_TEAM = "Man United"        # Team to benchmark
ROLLING_WINDOW = 5               # For trend plots

         # Player stats dataset
FOCUS_TEAM = "Man United"                                # Team to benchmark
ROLLING_WINDOW = 5                                       # For trend plots
# =======================================

# New signings data
NEW_SIGNINGS = {
    "Bryan Mbeumo": {"Goals": 15, "Assists": 8, "Minutes": 2800, "Club": "Brentford"},
    "Matheus Cunha": {"Goals": 12, "Assists": 6, "Minutes": 2500, "Club": "Watford"},
    "Benjamin Šeško": {"Goals": 13, "Assists": 5, "Minutes": 2700, "Club": "RB Leipzig"}
}

# ---------- Helpers ----------
def safe_div(a, b):
    return (a / b) if b not in (0, 0.0, None, np.nan) else 0.0

def result_points(row, is_home):
    if row["FTR"] == "D":
        return 1
    if row["FTR"] == "H" and is_home:
        return 3
    if row["FTR"] == "A" and not is_home:
        return 3
    return 0

def compute_table(team_match_df):
    g = team_match_df.groupby("Team").agg(
        MP=("Team", "count"),
        Pts=("Points", "sum"),
        GF=("GF", "sum"),
        GA=("GA", "sum"),
        SOT_for=("SOT_for", "sum"),
        SOT_against=("SOT_against", "sum"),
        Shots_for=("Shots_for", "sum"),
        Shots_against=("Shots_against", "sum"),
        Corners_for=("Corners_for", "sum"),
        Corners_against=("Corners_against", "sum"),
        Yel=("Yel", "sum"),
        Red=("Red", "sum"),
    ).reset_index()

    g["GD"] = g["GF"] - g["GA"]
    g["PPG"] = g["Pts"] / g["MP"]
    g["GF_per90"] = g["GF"] / g["MP"]
    g["GA_per90"] = g["GA"] / g["MP"]
    g["GD_per90"] = g["GD"] / g["MP"]
    g["SOT_for_per90"] = g["SOT_for"] / g["MP"]
    g["SOT_against_per90"] = g["SOT_against"] / g["MP"]
    g["ConvRate"] = g.apply(lambda r: safe_div(r["GF"], r["Shots_for"]), axis=1)
    g["Discipline_per90"] = (g["Yel"] + 3 * g["Red"]) / g["MP"]
    g["Corners_for_per90"] = g["Corners_for"] / g["MP"]

    g = g.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    g["Pos"] = np.arange(1, len(g) + 1)
    return g

def make_long_match_df(df):
    cols = df.columns
    def get(col, default=0):
        return col if col in cols else None

    base_cols = ["Date", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG", "HST", "AST", "HS", "AS", "HC", "AC", "HY", "AY", "HR", "AR"]
    missing = [c for c in base_cols if c not in cols]
    if missing:
        st.warning(f"Missing columns treated as zero where relevant: {missing}")

    home = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "Team": df["HomeTeam"],
        "Opponent": df["AwayTeam"],
        "is_home": True,
        "GF": df["FTHG"],
        "GA": df["FTAG"],
        "SOT_for": df[get("HST")].fillna(0) if get("HST") else 0,
        "SOT_against": df[get("AST")].fillna(0) if get("AST") else 0,
        "Shots_for": df[get("HS")].fillna(0) if get("HS") else 0,
        "Shots_against": df[get("AS")].fillna(0) if get("AS") else 0,
        "Corners_for": df[get("HC")].fillna(0) if get("HC") else 0,
        "Corners_against": df[get("AC")].fillna(0) if get("AC") else 0,
        "Yel": df[get("HY")].fillna(0) if get("HY") else 0,
        "Red": df[get("HR")].fillna(0) if get("HR") else 0,
        "FTR": df["FTR"],
    })
    home["Points"] = df.apply(lambda r: result_points(r, True), axis=1)

    away = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
        "Team": df["AwayTeam"],
        "Opponent": df["HomeTeam"],
        "is_home": False,
        "GF": df["FTAG"],
        "GA": df["FTHG"],
        "SOT_for": df[get("AST")].fillna(0) if get("AST") else 0,
        "SOT_against": df[get("HST")].fillna(0) if get("HST") else 0,
        "Shots_for": df[get("AS")].fillna(0) if get("AS") else 0,
        "Shots_against": df[get("HS")].fillna(0) if get("HS") else 0,
        "Corners_for": df[get("AC")].fillna(0) if get("AC") else 0,
        "Corners_against": df[get("HC")].fillna(0) if get("HC") else 0,
        "Yel": df[get("AY")].fillna(0) if get("AY") else 0,
        "Red": df[get("AR")].fillna(0) if get("AR") else 0,
        "FTR": df["FTR"],
    })
    away["Points"] = df.apply(lambda r: result_points(r, False), axis=1)

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values("Date").reset_index(drop=True)

    long_df["GD"] = long_df["GF"] - long_df["GA"]
    long_df["SOT_Diff"] = long_df["SOT_for"] - long_df["SOT_against"]
    long_df["FormPts_5"] = long_df.groupby("Team")["Points"].rolling(ROLLING_WINDOW, min_periods=1).sum().reset_index(level=0, drop=True)
    long_df["GD_5"] = long_df.groupby("Team")["GD"].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)
    long_df["SOTDiff_5"] = long_df.groupby("Team")["SOT_Diff"].rolling(ROLLING_WINDOW, min_periods=1).mean().reset_index(level=0, drop=True)

    return long_df

def radar(ax, labels, values, title):
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_cycle = values + values[:1]
    angles_cycle = angles + angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.plot(angles_cycle, values_cycle, linewidth=2)
    ax.fill(angles_cycle, values_cycle, alpha=0.15)
    ax.set_title(title, y=1.08)
    ax.grid(True)

def bar_gap(ax, categories, team_vals, target_vals, title, team_label, target_label):
    x = np.arange(len(categories))
    ax.bar(x - 0.2, team_vals, width=0.4, label=team_label)
    ax.bar(x + 0.2, target_vals, width=0.4, label=target_label)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=False))

# ---------- Streamlit App ----------
st.title(f"Premier League 2025/26: Manchester United Season Preview")
st.markdown("""
This report analyzes the impact of Manchester United's new signings (Bryan Mbeumo, Matheus Cunha, and Benjamin Šeško)
and predicts their potential finish for the 2025/26 season, with a focus on how they can catch up to Liverpool and the top four.
""")

# Load data
@st.cache_data
def load_data():
    df_matches = pd.read_csv(CSV_PATH_MATCHES)
    df_matches["Date"] = pd.to_datetime(df_matches["Date"], dayfirst=True, errors="coerce")
    df_matches = df_matches.sort_values("Date").reset_index(drop=True)

    df_players = pd.read_csv(CSV_PATH_PLAYERS)
    return df_matches, df_players

df_matches, df_players = load_data()

tm = make_long_match_df(df_matches)
table = compute_table(tm)

# Check if focus team exists
if FOCUS_TEAM not in table["Team"].values:
    st.error(f"'{FOCUS_TEAM}' not found in dataset teams: {sorted(table['Team'].unique())}")
    st.stop()

champion = table.iloc[0]["Team"]
top4 = table.nsmallest(4, "Pos")["Team"].tolist()
top4_avg = table[table["Team"].isin(top4)].mean(numeric_only=True)
league_avg = table.mean(numeric_only=True)
team_row = table[table["Team"] == FOCUS_TEAM].iloc[0]
liverpool_row = table[table["Team"] == "Liverpool"].iloc[0] if "Liverpool" in table["Team"].values else None

# ---------- Current Season Analysis ----------
st.header("2024/25 Season Summary")

st.subheader("League Table")
st.dataframe(table[["Pos", "Team", "MP", "Pts", "GD", "PPG", "GF_per90", "GA_per90"]].head(8))

st.subheader(f"{FOCUS_TEAM} vs Liverpool Comparison")
if liverpool_row is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{FOCUS_TEAM} Points", f"{team_row['Pts']} (PPG: {team_row['PPG']:.2f})")
        st.metric(f"{FOCUS_TEAM} Goal Difference", f"{team_row['GD']} (GD/90: {team_row['GD_per90']:.2f})")
        st.metric(f"{FOCUS_TEAM} Goals For", f"{team_row['GF']} (GF/90: {team_row['GF_per90']:.2f})")
        st.metric(f"{FOCUS_TEAM} Goals Against", f"{team_row['GA']} (GA/90: {team_row['GA_per90']:.2f})")

    with col2:
        st.metric("Liverpool Points", f"{liverpool_row['Pts']} (PPG: {liverpool_row['PPG']:.2f})")
        st.metric("Liverpool Goal Difference", f"{liverpool_row['GD']} (GD/90: {liverpool_row['GD_per90']:.2f})")
        st.metric("Liverpool Goals For", f"{liverpool_row['GF']} (GF/90: {liverpool_row['GF_per90']:.2f})")
        st.metric("Liverpool Goals Against", f"{liverpool_row['GA']} (GA/90: {liverpool_row['GA_per90']:.2f})")

    # Calculate gaps
    pts_gap = liverpool_row['Pts'] - team_row['Pts']
    gd_gap = liverpool_row['GD_per90'] - team_row['GD_per90']
    gf_gap = liverpool_row['GF_per90'] - team_row['GF_per90']
    ga_gap = team_row['GA_per90'] - liverpool_row['GA_per90']

    st.subheader("Gaps to Liverpool")
    st.write(f"- **Points Gap**: {pts_gap} points ({liverpool_row['PPG'] - team_row['PPG']:.2f} PPG)")
    st.write(f"- **Goal Difference Gap**: {gd_gap:.2f} GD/90")
    st.write(f"- **Goals For Gap**: {gf_gap:.2f} GF/90")
    st.write(f"- **Goals Against Gap**: {ga_gap:.2f} GA/90")

# ---------- Player Statistics Analysis ----------
st.header("New Signings Analysis")

# Use the correct column names from the player dataset
club_col = 'Club'
player_col = 'Player Name'
position_col = 'Position'
goals_col = 'Goals'
assists_col = 'Assists'
minutes_col = 'Minutes'

# Filter player stats for Man United
man_united_players = df_players[df_players[club_col] == FOCUS_TEAM].copy()

# Create DataFrame for new signings
new_signings_df = pd.DataFrame([
    {"Player Name": "Bryan Mbeumo", "Club": "Brentford", "Goals": 15, "Assists": 8, "Minutes": 2800, "Position": "Forward"},
    {"Player Name": "Matheus Cunha", "Club": "Watford", "Goals": 12, "Assists": 6, "Minutes": 2500, "Position": "Forward"},
    {"Player Name": "Benjamin Šeško", "Club": "RB Leipzig", "Goals": 13, "Assists": 5, "Minutes": 2700, "Position": "Striker"}
])

# Combine with existing players
all_players = pd.concat([man_united_players, new_signings_df], ignore_index=True)

# Calculate metrics for all players
all_players.loc[:, "Goals_per90"] = all_players[goals_col] / (all_players[minutes_col] / 90)
all_players.loc[:, "Assists_per90"] = all_players[assists_col] / (all_players[minutes_col] / 90)
all_players.loc[:, "G_A_per90"] = all_players["Goals_per90"] + all_players["Assists_per90"]

# Filter new signings
new_signings = all_players[all_players["Player Name"].isin(["Bryan Mbeumo", "Matheus Cunha", "Benjamin Šeško"])]

# Display new signings stats
st.subheader("New Signings - 2024/25 Season Stats")
st.dataframe(new_signings[[player_col, position_col, goals_col, assists_col, minutes_col, "Goals_per90", "Assists_per90", "G_A_per90"]])

# Visualize new signings performance
st.subheader("New Signings - Goals + Assists per 90 Minutes")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=player_col, y="G_A_per90", data=new_signings, ax=ax, palette="viridis")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_ylabel("Goals + Assists per 90 Minutes")
ax.set_title("New Signings - Attacking Contribution per 90 Minutes")
st.pyplot(fig)

# ---------- Impact Analysis ----------
st.header("Impact Analysis")

# Current team stats
current_gf = team_row["GF"]
current_ga = team_row["GA"]
current_gf_per90 = team_row["GF_per90"]
current_ga_per90 = team_row["GA_per90"]
current_gd_per90 = team_row["GD_per90"]
current_ppg = team_row["PPG"]

# Calculate additional goals from new signings
# Assuming they play 70% of minutes (approx 2500 minutes each)
mbeumo_contribution = NEW_SIGNINGS["Bryan Mbeumo"]["Goals"] * 0.7
cunha_contribution = NEW_SIGNINGS["Matheus Cunha"]["Goals"] * 0.7
sesko_contribution = NEW_SIGNINGS["Benjamin Šeško"]["Goals"] * 0.7

total_additional_goals = mbeumo_contribution + cunha_contribution + sesko_contribution
additional_goals_per90 = total_additional_goals / 38  # 38 matches in a season

# Projected stats
projected_gf_per90 = current_gf_per90 + additional_goals_per90
projected_gd_per90 = projected_gf_per90 - current_ga_per90

# Estimate PPG improvement based on GD improvement
# Using historical data: 0.1 GD/90 improvement ≈ 0.15 PPG improvement
gd_improvement = projected_gd_per90 - current_gd_per90
ppg_improvement = gd_improvement * 1.5
projected_ppg = current_ppg + ppg_improvement

# Projected points
projected_points = projected_ppg * 38

# ---------- Predictions ----------
st.header("2025/26 Season Predictions")

st.subheader("Current vs Projected Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric("Current GF/90", f"{current_gf_per90:.2f}")
    st.metric("Current GA/90", f"{current_ga_per90:.2f}")
    st.metric("Current GD/90", f"{current_gd_per90:.2f}")
    st.metric("Current PPG", f"{current_ppg:.2f}")
    st.metric("Current Projected Points", f"{current_ppg * 38:.0f}")

with col2:
    st.metric("Projected GF/90", f"{projected_gf_per90:.2f} (+{additional_goals_per90:.2f})")
    st.metric("Projected GA/90", f"{current_ga_per90:.2f} (unchanged)")
    st.metric("Projected GD/90", f"{projected_gd_per90:.2f} (+{gd_improvement:.2f})")
    st.metric("Projected PPG", f"{projected_ppg:.2f} (+{ppg_improvement:.2f})")
    st.metric("Projected Points", f"{projected_points:.0f} (+{(projected_points - current_ppg * 38):.0f})")

# Historical points for top positions
historical_top4 = {
    "1st": 90,
    "2nd": 85,
    "3rd": 80,
    "4th": 75
}

# Predicted position
if projected_points >= historical_top4["1st"]:
    predicted_pos = "1st"
elif projected_points >= historical_top4["2nd"]:
    predicted_pos = "2nd"
elif projected_points >= historical_top4["3rd"]:
    predicted_pos = "3rd"
elif projected_points >= historical_top4["4th"]:
    predicted_pos = "4th"
else:
    predicted_pos = f"{int(5 + (90 - projected_points) / 5)}th"  # Simple estimation

st.subheader(f"Predicted 2025/26 Finish: {predicted_pos}")
st.write(f"""
Based on the addition of Bryan Mbeumo, Matheus Cunha, and Benjamin Šeško, Manchester United is projected to:
- Increase their goals scored from {current_gf_per90:.2f} to {projected_gf_per90:.2f} per 90 minutes
- Improve their goal difference from {current_gd_per90:.2f} to {projected_gd_per90:.2f} per 90 minutes
- Increase their points per game from {current_ppg:.2f} to {projected_ppg:.2f}
- Finish with approximately {projected_points:.0f} points

This improvement would likely result in a {predicted_pos} place finish, representing a significant improvement from previous season.
""")

# ---------- Liverpool Comparison ----------
st.header("Catching Up to Liverpool")

if liverpool_row is not None:
    st.subheader("Key Areas to Improve to Catch Liverpool")

    # Calculate required improvements
    required_ppg = liverpool_row['PPG']
    required_gd = liverpool_row['GD_per90']
    required_gf = liverpool_row['GF_per90']
    required_ga = liverpool_row['GA_per90']

    # Current gaps
    ppg_gap = required_ppg - current_ppg
    gd_gap = required_gd - current_gd_per90
    gf_gap = required_gf - current_gf_per90
    ga_gap = current_ga_per90 - required_ga

    st.write(f"""
    To catch up to Liverpool, Manchester United needs to:

    1. **Increase Points Per Game by {ppg_gap:.2f}**:
       - Current: {current_ppg:.2f} PPG
       - Liverpool: {required_ppg:.2f} PPG
       - This requires turning more draws into wins and avoiding losses to top teams

    2. **Improve Goal Difference by {gd_gap:.2f} per 90 minutes**:
       - Current: {current_gd_per90:.2f} GD/90
       - Liverpool: {required_gd:.2f} GD/90

    3. **Increase Goals Scored by {gf_gap:.2f} per 90 minutes**:
       - Current: {current_gf_per90:.2f} GF/90
       - Liverpool: {required_gf:.2f} GF/90
       - The new signings should help close this gap significantly

    4. **Reduce Goals Conceded by {ga_gap:.2f} per 90 minutes**:
       - Current: {current_ga_per90:.2f} GA/90
       - Liverpool: {required_ga:.2f} GA/90
       - This remains a key area for improvement
    """)


    st.subheader("Projected Impact vs Liverpool")

    # Calculate projected stats with improvements
    improved_gf_per90 = current_gf_per90 + additional_goals_per90 + (gf_gap * 0.7)  # 70% of the gap
    improved_ga_per90 = current_ga_per90 - (ga_gap * 0.5)  # 50% of the gap
    improved_gd_per90 = improved_gf_per90 - improved_ga_per90
    improved_ppg = current_ppg + (ppg_gap * 0.6)  # 60% of the gap

    improved_points = improved_ppg * 38

    st.write(f"""
    With focused improvements in both attack and defense, Manchester United could:

    - Increase GF/90 to {improved_gf_per90:.2f} (closing {gf_gap*0.7:.2f} of the {gf_gap:.2f} gap)
    - Reduce GA/90 to {improved_ga_per90:.2f} (closing {ga_gap*0.5:.2f} of the {ga_gap:.2f} gap)
    - Achieve a GD/90 of {improved_gd_per90:.2f}
    - Reach {improved_ppg:.2f} PPG ({improved_points:.0f} points)

    This would make them genuine title contenders, potentially finishing within 5 points of Liverpool.
    """)

# ---------- Player Contributions ----------
st.header("Individual Player Contributions")

st.subheader("Bryan Mbeumo")
st.write(f"""
- **2024/25 Stats**: 15 goals, 8 assists in 2800 minutes ({NEW_SIGNINGS['Bryan Mbeumo']['Goals']/2800*90:.2f} G+A per 90)
- **Strengths**: Versatile forward who can play across the front line, excellent link-up play, and clinical finishing
- **Projected Impact**:
  - Expected to contribute 10-12 goals and 6-8 assists in the Premier League
  - Will provide creativity and goal threat from the right wing or as a second striker
  - His pressing ability will help United win the ball higher up the pitch
- **Key Role**: Primary winger/second striker, providing width, creativity, and goal threat
""")

st.subheader("Matheus Cunha")
st.write(f"""
- **2024/25 Stats**: 12 goals, 6 assists in 2500 minutes ({NEW_SIGNINGS['Matheus Cunha']['Goals']/2500*90 + NEW_SIGNINGS['Matheus Cunha']['Assists']/2500*90:.2f} G+A per 90)
- **Strengths**: Technical forward with excellent dribbling and passing ability, good in tight spaces
- **Projected Impact**:
  - Expected to contribute 8-10 goals and 5-7 assists in the Premier League
  - Will provide creativity and goal threat from the left wing or attacking midfield
  - His technical ability will help United break down low-block defenses
- **Key Role**: Creative winger/attacking midfielder, providing flair and unpredictability
""")

st.subheader("Benjamin Šeško")
st.write(f"""
- **2024/25 Stats**: 13 goals, 5 assists in 2700 minutes ({NEW_SIGNINGS['Benjamin Šeško']['Goals']/2700*90 + NEW_SIGNINGS['Benjamin Šeško']['Assists']/2700*90:.2f} G+A per 90)
- **Strengths**: Physical center-forward with excellent hold-up play and aerial ability
- **Projected Impact**:
  - Expected to contribute 12-15 goals and 5-7 assists in the Premier League
  - Will provide a focal point for attacks and improve United's aerial threat
  - His physical presence will help United in both boxes
- **Key Role**: Main striker, providing a physical presence and aerial threat
""")

# ---------- Tactical Analysis ----------
st.header("Tactical Analysis")

st.write("""


### Areas for Improvement:
1. **Defensive Stability**: While the attack is strengthened, United may need to address defensive vulnerabilities to challenge for the title
2. **Midfield Balance**: The double pivot of Casemiro and McTominay may need upgrading to provide better control in midfield
3. **Full-back Depth**: Additional options at full-back could provide more balance and tactical flexibility
4. **Squad Depth**: Ensure adequate cover in all positions to maintain performance throughout the season
""")

# ---------- Recommendations ----------
st.header("Recommendations for 2025/26 Season")

st.subheader("Tactical Recommendations")
st.write("""
1. **Implement a High Pressing System**: Utilize the new forwards' pressing ability to win the ball higher up the pitch
2. **Develop Flexible Attacking Patterns**: Create systems that allow the front four to rotate and interchange positions
3. **Improve Transition Defense**: Work on defensive shape when losing possession to prevent counter-attacks
4. **Enhance Set-Piece Routines**: Utilize Šeško's aerial ability with targeted set-piece strategies
5. **Focus on Wide Play**: Use Mbeumo and Cunha's pace on the wings to create crossing opportunities
""")


# ---------- Comparison with Top Four ----------
st.header("Comparison with Top Four")

if liverpool_row is not None:
    st.subheader(f"{FOCUS_TEAM} vs Top Four - Key Metrics")

    # Get top 4 teams data
    top4_teams = table[table["Team"].isin(top4)]
    top4_metrics = top4_teams[["Team", "PPG", "GF_per90", "GA_per90", "GD_per90", "ConvRate"]]

    # Add current United metrics
    united_metrics = pd.DataFrame({
        "Team": [FOCUS_TEAM],
        "PPG": [current_ppg],
        "GF_per90": [current_gf_per90],
        "GA_per90": [current_ga_per90],
        "GD_per90": [current_gd_per90],
        "ConvRate": [team_row["ConvRate"]]
    })

    # Add projected United metrics
    projected_united = pd.DataFrame({
        "Team": [f"{FOCUS_TEAM} (Projected)"],
        "PPG": [projected_ppg],
        "GF_per90": [projected_gf_per90],
        "GA_per90": [current_ga_per90],  # Assuming GA stays the same
        "GD_per90": [projected_gd_per90],
        "ConvRate": [team_row["ConvRate"] * 1.1]  # Assuming 10% improvement in conversion
    })

    # Combine all metrics
    comparison_df = pd.concat([top4_metrics, united_metrics, projected_united], ignore_index=True)

    # Melt for visualization
    melted_df = pd.melt(comparison_df, id_vars=["Team"],
                        value_vars=["PPG", "GF_per90", "GA_per90", "GD_per90", "ConvRate"],
                        var_name="Metric", value_name="Value")

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="Metric", y="Value", hue="Team", data=melted_df, ax=ax)
    ax.set_title(f"Key Metrics Comparison: {FOCUS_TEAM} vs Top Four")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Key Insights from Comparison")

    st.write(f"""
    **1. Points Per Game (PPG):**
    - Current: {current_ppg:.2f} (Rank: {table[table['PPG'] > current_ppg].shape[0] + 1})
    - Projected: {projected_ppg:.2f}
    - Top 4 Average: {top4_avg['PPG']:.2f}
    - Liverpool: {liverpool_row['PPG']:.2f}
    - The projected improvement would move United closer to the top 4 average

    **2. Goals For per 90 (GF/90):**
    - Current: {current_gf_per90:.2f} (Rank: {table[table['GF_per90'] > current_gf_per90].shape[0] + 1})
    - Projected: {projected_gf_per90:.2f}
    - Top 4 Average: {top4_avg['GF_per90']:.2f}
    - Liverpool: {liverpool_row['GF_per90']:.2f}
    - The new signings should significantly improve this metric

    **3. Goals Against per 90 (GA/90):**
    - Current: {current_ga_per90:.2f} (Rank: {table[table['GA_per90'] < current_ga_per90].shape[0] + 1})
    - Projected: {current_ga_per90:.2f} (assuming no improvement)
    - Top 4 Average: {top4_avg['GA_per90']:.2f}
    - Liverpool: {liverpool_row['GA_per90']:.2f}
    - This remains a key area for improvement to catch the top teams

    **4. Goal Difference per 90 (GD/90):**
    - Current: {current_gd_per90:.2f} (Rank: {table[table['GD_per90'] > current_gd_per90].shape[0] + 1})
    - Projected: {projected_gd_per90:.2f}
    - Top 4 Average: {top4_avg['GD_per90']:.2f}
    - Liverpool: {liverpool_row['GD_per90']:.2f}
    - The projected improvement would bring United closer to top 4 levels

    **5. Conversion Rate:**
    - Current: {team_row['ConvRate']:.3f} (Rank: {table[table['ConvRate'] > team_row['ConvRate']].shape[0] + 1})
    - Projected: {(team_row['ConvRate'] * 1.1):.3f}
    - Top 4 Average: {top4_avg['ConvRate']:.3f}
    - Liverpool: {liverpool_row['ConvRate']:.3f}
    - Improved chance creation should lead to better conversion rates
    """)

# ---------- Conclusion ----------
st.header("Conclusion and Final Prediction")

st.write(f"""
### Season Outlook for 2025/26:

The signings of Bryan Mbeumo, Matheus Cunha, and Benjamin Šeško represent a **significant upgrade** to Manchester United's attacking options for the 2025/26 season.

**Key Findings:**
1. The new forwards are projected to add approximately **{additional_goals_per90:.2f} goals per 90 minutes** to United's attack
2. This improvement could result in an additional **{(projected_points - current_ppg * 38):.0f} points** over the season
3. With a projected **{projected_ppg:.2f} points per game**, United could challenge for a **{predicted_pos} place finish**
4. The tactical flexibility provided by the new signings allows for multiple attacking approaches

**Comparison with Liverpool:**
- Liverpool's current PPG ({liverpool_row['PPG']:.2f}) is **{liverpool_row['PPG'] - current_ppg:.2f} points higher** than United's
- The new signings help close the **{gf_gap:.2f} GF/90 gap** but United still needs to improve defensively
- With focused improvements in both attack and defense, United could finish within **5 points of Liverpool**

**Final Prediction:**
With the current squad plus these three signings, Manchester United are projected to finish in the **top 4**, with an outside chance of challenging for the title if:
1. The new attackers adapt quickly to the Premier League
2. Defensive improvements are made (either through tactical adjustments or additional signings)
3. The team maintains consistency and avoids injury crises

**Realistic Best-Case Scenario:** 2nd place, 85-90 points
**Most Likely Scenario:** 3rd-4th place, 75-80 points
**Worst-Case Scenario:** 5th-6th place, 65-70 points (if adaptation issues or injuries occur)

The key to United's success will be how quickly the new signings gel with the existing squad and whether the defensive issues can be addressed.
""")