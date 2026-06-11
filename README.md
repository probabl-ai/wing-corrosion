# wing corrosion predictive model for maintenance
IBM x Airbus Hackathon on wing corrosion dataset ✈

https://www.kaggle.com/competitions/haks-airbus-x-ibm-x-aws-2026

## Context: Why this problem matters

Airbus aircraft are corroding much faster than expected, causing significant dissatisfaction among airlines. Aircraft undergo scheduled maintenance visits ("C-checks", e.g., after 6 years) where parts are repaired or replaced — but corrosion problems are appearing *before* these scheduled visits, meaning airlines discover issues during operations. Severe corrosion on wings is extremely costly to repair, requiring corroded sections to be cut out and replaced with plates.

A striking real-world example: during COVID, parked aircraft corroded so badly that some landing gear collapsed. Airbus discovered that corrosivity varies enormously between airports — some (e.g., desert locations) are 10x less corrosive than average, others 6x more, meaning a **factor of ~60 between the least and most corrosive airports**. By providing airlines with a corrosivity map so they could park aircraft at less corrosive airports (allowing mandatory $50k inspections to be stretched from every 12 weeks to every 24 weeks), Airbus saved airlines roughly **€400 million** — a concrete proof that predicting corrosion creates value.

## The task

The goal is essentially a "digital twin" of wing degradation: estimating the state of corrosion of an aircraft's wings at any point in time.

Concretely, you must **add a column to the test files** containing, for each aircraft and each month, the **probability (0–100%) that corrosion would be found if the aircraft were inspected** at that time.

- **Training data**: aircraft from 2014 onwards, with both their monthly environmental data and observed corrosion events.
- **Test data**: only aircraft **delivered in 2014** (all "born" in 2014), with environmental data but *without* the corrosion observation file.

Important domain knowledge: corrosion is **non-linear** — once it starts, it progresses exponentially fast.

## The datasets

All data is anonymized (aircraft identities belong to airlines, not Airbus). The sources are:

1. **Corrosion events** from two channels:
   - **Logbooks**: maintenance crews walk around the aircraft, note corrosion in the aircraft's logbook. These belong to airlines; some share them, others don't. Most corrosion events come from here.
   - **Tech requests**: when an airline finds corrosion, they contact Airbus support (often in a panic) asking what to do — these requests also flag corrosion events.

2. **Environmental data (Copernicus)**: via Airbus Defence and Space, the European Copernicus satellite program provides global air-quality data on a ~7 km grid, with about 50 "contaminants" (salt, pollution, etc.).

3. **METAR data**: meteorological sensors located at airports, providing humidity, wind, temperature, visibility — very precise since the sensors are on-site.

4. **Flight Radar data**: purchased from a private company, refreshed roughly every minute, with history back to 2013, used to know **where and when each aircraft is on the ground**.

Key insight: aircraft spend roughly two-thirds of their time parked on the ground (especially short-haul). In flight, all aircraft share essentially the same environment; it's *on the ground* that environments differ (near the sea, polluted areas, etc.). So Airbus computes the environment only for ground time (stays of at least ~20 minutes), and you'll receive, per aircraft, the **monthly average ground environment** plus the **duration spent on the ground** that month (some aircraft are parked the whole month, others fly a lot).

## Scoring

The metric is the **Brier score** — lowest score wins. Evaluation works as follows: Airbus knows the actual corrosion observation dates for the test aircraft (~150 aircraft, two evaluation dates each, ~300 samples). For each aircraft they check your predicted probability at:

- **The corrosion observation date** → expected to be close to **100%**
- **Two years before that date** → expected to be close to **0%** (since corrosion develops quickly, it wasn't present 2 years earlier)

You're not forced to output only 0s and 100s — intermediate probabilities are allowed and often wiser, because a confident wrong answer (predicting 0 when 100 was expected) is heavily penalized by the Brier score. Calibrating your risk estimates is part of the challenge.

## Evaluation beyond the score: the pitch

You'll also present a **pitch** explaining your approach. The presenter cares not only about methodology but about **what you learned about corrosion** and how you explain your results — he explicitly hopes to be surprised and to learn something Airbus hasn't discovered yet.

## The value lesson

Airlines initially rejected the idea of a corrosion prediction model ("I don't want predictions, I want *no corrosion* — a prediction just forces costly maintenance on me"). Airbus's answer leveraged the fact that aircraft fly repetitive route patterns, so their environment is roughly stable over their lifetime. The proposed solution: **swap aircraft between routes** so that fast-corroding aircraft are moved to gentler environments, ideally ensuring all aircraft reach corrosion only at end-of-life. The takeaway the presenter insists on: prediction itself solves problems and makes money — *"knowledge has value."*