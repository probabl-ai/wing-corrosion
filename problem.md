The problem
Corrosion is a progressive phenomenon that depends heavily on the environment an aircraft is exposed to on the ground: humidity, sea salt, pollutants, temperature, time spent parked, and more. Your mission: predict an aircraft's corrosion risk from its environmental exposure history.

What you have to do
For every aircraft and every month in the test set, estimate a corrosion probability (corrosion_risk), between 0 and 1.

The data
You are given three files:

environment_training.csv
The monthly environmental history of 758 aircraft (delivered between 2015 and 2024). One row = one aircraft (aircraft_id) in a given month (year_month), with around thirty variables: weather (temperature, humidity), aerosols (sea salt, dust, sulfates…), gaseous pollutants (ozone, NOx, SO₂…) and parking time. This is your raw material for training a model.

corrosions_training.csv
For those same training aircraft, the date of first observed corrosion (observation_date). You build your target from these dates (see "How to build the target" below).

environment_test.csv
The same monthly environmental history, for 142 aircraft delivered in 2014. These are the aircraft you must predict on. You do not have their observation dates — that's the whole challenge.

How to build the target (corrosion_risk)
The official convention, set by Airbus, relies on two reference points per aircraft:

the month of the corrosion observation → corrosion_risk = 1;
the month exactly 24 months earlier → corrosion_risk = 0 (assumption: no corrosion two years before the observation, reflecting the non-linear nature of the phenomenon).
On the training set, you can therefore reconstruct these (1 / 0) pairs from corrosions_training.csv and train your model to tell a "healthy" environment apart from an "at-risk" one.

Submission
You predict a corrosion_risk for every row of environment_test.csv (each aircraft × month). Your submission file must contain the same rows as sample_submission.csv, with two columns:

id,corrosion_risk
894378_2018-08,0.87
894378_2016-08,0.04
...
id: the row identifier, in the format aircraft_id_year_month (same as in sample_submission.csv).
corrosion_risk: your predicted probability (between 0 and 1).
You don't know which months are actually scored, so predict a calibrated probability for the whole monthly history of each aircraft.

Evaluation
Submissions are scored with the Brier Score (mean squared error between the predicted probability and the actual 0/1 outcome):


The score is computed only on the reference points described above: for each test aircraft, the month of corrosion observation (true value = 1) and the month 24 months earlier (true value = 0). All other rows of your submission are required for the format but are ignored when scoring.

The lower the score, the better. A constant prediction of 0.5 gives 0.25 — that's the baseline to beat.

Public leaderboard: computed continuously on half of the reference points during the challenge.
Private leaderboard: computed on the other half, revealed at the close — this is what determines the final ranking. Be careful not to overfit the public score.
Tips to get started
Reconstruct the (1 / 0) pairs of the training set from the observation dates.
Match each date to its environmental features in environment_training.csv.
Train a classifier that returns a probability (not just 0 or 1) — the Brier score rewards well-calibrated probabilities.
Apply your model to every row of environment_test.csv and submit the full file.