# Experiment 01: Baseline

**Status:** approved  
**Date:** 2026-06-11

## Goal

Establish a baseline for predicting aircraft corrosion risk from monthly environmental exposure data. Use all available environmental features (weather, aerosols, pollutants, parking time) with a simple gradient boosting classifier.

## Approach

1. **Target construction**: Build binary labels from corrosion observation dates
   - Positive samples (y=1): month of corrosion observation
   - Negative samples (y=0): exactly 24 months before observation
   
2. **Features**: Use all ~30 environmental variables as-is:
   - Weather: temperature, humidity
   - Aerosols: sea salt, dust, sulfates
   - Gaseous pollutants: ozone, NOx, SO₂
   - Parking time

3. **Model**: HistGradientBoostingClassifier (sklearn's native gradient boosting)
   - No hyperparameter tuning yet
   - Default parameters

4. **Evaluation**: 5-fold cross-validation with Brier score
   - Brier score measures calibration quality (lower is better)
   - Baseline to beat: 0.25 (constant prediction of 0.5)

## Acceptance criteria

- [ ] Pipeline builds successfully with skrub DataOps graph
- [ ] Smoke test passes (predict-time row count matches)
- [ ] Cross-validation completes without errors
- [ ] Brier score < 0.25 (beats naive baseline)
- [ ] Report persisted to Skore Hub workspace `debray.yann`

## Implementation notes

- Aircraft ID and year_month kept as metadata columns (not features)
- No feature engineering in baseline - use raw environmental data
- No handling of missing values yet - let HistGradientBoosting handle natively