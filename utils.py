"""
Shared utility mappings for classification and EDA.

This module contains constant mappings for ordinal encoding of categorical
features used across the Stack Overflow survey analysis:
- AGE_MAP: Maps age range strings to numeric midpoint values
- ED_MAP: Maps education level strings to ordinal rankings (1-7)

These mappings enable consistent feature encoding between preprocessing,
classification, and exploratory data analysis.
"""
from typing import Dict

# Mappe condivise (da classification & EDA)
AGE_MAP: Dict[str, int] = {
    "18-24 years old": 21,
    "25-34 years old": 30,
    "35-44 years old": 40,
    "45-54 years old": 50,
    "55-64 years old": 60,
    "65 years or older": 70,
}

ED_MAP: Dict[str, int] = {
    "Primary/elementary school": 1,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
    "Some college/university study without earning a degree": 3,
    "Associate degree (A.A., A.S., etc.)": 4,
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 5,
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
    "Other (please specify):": 3,
}