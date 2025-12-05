# Label Encoding Mapping

This document explains the integer encoding used for categorical variables in the processed dataset.

## Gender
- **0**: Female
- **1**: Male

## Occupation
**One-Hot Encoded**: The 'Occupation' column has been transformed into multiple binary columns (e.g., `Occupation_Engineer`, `Occupation_Doctor`, etc.) where 1 indicates the presence of the occupation and 0 indicates absence.

## BMI Category
**Ordinal Encoded**:
- **0**: Normal / Normal Weight
- **1**: Overweight
- **2**: Obese

## Sleep Disorder
**Target Variable (Integer Encoded)**:
- **0**: None
- **1**: Insomnia
- **2**: Sleep Apnea
