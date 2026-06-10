# imputation_plan.json Schema

Path: `outputs/logs/imputation_plan.json`

## JSON Schema (draft-07)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ImputationPlan",
  "type": "object",
  "required": ["columns", "summary"],
  "properties": {
    "columns": {
      "type": "object",
      "description": "Per-column imputation entries. Keys are column names.",
      "additionalProperties": {
        "type": "object",
        "required": ["strategy", "add_missing_indicator", "missing_rate", "dtype"],
        "properties": {
          "strategy": {
            "type": "string",
            "enum": [
              "no_imputation_needed",
              "numeric_median",
              "numeric_median_plus_indicator",
              "groupwise_numeric_median_plus_indicator",
              "categorical_missing_token",
              "categorical_missing_token_plus_indicator",
              "structural_zero_plus_indicator",
              "structural_none_token_plus_indicator",
              "drop_column",
              "model_based_imputation_optional",
              "structural_none_or_zero"
            ]
          },
          "add_missing_indicator": { "type": "boolean" },
          "missing_rate":          { "type": "number", "minimum": 0, "maximum": 1 },
          "dtype":                 { "type": "string" },
          "group_col": {
            "type": ["string", "null"],
            "description": "Grouping column used for groupwise imputation; null otherwise."
          },
          "mi_upgrade_recommended": {
            "type": "boolean",
            "description": "True when MICE is preferred over median for this column."
          },
          "mechanism_label": {
            "type": "string",
            "description": "Mechanism clue label from the mechanism auditor."
          },
          "safety_warning": {
            "type": ["string", "null"],
            "description": "Non-null when strategy required a fallback or assumption."
          },
          "evidence": {
            "type": "object",
            "description": "Raw evidence fields used to select the strategy.",
            "properties": {
              "target_association_evidence":   { "type": ["object", "null"] },
              "covariate_association_evidence":{ "type": ["object", "null"] },
              "group_dependency_evidence":     { "type": ["object", "null"] },
              "structural_evidence":           { "type": ["object", "null"] },
              "geo_grouping_applied":          { "type": "boolean" },
              "geo_grouping_skipped_reason":   { "type": ["string", "null"] }
            }
          }
        }
      }
    },
    "summary": {
      "type": "object",
      "required": ["total_columns_with_missing", "target_col_excluded"],
      "properties": {
        "total_columns_with_missing":      { "type": "integer" },
        "columns_needing_missing_indicator":{ "type": "array", "items": { "type": "string" } },
        "columns_to_drop":                 { "type": "array", "items": { "type": "string" } },
        "mi_upgrade_columns":              { "type": "array", "items": { "type": "string" } },
        "target_col_excluded": {
          "type": ["string", "null"],
          "description": "Name of the target column excluded from imputation (hard guard)."
        },
        "type_skipped_columns": {
          "type": "array",
          "items": { "type": "string" },
          "description": "ID / datetime columns excluded from imputation."
        }
      }
    }
  }
}
```

## Key fields

| Field | Type | Note |
|-------|------|------|
| `strategy` | string enum | See `references/strategies.md` for full table |
| `add_missing_indicator` | boolean | Add `{col}_was_missing` binary feature before imputing |
| `group_col` | string\|null | Non-null only for `groupwise_numeric_median_plus_indicator` |
| `mi_upgrade_recommended` | boolean | Flag for full MICE upgrade (inference use cases) |
| `evidence.geo_grouping_applied` | boolean | True only when between-group variance condition was met |
| `evidence.geo_grouping_skipped_reason` | string\|null | Why geo grouping was not applied despite geo_col existing |
| `summary.target_col_excluded` | string\|null | Hard guard: target is never in `columns` |
| `summary.type_skipped_columns` | array | Columns excluded due to type guard (ID, datetime) |
