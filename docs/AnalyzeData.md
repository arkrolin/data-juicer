# Data Analysis

Before deciding on filter thresholds, it helps to understand the statistical profile of your dataset. `dj-analyze` computes distributions and correlations for all operator-produced statistics, enabling data-driven threshold decisions.

> For the full parameter list, see [Global Configuration Reference](GlobalConfig.md).

---

## CLI

### Basic Usage

Run the analyzer with an existing recipe:

```bash
dj-analyze --config path/to/your-recipe.yaml
```

### Auto Mode

No dedicated analysis recipe needed—automatically uses all stats-producing Filters on a dataset subset:

```bash
dj-analyze --auto --dataset_path your-dataset.jsonl [--auto_num 1000]
```

- `--auto_num`: Number of samples to analyze (default 1000). Good for quick distribution overview.

---

## Python API

### Basic: Load Config and Run

```python
from data_juicer.config import init_configs
from data_juicer.core import Analyzer

cfg = init_configs(args=['--config', 'my-recipe.yaml'])
analyzer = Analyzer(cfg)
dataset = analyzer.run()

# Access results
print(analyzer.overall_result)
```

### Analyze an Existing Dataset

```python
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs

cfg = init_configs(args=[
    '--config', 'my-recipe.yaml',
    '--export_path', './analysis-output/stats.jsonl',
])
analyzer = Analyzer(cfg)

dataset = NestedDataset.from_json('my-data.jsonl')
analyzed = analyzer.run(dataset=dataset)
```

### Compute Stats Only (No Export)

Useful for programmatic decisions without writing files:

```python
analyzed = analyzer.run(dataset=dataset, skip_export=True)

# Access computed stats
stats = analyzed['stats']
avg_len = sum(s.get('text_length', 0) for s in stats) / len(stats)
print(f"Average text length: {avg_len:.0f}")
```

### Manual Analysis Pipeline

For full control over the analysis logic, use the underlying components directly:

```python
from data_juicer.ops.filter import LanguageIDScoreFilter, TextLengthFilter
from data_juicer.analysis import OverallAnalysis, ColumnWiseAnalysis
from data_juicer.core import NestedDataset

dataset = NestedDataset.from_json('my-data.jsonl')

# Compute stats only (disable filtering by nulling process)
filters = [
    TextLengthFilter(min_len=0, max_len=999999),
    LanguageIDScoreFilter(lang='en', min_score=0.0),
]

for f in filters:
    dataset = f.run(dataset=dataset, reduce=False)  # compute stats only, no filtering

# Run analysis
output_dir = './my-analysis'
overall = OverallAnalysis(dataset, output_dir)
result = overall.analyze()
print(result)

column_wise = ColumnWiseAnalysis(dataset, output_dir, overall_result=result)
column_wise.analyze()
```

### Dynamic Analysis Dimensions

Programmatically choose analysis operators based on data modality—useful for automated pipelines:

```python
from data_juicer.core import Analyzer, NestedDataset
from data_juicer.config import init_configs

dataset = NestedDataset.from_json('input.jsonl')
sample = dataset[0]

# Build analysis config based on data modality
process_config = []

# Text statistics
if 'text' in sample:
    process_config.extend([
        {'text_length_filter': {'min_len': 0, 'max_len': 999999}},
        {'language_id_score_filter': {'lang': 'en', 'min_score': 0.0}},
        {'alphanumeric_filter': {'min_ratio': 0.0}},
    ])

# Image statistics
if 'images' in sample and sample['images']:
    process_config.extend([
        {'image_shape_filter': {'min_width': 0, 'min_height': 0}},
        {'image_aspect_ratio_filter': {'min_ratio': 0.0, 'max_ratio': 999}},
    ])

cfg = init_configs(args=[
    '--auto',
    '--dataset_path', 'input.jsonl',
    '--export_path', './analysis/stats.jsonl',
], allow_auto=True)
cfg.process = process_config

analyzer = Analyzer(cfg)
analyzed = analyzer.run(dataset=dataset)
```

---

## Analysis Output

The analyzer produces:

- **Overall statistics table**: count, mean, std, min, max for each metric
- **Distribution plots**: histogram for each metric
- **Correlation analysis**: heatmap of metric correlations

Output is saved under the `analysis/` subdirectory of `export_path`.

---

## Which Operators Participate

The Analyzer processes two types of operators:
- **Filter operators** that produce stats in the `stats` field (most Filters do)
- **Tagging operators** that produce labels in the `meta` field

Registry markers:
- `NON_STATS_FILTERS`: Filters that do NOT produce stats
- `TAGGING_OPS`: Operators that produce tags

---

## Distributed Analysis

Set `executor_type: ray` to use `RayAnalyzer` with native Ray aggregation:

```bash
dj-analyze --config demos/analyze_simple/ray_analyzer.yaml
```

> RayAnalyzer does not produce per-column distribution plots or correlation analysis. See [Distributed Processing](Distributed.md).

---

## Font Configuration

If distribution plots show "Glyph missing" warnings:

```bash
export ANALYZER_FONT="Heiti SC"  # default; supports CJK characters
```

---

## Next Steps

- Adjust thresholds interactively? Use [Web Playground](Playground.md)
- Ready to process? See [Processing Data](ProcessData.md)
- Large-scale analysis? See [Distributed Processing](Distributed.md)
