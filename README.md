# Visualise Costs

A Python tool to visualize and project cost data from Excel spreadsheets with cumulative graphs and financial year tables.

## Installation

Using `uv`:
```bash
uv sync
```

Or install in development mode:
```bash
uv pip install -e .
```

## Usage

After installation, run:
```bash
plot-costs <directory> [projection_end_date] [year_start_month]
```

Or run directly:
```bash
python plot_costs.py <directory> [projection_end_date] [year_start_month]
```

### Arguments

- `directory`: Path to directory containing `.xlsx` files
- `projection_end_date` (optional): End date for projections (format: YYYY-MM-DD)
- `year_start_month` (optional): Month when financial year starts (1-12, default: 4 for April)

### Examples

```bash
# Basic usage with default settings (April-March financial year)
plot-costs /path/to/data

# Specify financial year start month (July-June)
plot-costs /path/to/data 7

# Specify projection end date and financial year start
plot-costs /path/to/data 2027-12-31 7
```

## Output

The script generates:
- `{directory_name}_staff_costs_cumulative.png` - Staff costs graph
- `{directory_name}_other_cumulative.png` - Other categories graph  
- `{directory_name}_yearly_totals_by_expenditure_category.csv` - Yearly totals table

All outputs are saved in the input directory.

## Requirements

- Python >= 3.13
- pandas
- matplotlib
- openpyxl

