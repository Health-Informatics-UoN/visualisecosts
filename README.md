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
plot-costs <config_file> [projection_end_date] [year_start_month]
```

Or run directly:
```bash
python plot_costs.py <config_file> [projection_end_date] [year_start_month]
```

### Arguments

- `config_file`: Path to a text file containing folder paths (one per line)
- `projection_end_date` (optional): End date for projections (format: YYYY-MM-DD)
- `year_start_month` (optional): Month when financial year starts (1-12, default: 4 for April)

### Config File Format

The config file should contain one folder path per line, optionally followed by settings. Empty lines and lines starting with `#` are ignored.

Format: `folder_path [projection_end_date] [year_start_month]`

- Settings in the config file override command-line defaults
- You can specify just `year_start_month`, or both `projection_end_date` and `year_start_month`
- If settings are not specified for a folder, the command-line defaults are used

Example `folders.txt`:
```
/path/to/folder1
/path/to/folder2 2027-12-31 7
/path/to/folder3 7
/path/to/folder4 2028-06-30
# This is a comment
/path/to/folder5
```

In this example:
- `folder1` and `folder5` use command-line defaults
- `folder2` uses custom projection end date (2027-12-31) and year start month (7)
- `folder3` uses custom year start month (7) with default projection end date
- `folder4` uses custom projection end date (2028-06-30) with default year start month

### Examples

```bash
# Process multiple folders with default settings (April-March financial year)
plot-costs folders.txt

# Specify financial year start month (July-June)
plot-costs folders.txt 7

# Specify projection end date and financial year start
plot-costs folders.txt 2027-12-31 7
```

The script will process each folder in sequence, generating outputs for each folder. If one folder fails, it will continue processing the remaining folders.

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

