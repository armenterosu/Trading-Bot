#!/usr/bin/env python3
"""Generate visual metric tables as images from backtest CSV files."""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table
import numpy as np

def format_metric_value(metric, value):
    """Format metric values for display."""
    if value == '' or value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    
    try:
        val = float(value)
        
        # Percentage metrics
        if metric in ['CAGR', 'MaxDD', 'WinRate', 'WinRate_Long', 'WinRate_Short']:
            return f"{val*100:.2f}%"
        
        # Ratio metrics
        elif metric in ['Sharpe', 'Calmar', 'ProfitFactor']:
            return f"{val:.2f}"
        
        # Integer metrics
        elif metric in ['NumTrades', 'Trades_Closed', 'Trades_Long', 'Trades_Short', 
                       'Wins_Total', 'Losses_Total', 'Wins_Long', 'Losses_Long', 
                       'Wins_Short', 'Losses_Short', 'DD_Duration']:
            return f"{int(val)}"
        
        # Money
        elif metric in ['FinalEquity']:
            return f"${val:,.2f}"
        
        # Scientific notation for small numbers
        elif abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        
        else:
            return f"{val:.4f}"
    except:
        return str(value)

def create_metrics_table_image(csv_path, output_path):
    """Create a visual table image from metrics CSV."""
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get strategy name
    strategy_name = df[df['Metric'] == 'Strategy']['Value'].values[0]
    
    # Organize metrics into sections
    sections = {
        'Performance': ['FinalEquity', 'CAGR', 'Sharpe', 'MaxDD', 'Calmar'],
        'Risk': ['DD_Duration', 'WinRate', 'ProfitFactor', 'Expectancy'],
        'Trades Overview': ['NumTrades', 'Trades_Closed', 'Wins_Total', 'Losses_Total'],
        'Long Trades': ['Trades_Long', 'Wins_Long', 'Losses_Long', 'WinRate_Long'],
        'Short Trades': ['Trades_Short', 'Wins_Short', 'Losses_Short', 'WinRate_Short'],
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Title
    fig.suptitle(f'Backtest Metrics: {strategy_name}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Prepare data for table
    table_data = []
    colors = []
    
    # Color scheme
    header_color = '#2C3E50'
    section_color = '#34495E'
    row_colors = ['#ECF0F1', '#FFFFFF']
    
    for section_name, metrics in sections.items():
        # Section header
        table_data.append([section_name, ''])
        colors.append(section_color)
        
        # Metrics in section
        for i, metric in enumerate(metrics):
            row = df[df['Metric'] == metric]
            if not row.empty:
                value = row['Value'].values[0]
                formatted_value = format_metric_value(metric, value)
                
                # Clean metric name
                display_name = metric.replace('_', ' ')
                table_data.append([display_name, formatted_value])
                colors.append(row_colors[i % 2])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor(header_color)
        cell.set_text_props(weight='bold', color='white', fontsize=13)
    
    # Row styling
    for i, (row, color) in enumerate(zip(table_data, colors), start=1):
        for j in range(2):
            cell = table[(i, j)]
            cell.set_facecolor(color)
            
            # Section headers
            if color == section_color:
                cell.set_text_props(weight='bold', color='white', fontsize=12)
            
            # Highlight good/bad values
            if j == 1 and color not in [section_color]:
                text = cell.get_text().get_text()
                # Green for wins, red for losses
                if 'Win' in row[0] and text != 'N/A' and text != '0':
                    cell.set_text_props(color='#27AE60', weight='bold')
                elif 'Loss' in row[0] and text != 'N/A' and text != '0':
                    cell.set_text_props(color='#E74C3C', weight='bold')
    
    # Add timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 0.02, f'Generated: {timestamp}', 
             ha='center', fontsize=9, style='italic', color='gray')
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {output_path}")

def create_comparison_table(csv_files, output_path):
    """Create a comparison table for multiple strategies."""
    if len(csv_files) < 2:
        return
    
    # Read all CSVs
    strategies = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        strategy_name = df[df['Metric'] == 'Strategy']['Value'].values[0]
        
        metrics_dict = {'Strategy': strategy_name}
        for _, row in df.iterrows():
            metrics_dict[row['Metric']] = row['Value']
        strategies.append(metrics_dict)
    
    # Key metrics for comparison
    key_metrics = ['FinalEquity', 'CAGR', 'Sharpe', 'MaxDD', 'NumTrades', 
                   'Trades_Closed', 'WinRate', 'Trades_Long', 'Trades_Short']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    fig.suptitle('Strategy Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # Prepare comparison data
    table_data = []
    for metric in key_metrics:
        row = [metric.replace('_', ' ')]
        for strat in strategies:
            value = strat.get(metric, '')
            row.append(format_metric_value(metric, value))
        table_data.append(row)
    
    # Column labels
    col_labels = ['Metric'] + [s['Strategy'] for s in strategies]
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center')
    
    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(weight='bold', color='white', fontsize=12)
    
    # Rows
    row_colors = ['#ECF0F1', '#FFFFFF']
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell = table[(i+1, j)]
            cell.set_facecolor(row_colors[i % 2])
    
    # Timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.5, 0.02, f'Generated: {timestamp}', 
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {output_path}")

def main():
    """Generate all metric images."""
    metrics_dir = 'metrics'
    output_dir = 'metrics/images'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all backtest CSV files
    csv_files = glob.glob(os.path.join(metrics_dir, 'backtest_*.csv'))
    
    if not csv_files:
        print("No backtest CSV files found in metrics/")
        return
    
    print(f"Found {len(csv_files)} backtest result(s)")
    
    # Generate individual tables
    for csv_path in csv_files:
        basename = os.path.basename(csv_path).replace('.csv', '')
        output_path = os.path.join(output_dir, f'{basename}_table.png')
        create_metrics_table_image(csv_path, output_path)
    
    # Generate comparison table if multiple strategies
    if len(csv_files) > 1:
        comparison_path = os.path.join(output_dir, 'strategy_comparison.png')
        create_comparison_table(csv_files, comparison_path)
    
    print(f"\n✓ All images saved to: {output_dir}/")

if __name__ == '__main__':
    main()
