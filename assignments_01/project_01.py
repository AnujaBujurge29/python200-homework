from pathlib import Path
from prefect import task, get_run_logger, flow
from scipy import stats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.system('cls')

data_dir = "https://raw.githubusercontent.com/Code-the-Dream-School/python-200/main/assignments/resources/happiness_project"
output_dir = Path("assignments_01/outputs")
YEARS = list(range(2015, 2025))

output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------
#                    Task 1:  Load Multiple Years of Data
# --------------------------------------------------------------------------------


@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    all_data = []

    for year in YEARS:
        file_path = f"{data_dir}/world_happiness_{year}.csv"
        # logger.info(f"Loading {file_path}")

        df = pd.read_csv(
            file_path,
            sep=";",
            encoding="utf-8",
            decimal=","
        )

        # Add Year Column
        df['Year'] = year

        # standardize Happiness score column name and Ladder score col name to "Ladder Score"
        if "Happiness score" in df.columns:
            df = df.rename(columns={'Happiness score': 'Ladder score'})

        all_data.append(df)
        # logger.info(f"Year {year}: Loaded {len(df)} countries")

    # merge all years into a single dataframe
    merged_df = pd.concat(all_data, ignore_index=True)

    # Clean col names - remove extra whitespace
    merged_df.columns = merged_df.columns.str.strip()

    # Save merged data
    output_path = output_dir/"merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Total Records: {len(merged_df)}")
    logger.info(f"Columns: {list(merged_df.columns)}")

    return merged_df

# --------------------------------------------------------------------------------
#                    Task 2: Descriptive Statistics
# --------------------------------------------------------------------------------


@task
def compute_descriptive_statistics(df):
    logger = get_run_logger()

    # mean, median and standard deviation for happinesss score/ ladder score
    happiness_col = 'Ladder score'

    mean_happiness = df[happiness_col].mean()
    med_happiness = df[happiness_col].median()
    std_happiness = df[happiness_col].std()

    logger.info("-" * 60)
    logger.info("Happiness Score Statistics")
    logger.info(f"Mean Happiness Score: {mean_happiness:.4f}")
    logger.info(f"Median Happiness Score: {med_happiness:.4f}")
    logger.info(f"Standard Deviation: {std_happiness:.4f}")

    # Mean Happiness grouped by year
    logger.info("-" * 60)
    logger.info("Mean Happiness Score Grouped by Year")
    logger.info("-" * 60)
    mean_by_year = df.groupby('Year')[happiness_col].mean().sort_index()
    for year, mean_val in mean_by_year.items():
        logger.info(f"Year {year}: {mean_val:.4f}")

    # Mean Happiness grouped by region
    logger.info("-" * 60)
    logger.info("Mean Happiness Score grouped by Region")
    logger.info("-" * 60)
    mean_by_region = df.groupby("Regional indicator")[
        happiness_col].mean().sort_values(ascending=False)
    for region, mean_val in mean_by_region.items():
        logger.info(f"{region}: {mean_val:.4f}")

    return {
        'mean': mean_happiness,
        'median': med_happiness,
        'std': std_happiness,
        'by_year': mean_by_year.to_dict(),
        'by_region': mean_by_region.to_dict()
    }

# --------------------------------------------------------------------------------
#                    Task 3: Visual Exploration
# --------------------------------------------------------------------------------


@task
def create_visualization(df: pd.DataFrame) -> None:
    logger = get_run_logger()

    happiness_col = 'Ladder score'
    gdp_col = 'GDP per capita'

    # 1. A histogram of all happiness scores across all years.
    logger.info("A histogram of all happiness scores across all years")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[happiness_col], bins=30, edgecolor='black',
            alpha=0.7, color='steelblue')
    ax.set_title('Happiness scores across all years')
    ax.set_xlabel('Happiness (ladder) Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.savefig(output_dir/"happiness_histogram.png")
    plt.close()
    logger.info('Saved: happiness_histogram.png')

    # 2. A boxplot comparing happiness score distributions across years (one box per year)
    logger.info("A boxplot comparing happiness score distributions across years")
    fig, ax = plt.subplots(figsize=(14, 6))
    df.boxplot(column=happiness_col, by='Year', ax=ax)
    ax.set_title('Happiness Score Dsictribution across years')
    ax.set_xlabel('Years')
    ax.set_ylabel('Happiness Score')
    ax.legend()
    plt.savefig(output_dir/"happiness_by_year.png")
    plt.close()
    logger.info("Saved: happiness_by_year.png")

    # 3. A scatter plot showing the relationship between GDP per capita and happiness score
    logger.info(
        'A scatter plot showing the relationship between GDP per capita and happiness score')
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df['GDP per capita'],
                         df[happiness_col], c=df['Year'], cmap='viridis', alpha=0.6, edgecolors='w', linewidths=0.5)
    ax.set_title('GDP per capita VS happiness score')
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Happiness Score')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Year')

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df[gdp_col], df[happiness_col])

    # Create trend line points
    trend_x = [df[gdp_col].min(), df[gdp_col].max()]
    trend_y = [slope * x + intercept for x in trend_x]

    # Plot trend line
    ax.plot(trend_x, trend_y, color='red', linewidth=2,
            label=f'Trend Line (R² = {r_value**2:.3f})')

    ax.legend()
    plt.savefig(output_dir/"gdp_vs_happiness.png")
    plt.close()
    logger.info('Saved: gdp_vs_happiness.png')

    # 4. Correlation heatmap showing the Pearson correlations between all numeric columns
    logger.info('Correlation heatmap')
    numeric_cols = ['Ranking',
                    'Ladder score',
                    'GDP per capita',
                    'Social support',
                    'Healthy life expectancy',
                    'Freedom to make life choices',
                    'Generosity',
                    'Perceptions of corruption',
                    'Year']
    # filter to only exixting numeric columns
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    corr_metrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_metrix, dtype=bool), k=1)
    sns.heatmap(corr_metrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, mask=mask, square=True, cbar_kws={'shrink': 0.8}, ax=ax)

    ax.set_title('Correlation heatmap of Happiness Indicator')
    plt.savefig(output_dir/"correlation_heatmap.png")
    plt.close()
    logger.info('Saved: correlation_heatmap.png')

    logger.info('All visualizations - completed and saved')


# --------------------------------------------------------------------------------
#                    Task 4: Hypothesis Testing
# --------------------------------------------------------------------------------

@task
def perform_hypothesis_tests(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    happiness_col = 'Ladder score'

    # Test 1: Pre-pandemic vs Pandemic Happiness Scores (2019 vs 2020)
    logger.info('-' * 60)
    logger.info(
        'TEST 1: Pre-pandemic vs Pandemic Happiness Scores (2019 vs 2020)')
    logger.info('-' * 60)

    scores_2019 = df[df['Year'] == 2019][happiness_col].dropna()
    scores_2020 = df[df['Year'] == 2020][happiness_col].dropna()

    t_stat_pandemic, p_stat_pandemic = stats.ttest_ind(
        scores_2019, scores_2020)

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()
    diff = mean_2020 - mean_2019

    logger.info(
        f"Mean Happiness Score 2019: {mean_2019:.4f} (n={len(scores_2019)})")
    logger.info(
        f"Mean Happiness Score 2020: {mean_2020:.4f} (n={len(scores_2020)})")
    logger.info(f"Difference (2020 - 2019): {diff:.4f}")
    logger.info(f"T-Statistic: {t_stat_pandemic:.4f}")
    logger.info(f"P-Value: {p_stat_pandemic:.6f}")

    alpha = 0.05
    if p_stat_pandemic < alpha:
        direction = "decreased" if mean_2020 < mean_2019 else "increased"
        logger.info(
            f"Interpretation: At alpha = {alpha}, we reject the null hypothesis.")
        logger.info(
            f"The pandemic had a statistically significant effect on global happiness.")
        logger.info(
            f"Average happiness {direction} from {mean_2019:.4f} in 2019 to {mean_2020:.4f} in 2020, a difference of {abs(diff):.4f} points.")
    else:
        logger.info(
            f"Interpretation: At alpha = {alpha}, we fail to reject the null hypothesis.")
        logger.info(
            f"Despite the pandemic beginning in 2020, we did not detect a statistically significant difference in global happiness scores.")
        logger.info(
            f"The difference of {abs(diff):.4f} points could be due to random variation.")

    # Test 2: Regional Comparison - Top vs Bottom Happiness Regions
    logger.info('-' * 60)
    logger.info(
        'TEST 2: Regional Comparison - Highest vs Lowest Average Happiness Regions')
    logger.info('-' * 60)

    # Compute mean happiness by region
    regional_means = df.groupby('Regional indicator')[
        happiness_col].mean().sort_values()

    top_region = regional_means.idxmax()
    bottom_region = regional_means.idxmin()

    top_region_scores = df[df['Regional indicator']
                           == top_region][happiness_col].dropna()
    bottom_region_scores = df[df['Regional indicator']
                              == bottom_region][happiness_col].dropna()

    t_stat_regions, p_stat_regions = stats.ttest_ind(
        top_region_scores, bottom_region_scores)

    mean_top = top_region_scores.mean()
    mean_bottom = bottom_region_scores.mean()
    region_diff = mean_top - mean_bottom

    logger.info(f"Top Region: {top_region}")
    logger.info(f"Mean Happiness: {mean_top:.4f} (n={len(top_region_scores)})")
    logger.info(f"Bottom Region: {bottom_region}")
    logger.info(
        f"Mean Happiness: {mean_bottom:.4f} (n={len(bottom_region_scores)})")
    logger.info(f"Difference: {region_diff:.4f}")
    logger.info(f"T-Statistic: {t_stat_regions:.4f}")
    logger.info(f"P-Value: {p_stat_regions:.6f}")

    if p_stat_regions < alpha:
        logger.info(
            f"Interpretation: At alpha = {alpha}, we reject the null hypothesis.")
        logger.info(
            f"There is a statistically significant difference in happiness between these regions.")
        logger.info(
            f"{top_region} has significantly higher happiness ({mean_top:.4f}) compared to {bottom_region} ({mean_bottom:.4f}).")
    else:
        logger.info(
            f"Interpretation: At alpha = {alpha}, we fail to reject the null hypothesis.")
        logger.info(
            f"Despite observing differences in sample means, the difference is not statistically significant.")
        logger.info(
            f"This could indicate region-level variation does not account for population differences.")

    return {
        'pandemic_test': {
            't_statistic': t_stat_pandemic,
            'p_value': p_stat_pandemic,
            'mean_2019': mean_2019,
            'mean_2020': mean_2020,
            'significant': p_stat_pandemic < alpha
        },
        'regional_test': {
            't_statistic': t_stat_regions,
            'p_value': p_stat_regions,
            'top_region': top_region,
            'top_mean': mean_top,
            'bottom_region': bottom_region,
            'bottom_mean': mean_bottom,
            'significant': p_stat_regions < alpha
        }
    }

# --------------------------------------------------------------------------------
#                    Task 5: Correlation and Multiple Comparisons
# --------------------------------------------------------------------------------


@task
def analyze_correlaltion(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    happiness_col = 'Ladder score'

    logger.info("Analyzing correlation with Happiness score")

    # explanatory variables to correlate with happiness
    explanatory_vars = [
        'GDP per capita',
        'Social support',
        'Healthy life expectancy',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ]

    # filter cols
    explanatory_vars = [var for var in explanatory_vars if var in df.columns]

    logger.info('-' * 60)
    logger.info("Pearson correlation with happiness score ")
    logger.info('-' * 60)

    results = []

    for var in explanatory_vars:
        clean_data = df[[happiness_col, var]].dropna()

        if len(clean_data) > 2:
            r, p_value = stats.pearsonr(
                clean_data[happiness_col], clean_data[var])
            results.append({
                'variable': var,
                'correlation': r,
                'p_value': p_value
            })

            logger.info(f"var:")
            logger.info(f" Pearson r = {r:.4f}, p-value = {p_value:.2e}")

    # apply Bonferroni correction
    num_tests = len(results)
    original_alpha = 0.05
    adjusted_alpha = original_alpha / num_tests

    logger.info("=" * 60)
    logger.info("MULTIPLE COMPARISONS ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Number of tests performed: {num_tests}")
    logger.info(f"Original alpha: {original_alpha}")
    logger.info(f"Bonferroni-corrected alpha: {adjusted_alpha:.6f}")

    logger.info("")
    logger.info("SIGNIFICANCE AT ORIGINAL ALPHA (0.05):")
    significant_original = []
    for res in results:
        if res['p_value'] < original_alpha:
            significant_original.append(res['variable'])
            logger.info(
                f" {res['variable']} (r={res['correlation']:.4f}, p={res['p_value']:.2e})")

    if not significant_original:
        logger.info(f"No significant correlation in alpha = 0.05")

    logger.info(
        f"SIGNIFICANCE AFTER BONFERRONI CORRECTION (alpha = {adjusted_alpha:.6f}):")
    significant_corrected = []
    for res in results:
        if res['p_value'] < adjusted_alpha:
            significant_corrected.append(res['variable'])
            logger.info(
                f" {res['variable']} (r={res['correlation']:.4f}, p={res['p_value']:.2e})")
    if not significant_corrected:
        logger.info("No significant correlations after Bonferroni correction")

    # Find strongest correlation that survives Bonferroni
    sorted_results = sorted(results, key=lambda x: abs(
        x['correlation']), reverse=True)
    strongest_significant = None

    for res in sorted_results:
        if res['p_value'] < adjusted_alpha:
            strongest_significant = res
            break

    if strongest_significant:
        logger.info("STRONGEST SIGNIFICANT CORRELATION (after Bonferroni):")
        logger.info(
            f"{strongest_significant['variable']} with r = {strongest_significant['correlation']:.4f}")
    else:
        for res in sorted_results:
            if res['p_value'] < original_alpha:
                strongest_significant = res
                break
        if strongest_significant:
            logger.info(
                "STRONGEST CORRELATION (significant at original alpha only):")
            logger.info(
                f"{strongest_significant['variable']} with r = {strongest_significant['correlation']:.4f}")

    return {
        'correlations': results,
        'num_tests': num_tests,
        'adjusted_alpha': adjusted_alpha,
        'significant_original': significant_original,
        'significant_corrected': significant_corrected,
        'strongest_significant': strongest_significant
    }

# --------------------------------------------------------------------------------
#                    Task 6: Summary Report
# --------------------------------------------------------------------------------


@task
def generate_summary_report(
    df: pd.DataFrame,
    descriptive_stats: dict,
    hypothesis_results: dict,
    correlation_results: dict
) -> None:

    logger = get_run_logger()
    logger.info(
        "----------------------WORLD HAPPINESS ANALYSIS SUMMARY----------------")

    # Total number of countries and years in the merged dataset.
    # 1. Dataset Overview

    num_countries = df['Country'].nunique()
    num_years = df['Year'].nunique()
    year_range = f"{df['Year'].min()} - {df['Year'].max()}"
    total_observations = len(df)

    logger.info("-----Dataset Overview-----")
    logger.info(f"Total Unique Countries: {num_countries}")
    logger.info(f"Years covered: {num_years} - ({year_range})")
    logger.info(f"Total Observations: {total_observations}")

    # The top 3 and bottom 3 regions by mean happiness score.
    # 2. Top and Bottom regions

    regions_sorted = sorted(
        descriptive_stats['by_region'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    logger.info("Top 3 Regions - Highest mean happiness")
    for i, (region, score) in enumerate(regions_sorted[:3], 1):
        logger.info(f"  {i}. {region}: {score:.4f}")

    logger.info("Bottom 3 Regions - lowest mean happiness")
    for i, (region, score) in enumerate(regions_sorted[-3:], 1):
        logger.info(f"  {i}. {region}: {score:.4f}")

    # The result of the pre/post-2020 t-test in plain language.
    # 3. Pandemic Impact:
    logger.info("-----Pandemic Impact-----")
    pandemic = hypothesis_results['pandemic_test']

    if pandemic['significant']:
        diff = pandemic['mean_2019'] - pandemic['mean_2020']
        direction = 'descreased' if diff > 0 else 'increased'

        logger.info(
            f"Golabl happiness {direction} significantly during the first year of covid")
        logger.info(
            f"Change: {pandemic['mean_2019']:.4f} → {pandemic['mean_2020']:.4f} (p={pandemic['p_value']:.4f})")
    else:
        logger.info(
            f" No statistically significant change in global happiness was detected")
        logger.info(
            f" between 2019 ({pandemic['mean_2019']:.4f}) and 2020 ({pandemic['mean_2020']:.4f}).")
        logger.info(f" P-value: {pandemic['p_value']:.4f} (threshold: 0.05)")

    # The variable most strongly correlated with happiness score (after Bonferroni correction)
    # 4. Strongest Happpiness Score
    logger.info("Strongest Happiness Score")

    strongest = correlation_results['strongest_significant']
    if strongest:
        logger.info("Strongest predictor (after Bonferroni correction):")
        logger.info(f"→ {strongest['variable']}")
        logger.info(f"Correlation: r = {strongest['correlation']:.4f}")
        logger.info(f"P-value: {strongest['p_value']:.2e}")

        # Interprtation
        if abs(strongest['correlation']) >= 0.7:
            strength = 'very strong'
        elif abs(strongest['correlation']) >= 0.5:
            strength = 'strong'
        elif abs(strongest['correlation']) >= 0.3:
            strength = 'moderate'
        else:
            strength = 'weak'

        direction = "positive" if strongest['correlation'] > 0 else "negative"
        logger.info(
            f" Interpretation: A {strength} {direction} relationship")
    else:
        logger.info(
            " No significant predictors identified after Bonferroni correction.")


# ---------------------------------------------------------------------------------
#                            Main Flow
# ---------------------------------------------------------------------------------


@flow
def happiness_pipeline():
    logger = get_run_logger()

    # Task 1: Load and merge the data
    df = load_and_merge_data()

    # Task 2: Compute Descriptive statistics
    descriptive_stats = compute_descriptive_statistics(df)

    # Task 3: Visual Exploration
    create_visualization(df)

    # Task 4: Hypothesis Testing
    hypothesis_results = perform_hypothesis_tests(df)

    # Task 5: Correlation and Multiple Comparisons
    correlation_results = analyze_correlaltion(df)

    # Task 6: Summary Report
    generate_summary_report(df, descriptive_stats,
                            hypothesis_results, correlation_results)

    logger.info("----------------Pipeline Completed------------")

    return {
        'dataframe': df,
        'descrptive_stats': descriptive_stats,
        'hypothesis_results': hypothesis_results,
        'correlation_results': correlation_results
    }


if __name__ == "__main__":
    happiness_pipeline()
