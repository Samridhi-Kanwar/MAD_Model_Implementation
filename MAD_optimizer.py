from pulp import *
import pandas as pd

def clean_returns_data(returns_df):

    cleaned_df = returns_df.dropna()
    asset_returns = cleaned_df.drop(['Date', 'Market Portfolio.1'], axis=1)

    return asset_returns


def mad_portfolio_optimization(returns_df, target_return):

    asset_returns = clean_returns_data(returns_df)
    n_assets = len(asset_returns.columns)
    n_periods = len(asset_returns)
    mean_returns = asset_returns.mean()

    print("Number of assets:", n_assets)
    print("Number of periods:", n_periods)
    print("Mean returns range:", mean_returns.min(), "to", mean_returns.max())

    prob = LpProblem("MAD_Portfolio_Optimization", LpMinimize)

    asset_weights = LpVariable.dicts("weights",
                                     asset_returns.columns,
                                     lowBound=0,
                                     upBound=1)

    y = LpVariable.dicts("y",
                         range(n_periods),
                         lowBound=0)

    prob += lpSum(y[i] for i in range(n_periods)) / n_periods

    # Constraints

    # 1. Sum of weights = 1
    prob += lpSum(asset_weights[asset] for asset in asset_returns.columns) == 1

    # 2. Return target constraint
    prob += lpSum(asset_weights[asset] * mean_returns[asset]
                  for asset in asset_returns.columns) >= target_return

    # 3. Absolute deviation constraints
    for period in range(n_periods):
        period_returns = asset_returns.iloc[period]

        prob += lpSum(asset_weights[asset] * period_returns[asset]
                      for asset in asset_returns.columns) - \
                lpSum(asset_weights[asset] * mean_returns[asset]
                      for asset in asset_returns.columns) <= y[period]

        prob += -lpSum(asset_weights[asset] * period_returns[asset]
                       for asset in asset_returns.columns) + \
                lpSum(asset_weights[asset] * mean_returns[asset]
                      for asset in asset_returns.columns) <= y[period]

    # Solve the optimization problem
    prob.solve()

    # Extract results
    if LpStatus[prob.status] == 'Optimal':
        optimal_weights = {asset: value(asset_weights[asset])
                           for asset in asset_returns.columns}

        # Calculate portfolio metrics
        portfolio_return = sum(optimal_weights[asset] * mean_returns[asset]
                               for asset in asset_returns.columns)

        portfolio_mad = value(prob.objective)

        return {
            'weights': optimal_weights,
            'portfolio_return': portfolio_return,
            'portfolio_mad': portfolio_mad,
            'status': LpStatus[prob.status]
        }
    else:
        return {
            'status': LpStatus[prob.status],
            'message': 'Optimization failed to find a solution'
        }


def display_portfolio_results(results):

    print("\nPortfolio Optimization Results:")
    print("-" * 50)
    print(f"Optimization Status: {results['status']}")

    if results['status'] == 'Optimal':
        print(f"\nPortfolio Metrics:")
        print(f"Expected Return: {results['portfolio_return']:.4f}")
        print(f"MAD Risk Measure: {results['portfolio_mad']:.4f}")

        print("\nOptimal Portfolio Weights:")
        weights_df = pd.DataFrame.from_dict(results['weights'],
                                            orient='index',
                                            columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.0001]  # Filter out tiny weights
        weights_df = weights_df.sort_values('Weight', ascending=False)
        print(weights_df)
    else:
        print("\nOptimization failed to find a solution")
