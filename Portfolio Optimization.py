import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import random
import time
from datetime import date
from pandas.tseries.offsets import BDay
from deap import base, creator, tools, algorithms

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

FMP_API_KEY = "omapP0bkAITjk969vNIRefx2fvRqkVc5"
FMP_API_URL = "https://financialmodelingprep.com/api/v3/profile"

def adjust_to_business_day(date_input, forward=True):
    date = pd.to_datetime(date_input)
    if date.weekday() >= 5:
        return (date + BDay(1)).date() if forward else (date - BDay(1)).date()
    return date.date()

def correct_ticker(ticker):
    return "AAPL" if ticker.upper() == "APPL" else ticker.upper()

def fetch_with_backoff(func, *args, max_retries=5):
    delays = [5, 10, 15, 30, 60]
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(delays[min(attempt, len(delays)-1)])

def cache_filename(ticker):
    return os.path.join(CACHE_DIR, f"{ticker.upper()}_cached_fmp.csv")

def fetch_stock_data_fmp(ticker):
    ticker = ticker.upper()
    filename = cache_filename(ticker)

    if os.path.exists(filename):
        try:
            print(f"Loading cached FMP data for {ticker}")
            df = pd.read_csv(filename)
            return float(df['mean_return'][0]), float(df['volatility'][0]), float(df['price'][0])
        except Exception as e:
            print(f"Cache for {ticker} is corrupted. Refetching... ({e})")
            os.remove(filename)

    def fetch():
        url = f"{FMP_API_URL}/{ticker}?apikey={FMP_API_KEY}"
        print(f"Fetching {ticker} data from FMP")
        response = requests.get(url)
        data = response.json()
        if not data or not isinstance(data, list):
            raise ValueError(f"No data returned for {ticker}.")
        item = data[0]
        price = item.get('price', 100)
        beta = item.get('beta', 1)
        mean_return = 0.0005 * beta
        volatility = 0.01 * beta

        df = pd.DataFrame([{'price': price, 'mean_return': mean_return, 'volatility': volatility}])
        df.to_csv(filename, index=False)
        return mean_return, volatility, price

    return fetch_with_backoff(fetch)

def simulate_price_paths(S0, mu, sigma, corr, days, sims, dt):
    L = np.linalg.cholesky(corr)
    assets = len(S0)
    paths = np.zeros((sims, assets, days))
    paths[:, :, 0] = S0
    for t in range(1, days):
        Z = np.random.standard_t(df=5, size=(sims, assets))
        corr_rand = np.dot(L, Z.T).T
        for i in range(assets):
            drift = (mu[i] - 0.5 * sigma[i]**2) * dt
            diffusion = sigma[i] * np.sqrt(dt) * corr_rand[:, i]
            paths[:, i, t] = paths[:, i, t-1] * np.exp(drift + diffusion)
    return paths

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def normalize(ind):
    total = sum(ind)
    if total == 0:
        return [1.0 / len(ind)] * len(ind)
    return [max(0, x / total) for x in ind]

def fitness(ind):
    weights = normalize(ind)
    port_return = np.dot(mean_returns, weights)
    port_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_risk
    return (port_return * sharpe,) if sharpe >= 1 else (port_return * sharpe**2,)

def check_sum(ind):
    return abs(sum(normalize(ind)) - 1.0) < 1e-4

def optimize_portfolio():
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(mean_returns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", tools.DeltaPenalty(check_sum, 1.0))
    toolbox.decorate("mutate", tools.DeltaPenalty(check_sum, 1.0))

    population = toolbox.population(n=100)
    for gen in range(20):
        print(f"Generation {gen + 1}")
        offspring = list(map(toolbox.clone, population))

        for i in range(1, len(offspring), 2):
            if random.random() < 0.7:
                mated = toolbox.mate(offspring[i - 1], offspring[i])
                if len(mated) == 2:
                    offspring[i - 1], offspring[i] = mated
            if random.random() < 0.2:
                toolbox.mutate(offspring[i - 1])
                toolbox.mutate(offspring[i])
            del offspring[i - 1].fitness.values
            del offspring[i].fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(offspring, len(population))

        best = tools.selBest(population, 1)[0]
        norm_best = normalize(best)
        print(f"   Best so far (top 3 weights): {np.round(norm_best[:3], 4)}... Total: {sum(norm_best):.4f}")

    return normalize(tools.selBest(population, 1)[0])

def simulate_portfolio_dual(portfolio_dict, start_input):
    np.random.seed(42)
    num_simulations = 1000
    num_days = 252
    time_horizon = 0.00273972602 * num_days

    start_date = adjust_to_business_day(start_input, forward=True)
    end_date = adjust_to_business_day(date.today(), forward=False)
    print(f"Fetching data from {start_date} to {end_date}")

    global risk_free_rate, mean_returns, cov_matrix
    risk_free_rate = 0.0425

    mean_returns = []
    volatilities = []
    initial_prices = []

    tickers = list(portfolio_dict.keys())

    for ticker in tickers:
        try:
            ret, vol, price = fetch_stock_data_fmp(ticker)
            mean_returns.append(ret)
            volatilities.append(vol)
            initial_prices.append(price)
        except Exception as e:
            print(f"{ticker} failed: {e}")
            initial_prices.append(0)

    mean_returns = np.array(mean_returns)
    volatilities = np.array(volatilities)
    initial_prices = np.array(initial_prices)
    correlation_matrix = np.identity(len(tickers))
    cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

    if np.any(np.isnan(mean_returns)) or np.any(np.isnan(volatilities)) or np.any(np.isnan(initial_prices)):
        print("Invalid data detected (NaN). Check cache.")
        return
    if np.any(volatilities == 0):
        print("One or more assets have zero volatility. Check cached values.")
        return

    sim_paths = simulate_price_paths(initial_prices, mean_returns, volatilities,
                                     correlation_matrix, num_days, num_simulations,
                                     time_horizon / num_days)

    actual_values = np.array([portfolio_dict[t] * p for t, p in zip(tickers, initial_prices)])
    actual_weights = actual_values / actual_values.sum()

    optimal_weights = optimize_portfolio()

    def analyze_portfolio(weights, label):
        final_vals = np.sum(sim_paths[:, :, -1] * weights, axis=1)
        VaR = np.percentile(final_vals, 5)
        ES = np.mean(final_vals[final_vals <= VaR])
        init_val = np.sum(initial_prices * weights)
        returns = final_vals / init_val - 1
        ann_return = np.mean(returns) * (252 / time_horizon)
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / volatility
        return final_vals, (VaR, ES, ann_return, volatility, sharpe, init_val)

    final_actual, stats_actual = analyze_portfolio(actual_weights, "Actual")
    final_opt, stats_opt = analyze_portfolio(optimal_weights, "Optimized")

    plt.figure(figsize=(12, 6))
    plt.hist(final_actual, bins=50, alpha=0.6, label=f"Actual (VaR: ${stats_actual[0]:.2f}, ES: ${stats_actual[1]:.2f})")
    plt.hist(final_opt, bins=50, alpha=0.6, label=f"Optimized (VaR: ${stats_opt[0]:.2f}, ES: ${stats_opt[1]:.2f})")
    plt.axvline(stats_actual[0], color='r', linestyle='--')
    plt.axvline(stats_actual[1], color='r', linestyle=':')
    plt.axvline(stats_opt[0], color='g', linestyle='--')
    plt.axvline(stats_opt[1], color='g', linestyle=':')
    plt.title("Monte Carlo Simulation: Actual vs Optimized Portfolio")
    plt.xlabel("Final Portfolio Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nMetrics Summary:")
    for label, stats in zip(["Actual", "Optimized"], [stats_actual, stats_opt]):
        print(f"\n--- {label} Portfolio ---")
        print(f"Initial Value: ${stats[5]:,.2f}")
        print(f"VaR (95%): ${stats[0]:,.2f}")
        print(f"Expected Shortfall: ${stats[1]:,.2f}")
        print(f"Annualized Return: {stats[2]:.2%}")
        print(f"Volatility: {stats[3]:.2%}")
        print(f"Sharpe Ratio: {stats[4]:.2f}")

    print("\nOptimized Share Allocation Plan:")
    portfolio_value = actual_values.sum()
    opt_dollar_alloc = portfolio_value * np.array(optimal_weights)
    opt_share_alloc = np.floor(opt_dollar_alloc / initial_prices).astype(int)
    actual_shares = np.array([portfolio_dict[t] for t in tickers])

    for i, ticker in enumerate(tickers):
        delta = opt_share_alloc[i] - actual_shares[i]
        if delta > 0:
            action = f"Buy {delta} more"
        elif delta < 0:
            action = f"Sell {-delta}"
        else:
            action = "Keep as is"
        print(f"{ticker:>5}: Own {actual_shares[i]}, New Target: {opt_share_alloc[i]} → {action}")

    for i, w in enumerate(optimal_weights):
        if w < 0.01:
            print(f"Consider removing {tickers[i]} (only {w:.2%} allocation)")

simulate_portfolio_dual({"TSLA": 45, "NVDA": 43}, "2019-09-03")
