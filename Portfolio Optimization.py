import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from deap import base, creator, tools, algorithms
from datetime import datetime, date
import yfinance as yf
import random
import time
import sys

np.random.seed(42)
NUM_SIMULATIONS = 1000
NUM_DAYS = 252
RISK_FREE_RATE = 0.03 
SIMULATION_DAYS = 252


def adjust_to_business_day(date_input):
    date = pd.to_datetime(date_input)
    return date if date.weekday() < 5 else date + BDay(1)

def download_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        time.sleep(1)  
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        df['Returns'] = df['Adj Close'].pct_change().dropna()
        data[ticker] = df
    return data

def get_mean_return_volatility(data_dict):
    mean_returns = []
    volatilities = []
    initial_prices = []
    for ticker, df in data_dict.items():
        mean_returns.append(df['Returns'].mean())
        volatilities.append(df['Returns'].std())
        initial_prices.append(df['Adj Close'].iloc[0])
    return np.array(mean_returns), np.array(volatilities), np.array(initial_prices)

def get_correlation_matrix(data_dict):
    returns_df = pd.DataFrame({k: v['Returns'] for k, v in data_dict.items()})
    return returns_df.corr().values

def build_cov_matrix(volatilities, correlation_matrix):
    return np.outer(volatilities, volatilities) * correlation_matrix



def simulate_correlated_paths(S0, mu, sigma, corr_matrix, num_days, num_simulations):
    dt = 1 / num_days
    L = np.linalg.cholesky(corr_matrix)
    num_assets = len(S0)
    paths = np.zeros((num_simulations, num_assets, num_days))
    paths[:, :, 0] = S0
    for t in range(1, num_days):
        Z = np.random.standard_normal((num_simulations, num_assets))
        correlated_randoms = np.dot(Z, L.T)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * correlated_randoms
        paths[:, :, t] = paths[:, :, t - 1] * np.exp(drift + diffusion)
    return paths



creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def fitness(individual, mean_returns, cov_matrix):
    weights = np.array(individual)
    port_return = np.dot(mean_returns, weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else -1
    return sharpe,

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: list(np.random.dirichlet(np.ones(num_assets))))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness, mean_returns=mean_returns, cov_matrix=cov_matrix)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=300)
    NGEN = 100
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    best = tools.selBest(population, 1)[0]
    return np.array(best)


def main(portfolio_dict, start_date_str, risk_free_rate=0.03):
    global RISK_FREE_RATE
    RISK_FREE_RATE = risk_free_rate

    tickers = list(portfolio_dict.keys())
    shares = np.array(list(portfolio_dict.values()))
    start = adjust_to_business_day(start_date_str)
    end = adjust_to_business_day(date.today())

    data = download_stock_data(tickers, start, end)
    mean_returns, volatilities, initial_prices = get_mean_return_volatility(data)
    correlation_matrix = get_correlation_matrix(data)
    cov_matrix = build_cov_matrix(volatilities, correlation_matrix)

    weights = optimize_portfolio(mean_returns, cov_matrix)

    simulated = simulate_correlated_paths(initial_prices, mean_returns, volatilities,
                                          correlation_matrix, SIMULATION_DAYS, NUM_SIMULATIONS)
    final_values = np.sum(simulated[:, :, -1] * weights, axis=1)

    initial_value = np.sum(initial_prices * shares)
    VaR = np.percentile(final_values, 5)
    ES = np.mean(final_values[final_values <= VaR])

    returns = final_values / initial_value - 1
    annualized_return = np.mean(returns) * SIMULATION_DAYS
    volatility = np.std(returns) * np.sqrt(SIMULATION_DAYS)
    sharpe = (annualized_return - RISK_FREE_RATE) / volatility if volatility > 0 else -1

    plt.figure(figsize=(10, 6))
    plt.hist(final_values, bins=50, alpha=0.6)
    plt.axvline(VaR, color='r', linestyle='--', label=f'VaR (5%): ${VaR:,.2f}')
    plt.axvline(ES, color='g', linestyle='--', label=f'Expected Shortfall: ${ES:,.2f}')
    plt.title("Simulated Portfolio Value Distribution")
    plt.xlabel("Portfolio Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"VaR (95% confidence): ${VaR:,.2f}")
    print(f"Expected Shortfall: ${ES:,.2f}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")


