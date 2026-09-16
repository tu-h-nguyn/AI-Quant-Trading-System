"""Prediction-market research and trading layer for Polymarket.

The module keeps the same research contract as the equities stack: every
economic quantity is computed from information available at decision time,
frictions are charged explicitly, and no component assumes an edge exists.

Sub-modules
-----------
``pricing``     binary-contract arithmetic, fees, break-even probabilities
``orderbook``   depth-aware execution simulation against a CLOB book
``markets``     normalized market/outcome records from the public APIs
``client``      read-only Gamma + CLOB HTTP client with on-disk snapshots
``arbitrage``   structural (model-free) mispricing scanners
``kelly``       bankroll sizing under edge uncertainty
``features``    snapshot feature engineering for probability forecasting
``model``       calibrated probability estimator benchmarked against price
``backtest``    event-driven simulation with settlement at 0/1
``metrics``     forecast-quality and bankroll diagnostics
``execution``   risk-gated order planning and a paper broker
``simulation``  deterministic synthetic markets for hermetic testing
"""
