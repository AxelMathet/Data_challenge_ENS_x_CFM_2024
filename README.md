# Data_challenge_ENS_x_CFM_2024

# High-Frequency Market Data: Could You Guess the Stock?

## Overview

This repository presents my implementation of a **Temporal Fusion Transformer (TFT)** for the CFM 2024 challenge *High-Frequency Market Data: Could You Guess the Stock?* on ChallengeData ENS.

This model achieved **5th place out of 170 teams in the final competition ranking in December 2024**.

Please note that the preprocessing section has been intentionally omitted from this repository.

## Challenge Description

**Organizer:** Capital Fund Management (CFM)  
**Platform:** ChallengeData ENS  
**Start date:** January 10, 2024

### Objective

The objective of this challenge is to identify which of 24 stocks generated a given sequence of high-frequency market data.

Each sample consists of 100 consecutive order-book events describing changes in an aggregated order book across multiple trading venues. These events include new orders, order deletions, and order updates, which may result from trades or cancellations.

Participants must develop a classification model capable of identifying the stock associated with each sequence based on the observed market microstructure data.

### Dataset

The dataset contains approximately 24.24 million event rows, covering 24 stocks over approximately 504 trading days. For each stock and each day, 20 sequences of 100 consecutive order-book events are sampled.

The dataset includes the following features:

| Feature | Description |
|---|---|
| `obs_id` | Identifier of a sequence of 100 events. |
| `venue` | Identifier of the exchange or trading venue where the event occurred. |
| `action` | Type of event: `A` (addition), `D` (deletion), or `U` (update). |
| `order_id` | Obfuscated identifier used to track an order within a sequence. |
| `side` | Side of the order book affected by the event: `A` or `B`. |
| `price` | Price of the order affected by the event. |
| `bid` | Best bid price in the aggregated order book. |
| `ask` | Best ask price in the aggregated order book. |
| `bid_size` | Volume displayed at the best bid. |
| `ask_size` | Volume displayed at the best ask. |
| `flux` | Change in order-book volume caused by the event. |
| `trade` | Boolean indicator distinguishing trade-related events from cancellations. |

To reduce the information conveyed by absolute price levels, the best bid price of the first event in each sequence is subtracted from the `price`, `bid`, and `ask` values.

### Target Variable

The target variable, `eqt_code_cat`, represents the identity of the stock. It is encoded as an integer from `0` to `23`, corresponding to one of the 24 stock classes.

The training data and test data cover different time periods, with the test period occurring later than the training period.

### Official Baseline Model

The challenge provides a baseline based on a bidirectional Gated Recurrent Unit (GRU) neural network.

Each sequence is represented as a tensor of shape `(100, 30)`, corresponding to 100 events with 30 features per event.

The feature representation includes:

- An 8-dimensional embedding for `venue`.
- An 8-dimensional embedding for `action`.
- An 8-dimensional embedding for `trade`.
- The numerical features `bid`, `ask`, and `price`.
- Log-transformed `bid_size`, `ask_size`, and `flux` features.

The resulting 30-dimensional event representations are processed by two GRUs, each with 64 hidden units. One GRU processes the sequence in chronological order, while the other processes it in reverse order.

The two final hidden representations are concatenated into a 128-dimensional vector. This vector is passed through a fully connected layer with 64 units and a SeLU activation, followed by a final linear layer producing 24 class logits.

### Training Configuration

The official baseline uses the following training setup:

- **Loss function:** Cross-entropy.
- **Batch size:** 1,000 sequences.
- **Optimizer:** Adam.
- **Learning rate:** `3e-3`.
- **Training duration:** 10,000 batches.

### Challenge Link

[CFM 2024 — High frequency market data: could you guess the stock?](https://challengedata.ens.fr/challenges/146)
