# Deadlock Build Recommender

A GRU-based sequence model that recommends optimal item build orders for heroes in Deadlock, trained on live match data from the [Deadlock API](https://deadlock-api.com).

## How It Works

1. Fetches item win rates and ordered item pair stats from the Deadlock API per hero and rank tier
2. Constructs build sequences by chaining item pairs into longer buy-order paths via graph traversal
3. Trains a GRU model to predict the next best item given a hero, rank, and items bought so far
4. Uses beam search at inference time to generate complete recommended builds

## Project Structure
deadlock-build-recommender/
    --Notebooks/
        --data_cache/        # cached API responses (gitignored)
        best_model.pt        # trained model checkpoint (gitignored)
        deadlock_build_recommender.ipynb
README.md

## Setup

```bash
conda create -n deadlock-build-recommender python=3.11
conda activate deadlock-build-recommender
pip install torch requests pandas numpy scikit-learn tqdm matplotlib seaborn scipy
```

For GPU support, install PyTorch with the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

Open `Notebooks/deadlock_build_recommender.ipynb` and run all cells. Key config options at the top of the notebook:

- `RANK_TIER` — `"low"`, `"mid"`, or `"high"`
- `TARGET_HERO_IDS` — list of hero IDs to train on
- `N_EPOCHS` — number of training epochs

## Model

- **Architecture**: 2-layer GRU with item, hero, and rank embeddings
- **Input**: hero ID + rank tier + sequence of items bought so far
- **Output**: probability distribution over all shop items (next item prediction)
- **Loss**: weighted cross-entropy, upweighting high win-rate build sequences
- **Inference**: beam search with configurable width and build length

## Roadmap

### Current State
- GRU sequence model trained on item pair win rates from the Deadlock API
- Beam search inference for full build recommendations
- Conditioned on hero and rank tier

### Near Term
- [ ] Expand training to all heroes and all rank tiers
- [ ] Incorporate ability upgrade order into recommendations
- [ ] Add opponent hero context for counter-build suggestions
- [ ] Patch-aware training to handle meta shifts over time
- [ ] Improve early-game item ordering using buy-time data

### Full App Vision
- [ ] REST API wrapping the model (FastAPI)
- [ ] Web frontend where you pick your hero, rank, and optionally enemy heroes
- [ ] Real-time recommendations that update as you add items mid-game
- [ ] Build sharing and community voting
- [ ] Win rate tracking for recommended builds vs. actual match outcomes
- [ ] Mobile app for quick in-game reference