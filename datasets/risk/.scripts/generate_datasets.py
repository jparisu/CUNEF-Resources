import pandas as pd
from enterprise_risk_generator import generate_dataset

###############
# Train dataset
train_df, _ = generate_dataset(
    N=5000,
    seed=0,
    outlier_prob=0.005
)

# store the dataset as a csv file
train_df.to_csv("../risk_train_dataset.csv", index=False)

###############
# Test dataset
for seed in [1, 24, 42, 20260326]:

    # generate a test dataset with a different seed and a lower outlier probability
    test_df, _ = generate_dataset(
        N=500,
        seed=seed,
        outlier_prob=0.002
    )

    # store the dataset as a csv file
    test_df.to_csv(f"../risk_test_dataset_{seed}.csv", index=False)
