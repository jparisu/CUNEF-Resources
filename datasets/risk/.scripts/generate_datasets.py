import pandas as pd
from risk_dataset_generator import generate_company_risk_dataset

###############
# Train dataset
train_df = generate_company_risk_dataset(
    n=5000,
    seed=0,
    outlier_prob=0.002,
    return_all_columns=False,
)

# store the dataset as a csv file
train_df.to_csv("../risk_train_dataset.csv", index=False)



###############
# Test dataset
test_df = generate_company_risk_dataset(
    n=500,
    seed=1,
    outlier_prob=0.001,
    return_all_columns=False,
)

# store the dataset as a csv file
test_df.to_csv("../risk_test_dataset_1.csv", index=False)
