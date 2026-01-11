import pandas as pd

file_path = r"F:\datasets\ball_datasets\test.txt"

df = pd.read_csv(file_path, sep=r'\s+', engine='python')

expected_columns = ["Video_Name", "Difficulty_Score", "Execution_Score", "Total_Score", "Penalty", "Video_length"]
df.columns = expected_columns

output_path = r"F:\datasets\ball_datasets\test.csv"
df.to_csv(output_path, index=False)

