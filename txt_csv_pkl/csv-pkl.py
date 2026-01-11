import pandas as pd
import pickle


df = pd.read_csv(r'D:\openpose\data.csv')


score_dict = dict(zip(df['Video_Name'], df['Total_Score'].astype(float)))

with open(r'D:\openpose\data.pkl', 'wb') as f:
    pickle.dump(score_dict, f)

