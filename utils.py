import matplotlib.pyplot as plt
import seaborn as sns

def plot_data(df):
    numeric_cols = df.select_dtypes(include='number').columns

    plots = []
    for col in numeric_cols[:3]:
        fig, ax = plt.subplots()
        sns.histplot(df[col], ax=ax)
        plots.append(fig)

    return plots