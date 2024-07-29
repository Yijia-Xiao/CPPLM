import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
data = {
    'Strategy': ['Vanilla', 'Removal', 'Substitution', 'Private Transformers', 'IT_PN', 'IT_NP'],
    'ROUGE-1': [0.3342, 0.2947, 0.2983, 0.3172, 0.3273, 0.3261],
    'ROUGE-2': [0.2174, 0.2071, 0.2073, 0.2096, 0.2112, 0.2162],
    'ROUGE-L': [0.3297, 0.3092, 0.3173, 0.3192, 0.3221, 0.3252],
    'BERTScore': [0.8162, 0.7983, 0.8012, 0.8119, 0.8101, 0.8132],
    'y': [100.00, 53.24, 55.14, 54.19, 53.62, 54.38]
    # 'y': [0.1050, 0.0559, 0.0579, 0.0568, 0.0563, 0.0571]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to determine Pareto frontier
def pareto_frontier(x, y):
    pareto_front = []
    for i in range(len(x)):
        dominated = False
        for j in range(len(x)):
            if (x[j] <= x[i] and y[j] >= y[i]) and (x[j] < x[i] or y[j] > y[i]):
                dominated = True
                break
        if not dominated:
            pareto_front.append([x[i], y[i]])
    pareto_front.sort()
    return pareto_front

# Plotting with seaborn
sns.set(style="whitegrid")

fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Plotting the data points for each case
metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']
titles = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']

for i, ax in enumerate(axs.flat):
    metric = metrics[i]
    sns.scatterplot(data=df, x='y', y=metric, hue='Strategy', style='Strategy', s=100, ax=ax)

    # Determine Pareto frontier
    pareto_points = pareto_frontier(df['y'], df[metric])
    pareto_x = [point[0] for point in pareto_points]
    pareto_y = [point[1] for point in pareto_points]

    # Plotting the Pareto frontier
    ax.plot(pareto_x, pareto_y, linestyle='--', color='blue', linewidth=2)

    # Invert x-axis
    ax.invert_xaxis()

    # Adding labels and title
    ax.set_xlabel('PII Leak Ratio', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f'PII-Utility Pareto Frontier: {titles[i]}', fontsize=16)

# Adjust legend
handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', title='Strategy', fontsize=12, ncol=3)

plt.tight_layout()
plt.show()