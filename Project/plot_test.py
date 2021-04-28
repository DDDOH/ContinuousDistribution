import seaborn as sns
geyser = sns.load_dataset("geyser")
sns.kdeplot(data=geyser, x="waiting", y="duration", hue="kind")
