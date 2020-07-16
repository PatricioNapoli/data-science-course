import seaborn
import matplotlib.pyplot as plt


def explore(df):
    print()
    print()
    print("========================================")
    print("================EXPLORE=================")
    print("========================================")

    df.info()
    print(df.describe(include="all").transpose())

    seaborn.boxplot(df[df["lastLevel"] <= 200]["lastLevel"][:5000])
    plt.savefig("plots/lastLevel.png")
    plt.clf()

    seaborn.boxplot(df[df["totalSessions"] <= 200]["totalSessions"][:5000])
    plt.savefig("plots/totalSessions.png")
    plt.clf()

    seaborn.boxplot(df[df["totalAge"] <= 250]["totalAge"][:5000])
    plt.savefig("plots/totalAge.png")
    plt.clf()

    seaborn.boxplot(df["winRate"][:5000])
    plt.savefig("plots/winRate.png")
    plt.clf()

    seaborn.boxplot(df[df["dailySessions"] <= 100]["dailySessions"][:5000])
    plt.savefig("plots/dailySessions.png")
    plt.clf()