import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import operator

from collections import defaultdict

DATA_SET_PATH = 'data/'

df_price = pd.read_csv(DATA_SET_PATH + 'EoS_price_en.csv', sep=',')
df_fb = pd.read_csv(DATA_SET_PATH + 'EoS_feedback_en.csv', sep=',')
df_item = pd.read_csv(DATA_SET_PATH + 'EoS_item_en.csv', sep=',')
df_categories = pd.read_csv(DATA_SET_PATH + 'categories_new.csv', sep=',', header=None)


def format_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")


def normalize_countries(df):
    mapping = {
        'US': 'United States of America',
        'USA': 'United States of America',
        'USA ': 'United States of America',
        'Ships from: United States of America<br />': 'United States of America',
        'US East': 'United States of America',
        'us': 'United States of America',
        'Left Coast': 'United States of America',
        'Ships from: United Kingdom<br />': 'United Kingdom',
        'UK': 'United Kingdom',
        'Ships from: Germany<br />': 'Germany',
        'germany': 'Germany',
        'Netherlands': 'The Netherlands',
        'Ships from: Belgium<br />': 'Belgium',
        'EU': 'European Union',
        'Australia and New Zeland': 'Australia and New Zealand',
        'worldwide': 'Worldwide',
        'world': 'Worldwide',
        'World Wide': 'Worldwide',
        'World wide': 'Worldwide',
        'Worldl wide': 'Worldwide',
        'Anywhere': 'Worldwide',
        'anywhere': 'Worldwide',
        'undeclared': 'Unknown',
    }
    df.ships_from = df.ships_from.apply(lambda c: mapping[c] if c in mapping else c)
    df.ships_to = df.ships_to.apply(lambda c: mapping[c] if c in mapping else c)

    return df


def plot_top(d, n=10):
    counts = defaultdict(int)
    for v in d:
        counts[v] += 1

    top_elements = sorted(counts.items(), key=operator.itemgetter(1))
    top_elements.reverse()

    top_names = [name for name, count in top_elements]
    top_counts = [count for name, count in top_elements]

    y_pos = np.arange(n)
    fig, ax = plt.subplots()
    ax.barh(y_pos, top_counts[:n], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[:n])
    ax.set_xlabel('Listings')
    ax.invert_yaxis()
    plt.tight_layout()

    plt.show(block=True)


def load_categories(df):
    category_mapping = {}
    for _, row in df.iterrows():
        category_mapping[int(row[0])] = row[1]
    # add mapping
    for i in range(0x100):
        if i not in category_mapping:
            category_mapping[i] = 'Unknown'
    category_names = list(set(category_mapping.values()))
    return category_mapping, category_names


def plot_item_development(df_i, df_c):
    category_mapping, category_names = load_categories(df_c)

    min_time = int(df_i.first_seen.min())
    max_time = int(df_i.last_seen.max())

    steps = 50
    step_time = int((max_time - min_time) / float(steps))

    time_range = range(min_time, max_time - step_time, step_time)

    bins = []
    for begin_time in time_range:
        bins.append(df_i[(df_i.first_seen < begin_time + step_time) & (df_i.last_seen > begin_time)])

    category_count = {c: 0 for c in category_names}
    total_count = 0

    categories_per_bin = []
    row_indices = []

    for df_bin, begin_time in zip(bins, time_range):
        categories = {c: 0 for c in category_names}
        for _, row in df_bin.iterrows():
            c = category_mapping[row.category]
            categories[c] += 1
            category_count[c] += 1
            total_count += 1
        row_indices.append(format_timestamp(begin_time + step_time / 2))
        categories_per_bin.append(categories)

    top_categories = sorted(category_count.items(), key=operator.itemgetter(1))
    top_categories.reverse()

    top_category_names = [name for name, count in top_categories]
    top_category_counts = [count for name, count in top_categories]

    # plot most popular categories
    n_categories = 20
    y_pos = np.arange(n_categories)
    fig, ax = plt.subplots()
    ax.barh(y_pos, top_category_counts[:n_categories], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_category_names[:n_categories])
    ax.set_xlabel('Listings')
    ax.invert_yaxis()
    plt.tight_layout()

    plt.show(block=True)

    # plot progression of most popular categories
    df_categories_per_bin = pd.DataFrame(categories_per_bin, index=row_indices)

    df_categories_per_bin[top_category_names[:10]].plot(kind='line')
    plt.tight_layout()

    plt.show(block=True)

    df_categories_per_bin[top_category_names[10:20]].plot(kind='line')
    plt.tight_layout()

    plt.show(block=True)


def plot_price_development(df_p, df_i, df_c):
    category_mapping, category_names = load_categories(df_c)

    item_category_mapping = {}
    for _, item in df_i.iterrows():
        if item.item_id not in item_category_mapping:
            item_category_mapping[item.item_id] = category_mapping[item.category]

    min_time = int(df_p.time.min())
    max_time = int(df_p.time.max())

    steps = 50
    step_time = int((max_time - min_time) / float(steps))

    time_range = range(min_time, max_time - step_time, step_time)

    bins = []
    for begin_time in time_range:
        bins.append(df_p[(df_p.time <= begin_time + step_time) & (df_p.time >= begin_time)])

    category_price = {c: 0 for c in category_names}
    category_count = {c: 0 for c in category_names}

    total_price = 0
    total_count = 0

    categories_per_bin = []
    row_indices = []

    for df_bin, begin_time in zip(bins, time_range):
        categories = {c: 0 for c in category_names}
        counts = {c: 0 for c in category_names}
        for _, row in df_bin.iterrows():
            if row.item_id not in item_category_mapping:
                continue
            c = item_category_mapping[row.item_id]
            categories[c] += row.price
            counts[c] += 1
            category_price[c] += row.price
            category_count[c] += 1
            total_price += row.price
            total_count += 1

        row_indices.append(format_timestamp(begin_time + step_time / 2))
        avg_category_price = {c: categories[c] / counts[c] if counts[c] > 0 else 0 for c in category_names}
        categories_per_bin.append(categories)

    avg_category_price = {c: category_price[c] / category_count[c] if category_count[c] > 0 else 0 for c in
                          category_names}

    top_categories = sorted(avg_category_price.items(), key=operator.itemgetter(1))
    top_categories.reverse()

    top_category_names = [name for name, count in top_categories]
    top_category_counts = [count for name, count in top_categories]

    # plot most popular categories
    n_categories = 20
    y_pos = np.arange(n_categories)
    fig, ax = plt.subplots()
    ax.barh(y_pos, top_category_counts[:n_categories], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_category_names[:n_categories])
    ax.set_xlabel('Price')
    ax.invert_yaxis()
    plt.tight_layout()

    plt.show(block=True)

    # plot progression of most popular categories
    df_categories_per_bin = pd.DataFrame(categories_per_bin, index=row_indices)

    df_categories_per_bin[top_category_names[:10]].plot(kind='line')
    plt.tight_layout()

    plt.show(block=True)

    df_categories_per_bin[top_category_names[10:20]].plot(kind='line')
    plt.tight_layout()

    plt.show(block=True)


def plot_rating_development(df_f, df_i, df_c):
    category_mapping, category_names = load_categories(df_c)

    item_category_mapping = {}
    for _, item in df_i.iterrows():
        if item.item_id not in item_category_mapping:
            item_category_mapping[item.item_id] = category_mapping[item.category]

    # don't know why this workaround is required
    min_time = int(df_f[df_f.feedback_time > 0].feedback_time.min())
    max_time = int(df_f.feedback_time.max())

    steps = 20
    step_time = int((max_time - min_time) / float(steps))

    time_range = range(min_time, max_time - step_time, step_time)

    bins = []
    for begin_time in time_range:
        bins.append(df_f[(df_f.feedback_time <= begin_time + step_time) & (df_f.feedback_time >= begin_time)])

    category_rating = {c: 0 for c in category_names}
    category_count = {c: 0 for c in category_names}

    total_rating = 0
    total_count = 0

    categories_per_bin = []
    row_indices = []

    for df_bin, begin_time in zip(bins, time_range):
        categories = {c: 0 for c in category_names}
        counts = {c: 0 for c in category_names}
        for _, row in df_bin.iterrows():
            if row.item_id not in item_category_mapping:
                continue
            c = item_category_mapping[row.item_id]
            categories[c] += row.feedback_rating
            counts[c] += 1
            category_rating[c] += row.feedback_rating
            category_count[c] += 1
            total_rating += row.feedback_rating
            total_count += 1

        row_indices.append(format_timestamp(begin_time + step_time / 2))
        avg_category_rating = {c: categories[c] / counts[c] if counts[c] > 0 else 0 for c in category_names}
        categories_per_bin.append(avg_category_rating)

    category_count = {c: category_count[c] if category_count[c] > 100 else 0 for c in category_names}

    avg_category_rating = {c: category_rating[c] / category_count[c] if category_count[c] > 0 else 0 for c in
                           category_names}

    top_categories = sorted(avg_category_rating.items(), key=operator.itemgetter(1))
    top_categories.reverse()

    top_category_names = [name for name, count in top_categories]
    top_category_counts = [count for name, count in top_categories]

    # plot most popular categories
    n_categories = 20
    y_pos = np.arange(n_categories)
    fig, ax = plt.subplots()
    ax.barh(y_pos, top_category_counts[:n_categories], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_category_names[:n_categories])
    ax.set_xlabel('Rating')
    ax.invert_yaxis()
    plt.tight_layout()

    plt.show(block=True)

    # plot progression of most popular categories
    df_categories_per_bin = pd.DataFrame(categories_per_bin, index=row_indices)

    df_categories_per_bin[top_category_names[:10]].plot(kind='line')
    plt.tight_layout()

    plt.show(block=True)

    df_categories_per_bin[top_category_names[10:20]].plot(kind='line')
    plt.tight_layout()

    plt.show(block=True)


df_item = normalize_countries(df_item)

plot_top(df_item.ships_from)
plot_top(df_item.ships_to)

plot_item_development(df_item, df_categories)
plot_item_development(df_item[df_item['ships_from'] == 'The Netherlands'], df_categories)

# plot_price_development(df_price, df_item, df_categories)
# plot_price_development(df_price, df_item[df_item['ships_from'] == 'The Netherlands'], df_categories)

# plot_rating_development(df_fb, df_item, df_categories)
# plot_rating_development(df_fb, df_item[df_item['ships_from'] == 'The Netherlands'], df_categories)
