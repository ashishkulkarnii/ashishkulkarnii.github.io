---
title: "Graph Based Recommender"
excerpt: "Semester 6: NAM Project. A recommender system using link prediction algorithms.<br><img src='https://ashishkulkarnii.github.io/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/31699b46879260ba17e80fff9f144439d0deba91.png'>"
collection: portfolio
permalink: "projects/graph-based-recommender"
---

#### Course Project: Recommender System by Link Prediction using Algorithms

#### Importing necessary libraries

``` python
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import kernighan_lin_bisection, louvain_communities
from networkx.algorithms.community.label_propagation import label_propagation_communities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
```
#### Miscellaneous functions

``` python
def newColorHex():
    color = '#'
    for i in range(6):
        color += np.random.choice(list("6789ABCD"))
    return color
```

#### Reading CSV files containing relations between people and the products they bought

``` python
edgelist_df = pd.read_csv('data/relations.csv')
people_df = pd.read_csv('data/people.csv')
names = people_df.set_index('id').T.to_dict('list')
names = {k: v[0] for k, v in names.items()}


purchases = pd.read_csv('data/purchases.csv')
products = pd.read_csv('data/products.csv').set_index('id').T.to_dict()
product_names = {k: v['product'] for k, v in products.items() if k in purchases['product_id'].unique()}
product_prices = {k: int(v['price']) for k, v in products.items() if k in purchases['product_id'].unique()}
```

### Plotting relations between people

``` python
G = nx.Graph()
G = nx.from_pandas_edgelist(edgelist_df, source='id1', target='id2')
G = nx.relabel_nodes(G, names)

rcParams['figure.figsize'] = 14, 10
pos = nx.spring_layout(G, scale=20, k=3/np.sqrt(G.order()))
d = dict(G.degree)
nx.draw(G, pos, node_color=newColorHex(),
        with_labels=True,
        nodelist=d,
        node_size=[d[k]*300 for k in d])
```

![](/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/4ebdad17cd6f75ae3a015f54cab81c4364bdf8fc.png)

### Finding communities

#### Partitioning the set of people via centraility measures (Girvan-Newman algorithm)

``` python
G = nx.Graph()
G = nx.from_pandas_edgelist(edgelist_df, source='id1', target='id2')
G = nx.relabel_nodes(G, names)

communities = girvan_newman(G)

node_groups = []
for com in next(communities):
    node_groups.append(set(com))

group_colors = [newColorHex() for _ in node_groups]

color_map = []
for node in G:
    for group_num in range(len(node_groups)):
        if node in node_groups[group_num]:
            color_map.append(group_colors[group_num])

plt.figure(5, figsize=(14, 10))
nx.draw(G, node_color=color_map, with_labels=True, node_size=2000)
```

![](/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/0535a97ddc9039aff56b542109c95d12e3658ea3.png)

#### Finding bipartitions (Kernighan-Lin bipartition algorithm)

``` python
G = nx.Graph()
G = nx.from_pandas_edgelist(edgelist_df, source='id1', target='id2')
G = nx.relabel_nodes(G, names)

communities = kernighan_lin_bisection(G)

node_groups = []
for com in communities:
    node_groups.append(com)

group_colors = [newColorHex() for _ in node_groups]

color_map = []
for node in G:
    for group_num in range(len(node_groups)):
        if node in node_groups[group_num]:
            color_map.append(group_colors[group_num])

plt.figure(5, figsize=(14, 10))
nx.draw(G, node_color=color_map, with_labels=True, node_size=2000)
```

![](/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/31699b46879260ba17e80fff9f144439d0deba91.png)

#### Finding communities using label propagation community detection algorithm

``` python
G = nx.Graph()
G = nx.from_pandas_edgelist(edgelist_df, source='id1', target='id2')
G = nx.relabel_nodes(G, names)

communities = label_propagation_communities(G)

node_groups = []
for com in communities:
    node_groups.append(com)

group_colors = [newColorHex() for _ in node_groups]

color_map = []
for node in G:
    for group_num in range(len(node_groups)):
        if node in node_groups[group_num]:
            color_map.append(group_colors[group_num])

plt.figure(5, figsize=(14, 10))
nx.draw(G, node_color=color_map, with_labels=True, node_size=2000)
```

![](/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/e185f9c8361eb7bb9a67cf9e842575e344a98fbd.png)

#### Detecting communities using the Louvain community detection algorithm

``` python
G = nx.Graph()
G = nx.from_pandas_edgelist(edgelist_df, source='id1', target='id2')
G = nx.relabel_nodes(G, names)

communities = louvain_communities(G)

node_groups = []
for com in communities:
    node_groups.append(com)

group_colors = [newColorHex() for _ in node_groups]

color_map = []
for node in G:
    for group_num in range(len(node_groups)):
        if node in node_groups[group_num]:
            color_map.append(group_colors[group_num])

plt.figure(5, figsize=(14, 10))
nx.draw(G, node_color=color_map, with_labels=True, node_size=2000)
```

![](/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/57380f6cb0a037b4e7c9f534913d491dfaa292f0.png)

### Plotting people and their purchases

``` python
G = nx.Graph()
G.add_edges_from(edgelist_df.values)
G.add_edges_from(purchases.values)

person_nodes = set(people_df.id.tolist())
product_nodes = set(purchases.product_id)

person_color = newColorHex()
product_color = newColorHex()

color_map = []
for node in G:
    if node in person_nodes:
        color_map.append(person_color)
    elif node in product_nodes:
        color_map.append(product_color)

G = nx.relabel_nodes(G, names)
G = nx.relabel_nodes(G, product_names)

plt.figure(5, figsize=(14, 10))
nx.draw(G, node_color=color_map, with_labels=True, node_size=2000)
```

![](/images/vertopal_5539c3faac5f400784bd83c73ba51ea8/2eeae7af3e4ab95d0b477f2fb4e153bad232718e.png)

## Link Prediction using Algorithms

#### Using resource allocation index

``` python
G = nx.Graph()
G.add_edges_from(edgelist_df.values)
G.add_edges_from(purchases.values)

ebunch = []
for node1 in G:
    for node2 in G:
        if node1 != node2 and (node1, node2) not in G.edges and (node2, node1) not in G.edges:
            if node1 in person_nodes and node2 in product_nodes:
                ebunch.append((node1, node2))

iterator_list = list(nx.resource_allocation_index(G, ebunch=ebunch))
iterator_list_sorted = sorted(iterator_list, key=lambda x: x[2], reverse=True)
rai_predictions = list(map(lambda x: (names[x[0]], product_names[x[1]], x[2]), iterator_list_sorted))

df_predictions = pd.DataFrame(rai_predictions, columns=['person', 'product', 'resource_allocation_index'])
df_predictions
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person</th>
      <th>product</th>
      <th>resource_allocation_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ruben Shaw</td>
      <td>Sony Alpha a7 III Mirrorless Camera</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sasha Howard</td>
      <td>Google Nest Mini (2nd Gen)</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dereon Bright</td>
      <td>Google Nest Mini (2nd Gen)</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Paola Whitaker</td>
      <td>Canon EOS Rebel T7 DSLR Camera</td>
      <td>0.366667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sasha Howard</td>
      <td>Nintendo Switch</td>
      <td>0.366667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Katrina Barr</td>
      <td>iPhone XR 64GB</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>230</th>
      <td>Katrina Barr</td>
      <td>PlayStation 5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Katrina Barr</td>
      <td>Xbox Series X</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Katrina Barr</td>
      <td>Dell XPS 13</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>Katrina Barr</td>
      <td>Garmin Forerunner 945</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p><tt>234 rows × 3 columns</tt></p>
</div>

#### Using Jaccard coefficient

``` python
G = nx.Graph()
G.add_edges_from(edgelist_df.values)
G.add_edges_from(purchases.values)

ebunch = []
for node1 in G:
    for node2 in G:
        if node1 != node2 and (node1, node2) not in G.edges and (node2, node1) not in G.edges:
            if node1 in person_nodes and node2 in product_nodes:
                ebunch.append((node1, node2))

jaccard_coefficients_list = list(nx.jaccard_coefficient(G, ebunch=ebunch))
jaccard_coefficients_sorted = sorted(jaccard_coefficients_list, key=lambda x: x[2], reverse=True)
jaccard_predictions = list(map(lambda x: (names[x[0]], product_names[x[1]], x[2]), jaccard_coefficients_sorted))

df_jaccard_predictions = pd.DataFrame(jaccard_predictions, columns=['person', 'product', 'jaccard_coefficient'])
df_jaccard_predictions
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person</th>
      <th>product</th>
      <th>jaccard_coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lilliana Keith</td>
      <td>PlayStation 5</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lilliana Keith</td>
      <td>Nintendo Switch</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Camille Khan</td>
      <td>Garmin Forerunner 945</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Kenny Kidd</td>
      <td>PlayStation 5</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Damon Webb</td>
      <td>iPhone XR 64GB</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Katrina Barr</td>
      <td>iPhone XR 64GB</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>230</th>
      <td>Katrina Barr</td>
      <td>PlayStation 5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Katrina Barr</td>
      <td>Xbox Series X</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Katrina Barr</td>
      <td>Dell XPS 13</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>Katrina Barr</td>
      <td>Garmin Forerunner 945</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p><tt>234 rows × 3 columns</tt></p>
</div>

#### Using Adamic-Adar index

``` python
G = nx.Graph()
G.add_edges_from(edgelist_df.values)
G.add_edges_from(purchases.values)

ebunch = []
for node1 in G:
    for node2 in G:
        if node1 != node2 and (node1, node2) not in G.edges and (node2, node1) not in G.edges:
            if node1 in person_nodes and node2 in product_nodes:
                ebunch.append((node1, node2))

iterator_list = list(nx.resource_allocation_index(G, ebunch=ebunch))
iterator_list_sorted = sorted(iterator_list, key=lambda x: x[2], reverse=True)
aai_predictions = list(map(lambda x: (names[x[0]], product_names[x[1]], x[2]), iterator_list_sorted))

df_predictions = pd.DataFrame(aai_predictions, columns=['person', 'product', 'adamic_adar_index'])
df_predictions
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person</th>
      <th>product</th>
      <th>adamic_adar_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ruben Shaw</td>
      <td>Sony Alpha a7 III Mirrorless Camera</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sasha Howard</td>
      <td>Google Nest Mini (2nd Gen)</td>
      <td>0.416667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dereon Bright</td>
      <td>Google Nest Mini (2nd Gen)</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Paola Whitaker</td>
      <td>Canon EOS Rebel T7 DSLR Camera</td>
      <td>0.366667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sasha Howard</td>
      <td>Nintendo Switch</td>
      <td>0.366667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Katrina Barr</td>
      <td>iPhone XR 64GB</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>230</th>
      <td>Katrina Barr</td>
      <td>PlayStation 5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Katrina Barr</td>
      <td>Xbox Series X</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Katrina Barr</td>
      <td>Dell XPS 13</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>Katrina Barr</td>
      <td>Garmin Forerunner 945</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p><tt>234 rows × 3 columns</tt></p>
</div>

#### Using common neighbour and centrality based parameterized algorithm score

``` python
G = nx.Graph()
G.add_edges_from(edgelist_df.values)
G.add_edges_from(purchases.values)

ebunch = []
for node1 in G:
    for node2 in G:
        if node1 != node2 and (node1, node2) not in G.edges and (node2, node1) not in G.edges:
            if node1 in person_nodes and node2 in product_nodes:
                ebunch.append((node1, node2))

iterator_list = list(nx.common_neighbor_centrality(G, ebunch=ebunch, alpha=0.8)) # default value alpha = 0.8
iterator_list_sorted = sorted(iterator_list, key=lambda x: x[2], reverse=True)
ccpa_predictions = list(map(lambda x: (names[x[0]], product_names[x[1]], x[2]), iterator_list_sorted))

df_predictions = pd.DataFrame(ccpa_predictions, columns=['person', 'product', 'ccpa_score'])
df_predictions
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>person</th>
      <th>product</th>
      <th>ccpa_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Paola Whitaker</td>
      <td>Nintendo Switch</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Paola Whitaker</td>
      <td>Canon EOS Rebel T7 DSLR Camera</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lilliana Keith</td>
      <td>MacBook Air M1</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lilliana Keith</td>
      <td>Nintendo Switch</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Lilliana Keith</td>
      <td>Samsung Galaxy S10</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>Yusuf Wagner</td>
      <td>Amazon Echo Dot (4th Gen)</td>
      <td>1.360000</td>
    </tr>
    <tr>
      <th>230</th>
      <td>Brennan Trevino</td>
      <td>Google Nest Mini (2nd Gen)</td>
      <td>1.360000</td>
    </tr>
    <tr>
      <th>231</th>
      <td>Brennan Trevino</td>
      <td>PlayStation 5</td>
      <td>1.360000</td>
    </tr>
    <tr>
      <th>232</th>
      <td>Brennan Trevino</td>
      <td>Amazon Echo Dot (4th Gen)</td>
      <td>1.360000</td>
    </tr>
    <tr>
      <th>233</th>
      <td>Damon Webb</td>
      <td>Amazon Echo Dot (4th Gen)</td>
      <td>1.133333</td>
    </tr>
  </tbody>
</table>
<p><tt>234 rows × 3 columns</tt>></p>
</div>
