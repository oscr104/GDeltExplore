import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import tqdm as tq

from google.cloud import bigquery
from itertools import chain, product
from typing import List, Tuple



"""
-------------------------------------------------------------------
BigQuery Tools
-------------------------------------------------------------------
"""


def get_client(project_name: str = "amplified-brook-411020"):
    return bigquery.Client(project=project_name)


def sql_query(
    client=None,
    db: str = "gdelt-bq.gdeltv2.gkg",
    search_cols: List[str] = None,
    kwd_list: str = None,
    and_or:str = "AND",
):
    query = f"""SELECT * FROM {db} WHERE"""
    print(sql_query)
    for col, kwd in zip(search_cols, kwd_list):
        add = f" {col} like '%{kwd}%' {and_or}"
        print(add)
        query += add
    print("\n")
    query = query[:-3]
    query += ";"
    df = client.query(query).to_dataframe()
    print(f"{df.shape[0]} rows!")
    print(df.head().to_markdown())
    return df


"""
-------------------------------------------------------------------
Graph Tools
-------------------------------------------------------------------
"""


def cull_edges_by_weight(
    g, cull_below: float = 10, ew_col: str = "count", trim: bool = True
):
    count_list = list(
        set(chain.from_iterable(d.values() for *_, d in g.edges(data=True)))
    )
    threshold = np.percentile(count_list, cull_below)
    # filter out all edges above threshold and grab id's
    cull_edges = list(
        filter(lambda e: e[2] < threshold, (e for e in g.edges.data(ew_col)))
    )
    ce_ids = list(e[:2] for e in cull_edges)
    # remove filtered edges from graph G
    g.remove_edges_from(ce_ids)

    if trim:
        nodes = max(nx.connected_components(g), key=len)
        g = nx.subgraph(g, nodes)

    return g

def undirected_graph_from_str_col(
    df: pd.DataFrame = None,
    str_col: str = None,
    date_col: str = "DATE",
    delimiter: str = ";",
    max_nodes: int = 50,
):
    title = f"Graphing {str_col} from {df[date_col].min()} to {df[date_col].max()}"

    # Splits values in str_col into lists, then explode to additional rows
    df = df.dropna(subset=[str_col])
    df = df.drop_duplicates()
    str_col_lists = df[str_col].apply(lambda x: x.split(delimiter))
    extended_list = str_col_lists.explode()
    # Keep n most common organisations
    keepers = extended_list.value_counts().iloc[0:max_nodes].index.values

    count_df = pd.DataFrame(
        list(product(keepers, keepers)), columns=["source", "target"]
    )
    count_df["count"] = np.nan

    for source, target in tq.tqdm(list(product(keepers, keepers))):
        if source == target:
            continue
        paircount = len(
            df[
                (df[str_col].str.contains(source)) & (df[str_col].str.contains(target))
            ].index
        )
        count_df.loc[
            (count_df["source"] == source) & (count_df["target"] == target), "count"
        ] = paircount

    count_df = count_df.dropna(subset=["count"])

    g = nx.from_pandas_edgelist(count_df, edge_attr="count")

    return g


def plot_ud_graph(
    g,
    layout: str = "spring",
    spring_iterations: int = 100,
    fig_size: Tuple = [15, 10],
    ew_attr: str = "count",
    ew_rescale: Tuple = (0.1, 5),
    degree_rescale: Tuple = (50, 1000),
    node_color: str = "firebrick",
    edge_color: str = "gainsboro",
    annotate: bool = True,
) -> None:

    plt.rcParams["figure.figsize"] = fig_size


    if layout == "spring":
        pos = nx.spring_layout(g, weight=ew_attr, iterations=spring_iterations)
    else:
        pos = nx.nx_pydot.graphviz_layout(g, prog=layout)

    edge_thicc = list(nx.get_edge_attributes(g, ew_attr).values())

    scale_thic = np.interp(
        edge_thicc, (np.min(edge_thicc), np.max(edge_thicc)), ew_rescale
    )

    node_deg = [g.degree(node, weight=ew_attr) for node in g.nodes()]

    scale_deg = np.interp(
        node_deg, (np.min(node_deg), np.max(node_deg)), degree_rescale
    )


    _ = nx.draw(
        g,

        pos=pos,
        width=scale_thic,
        node_size=scale_deg,
        with_labels=annotate,
        node_color=node_color,
        edge_color=edge_color,
    )