"""
GDelt Explore
"""

import pandas as pd
import networkx as nx
import numpy as np
from itertools import chain, product
from typing import List, Tuple, Dict
from requests.adapters import HTTPAdapter
import requests
from bs4 import BeautifulSoup as bs
import networkx as nx
from itertools import product
from typing import List
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colours
import seaborn as sns
import plotly.express as px




"""
-------------------------------------------------------------------
Graph Tools
-------------------------------------------------------------------
"""

def renormalise(n, range1: List = [10, 5000], range2: List = [10, 0.3]):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


def get_edge_weights(g, w_attr="weight"):
    return[w[w_attr] for (_, _, w) in g.edges.data()]


def art_graph(text, nlp, keep_types: List[str] = None):
    g = nx.Graph()
    doc = nlp(text)
    for src, tgt in product(doc.ents, doc.ents):
        if (
            (src.label_ in keep_types)
            and (tgt.label_ in keep_types)
            and (src.text != tgt.text)
            and (src.start_char < tgt.start_char)
            and (src.text[0].isalpha())
            and (tgt.text[0].isalpha())
        ):
            edge_weight = renormalise(n=(src.start_char - tgt.start_char))
            source = src.text.replace(":", "").upper()
            target = tgt.text.replace(":", "").upper()
            g.add_node(source, e_type=src.label_)
            g.add_node(target, e_type=tgt.label_)
            g.add_edge(source, target, weight=edge_weight)
    return g


def add_url_graph(g: nx.Graph = None, text=None, nlp=None, keyword=None, keep_types: List[str] = None):

    doc = nlp(text)
    for tgt in doc.ents:
        if (
            (tgt.label_ in keep_types)
            and (keyword != tgt.text)
            and (tgt.text[0].isalpha())
        ):
            target = tgt.text.replace(":", "").upper()
            if g.has_edge(keyword, target):
                g[keyword][target]["weight"] += 0.1
            else:
                g.add_node(target, e_type=tgt.label_)
                g.add_edge(keyword, target, weight=0.1)
    return g


def kwd_g_from_df(
    df: pd.DataFrame = None,
    nlp_model: str = "en_core_web_sm",
    html_tags: Dict = None,
    keep_types: List[str] = None,
    keyword: str = None,
):
    # summ_df = summarise_df(art_df)
    if not keep_types:
        keep_types = [
            "EVENT",
            "FAC",
            "LAW",
            "LOC",
            "NORP",
            "ORG",
            "PERSON",
            "PRODUCT",
            "WORK_OF_ART",
        ]
    nlp = sp.load(nlp_model)
    master_graph = nx.Graph()
    master_graph.add_node(keyword, e_type="KEYWORD")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        text = text_from_url(url, html_tags=html_tags)
        if text:
            master_graph = add_url_graph(g=master_graph, text=text, nlp=nlp, keep_types=keep_types, keyword=keyword)
    return master_graph

def entity_graph(
    df: pd.DataFrame = None,
    nlp_model: str = "en_core_web_sm",
    html_tags: Dict = None,
    keep_types: List[str] = None,
):
    # summ_df = summarise_df(art_df)
    if not keep_types:
        keep_types = [
            "EVENT",
            "FAC",
            "LAW",
            "LOC",
            "NORP",
            "ORG",
            "PERSON",
            "PRODUCT",
            "WORK_OF_ART",
        ]
    nlp = sp.load(nlp_model)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        text = text_from_url(url, html_tags=html_tags)
        if text:
            art_g = art_graph(text, nlp=nlp, keep_types=keep_types)
            if idx == 0:
                master_g = art_g
            else:
                master_g = nx.compose(master_g, art_g)
    return master_g




"""
-------------------------------------------------------------------
GDeltDoc methods
-------------------------------------------------------------------
"""


def url_request(url):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36",
        "Connection": "keep-alive",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    try:

        response = requests.get(url, timeout=5, headers=headers, allow_redirects=False)

    except requests.exceptions.Timeout:

        try:

            response = requests.get(
                url, timeout=5, headers=headers, allow_redirects=False
            )

        except requests.exceptions.Timeout:

            response = None
    return response


def text_from_url(url, html_tags: Dict = None, parser="html.parser"):
    webpage = url_request(url)
    if webpage:
        soup = bs(webpage.content, parser)
        if html_tags:
            article = soup.find(attrs=html_tags)
        else:
            article = soup
        if article:
            text = article.get_text()
            return text
        else:
            return None


"""
-------------------------------------------------------------------
NLP methods
-------------------------------------------------------------------
"""
  

def sentiment_analyser(df, nlp_model:str = "en_core_web_sm"):
    sia = SentimentIntensityAnalyzer()
    nlp = sp.load(nlp_model)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        text = text_from_url(url)
        doc = nlp(text)
        sent_list = [sent.text for sent in doc.sents]
        art_df = pd.DataFrame(index=sent_list, columns=["sentence", "pos", "neu", "neg", "compound"])
        for sent in doc.sents:
            res = sia.polarity_scores(sent.text)
            art_df.at[sent.text, "pos"] = res["pos"]
            art_df.at[sent.text, "neu"] = res["neu"]
            art_df.at[sent.text, "neg"] = res["neg"]
            art_df.at[sent.text, "compound"] = res["compound"]
        art_df["url"] = url
        if idx == 0:
            result_df = art_df
        else:
            result_df = pd.concat([result_df, art_df], axis=0)
    return result_df


"""
-------------------------------------------------------------------
Plot methods
-------------------------------------------------------------------
"""


def plot_sentiments(df):
    _, axs = plt.subplots(2, 2)
    sns.histplot(df, x="compound", color="midnightblue", ax=axs[0, 0])
    sns.histplot(df, x="pos", color="forestgreen", ax=axs[0, 1])
    sns.histplot(df, x="neu", color="dimgrey", ax=axs[1, 0])
    sns.histplot(df, x="neg", color="firebrick", ax=axs[1, 1])
    plt.tight_layout()


def plot_c_g(g, clr):
    fig, _ = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(g, iterations=100, weight="weight")
    nx.draw(g, pos=pos, with_labels=True, node_color=clr, edge_color="gainsboro")
    fig.set_facecolor("lightsteelblue")
    plt.axis("off")
    plt.show()

def plot_all_communities(g):
    communities = nx.community.greedy_modularity_communities(g, weight="weight")
    node_groups = [list(c) for c in communities]
    for idx, c in enumerate(communities):
        for v in c:
            g.nodes[v]["community"] = idx+1

    for (v, w) in g.edges:
        if g.nodes[v]["community"] == g.nodes[w]["community"]:
            g.edges[v, w]["community"] = g.nodes[v]["community"]
        else: #mark externel edges as 0
            g.edges[v, w]["community"] = 0

    c_map = ["gainsboro"] * len(g.nodes())
    c_cols = px.colors.qualitative.Dark24
    c_cols.append("gainsboro")
    for idx, node in enumerate(g):
        for c_idx in range(len(node_groups)):
            if node in node_groups[c_idx]:
                c_map[idx] = c_cols[c_idx]
        c_map[idx]

    e_coms = [] # edge communities
    com_g_list = [nx.Graph() for _ in range(len(communities))]
    for idx in range(len(communities)):
        c_g = com_g_list[idx]
        e_coms.append([(u, v, d) for (u, v, d) in g.edges(data=True) if d["community"]== idx+1])
        c_g.add_edges_from(e_coms[idx])
        c_clr = [c_map[idx] ]* len(c_g.nodes)
        
        plot_c_g(g=c_g, clr=c_clr)


def plot_kw_graph(plot_g):
    color_dict = {
        "KEYWORD": "gainsboro",
        "EVENT": "lightcoral",
        "FAC": "orangered",
        "GPE": "peru",
        "LAW": "yellow",
        "LOC": "palegreen",
        "NORP": "darkgreen",
        "ORG": "mediumturquoise",
        "PERSON": "cornflowerblue",
        "PRODUCT": "blueviolet",
        "WORK_OF_ART": "crimson",
    }


    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.spring_layout(plot_g, iterations=100, weight="weight")
    c_list = []
    for node in dict(plot_g.nodes(data="e_type")).items():
        node_colour = color_dict[node[1]]
        c_list.append(plt_colours.to_rgba(node_colour))

    nx.draw(
        plot_g,
        pos=pos,
        with_labels=True,
        ax=ax,
        node_color=c_list,
        edge_color="gainsboro",
    )
    fig.set_facecolor("lightblue")

def map_color_list(g, color_dict, c_attr="e_type"):
    c_list = []
    for node in dict(g.nodes(data=c_attr)).items():
        node_colour = color_dict[node[1]]
        c_list.append(plt_colours.to_rgba(node_colour))
    return c_list


def plot_entity_graph(plot_g, color_attr="e_type"):
    color_dict = {
        "KEYWORD": "gainsboro",
        "EVENT": "lightcoral",
        "FAC": "orangered",
        "GPE": "peru",
        "LAW": "yellow",
        "LOC": "palegreen",
        "NORP": "darkgreen",
        "ORG": "mediumturquoise",
        "PERSON": "cornflowerblue",
        "PRODUCT": "blueviolet",
        "WORK_OF_ART": "crimson",
    }

    fig, ax = plt.subplots(figsize=(15, 15))
    #pos = nx.spring_layout(plot_g, iterations=10, weight="weight")
    pos = nx.kamada_kawai_layout(plot_g, weight="weight")
    c_list = map_color_list(plot_g, color_dict=color_dict, c_attr=color_attr)

    nx.draw(
        plot_g,
        pos=pos,
        with_labels=True,
        ax=ax,
        node_color=c_list,
        edge_color="gainsboro",
    )
    fig.set_facecolor("lightblue")