"""
GDelt Explore


NB:

Named entity types in Spacy

PERSON - People, including fictional
NORP - Nationalities or religious or political groups
FACILITY - Buildings, airports, highways, bridges, etc.
ORGANIZATION - Companies, agencies, institutions, etc.
GPE - Countries, cities, states
LOCATION - Non-GPE locations, mountain ranges, bodies of water
PRODUCT - Vehicles, weapons, foods, etc. (Not services)
EVENT - Named hurricanes, battles, wars, sports events, etc.
WORK OF ART - Titles of books, songs, etc.
LAW - Named documents made into laws 
LANGUAGE - Any named language 

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

nltk.download("vader_lexicon")
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


def renormalise(
    n: float = None, range1: List = [10, 5000], range2: List = [10, 0.3]
) -> float:
    """
    Renormalise n from range1 to range2
    """
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


def get_edge_weights(g=None, w_attr="weight"):
    """
    Get edge weights of g as list
    """
    return [w[w_attr] for (_, _, w) in g.edges.data()]


def art_graph(text: str = None, nlp: sp.Language = None, keep_types: List[str] = None):
    """
    Construct nx Graph from named entities in text, with edges weighted by proximity in text
    (e.g. entities mentioned near each other have higher weight edges)

    Inputs
        text: body of text from single article
        nlp: Spacy Language class to process text with
        keep_types: list of GDelt entity types to accept (see top)
    Outputs:
        g - nx Graph connecting all entities in the text, with edge weights
            proportional to the nodes' proximity in the text
    """
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


def add_url_graph(
    g=None,
    text: str = None,
    nlp: sp.Language = None,
    keyword: str = None,
    keep_types: List[str] = None,
):
    """
    Add entities from text to existing keyword star graph, with edges weighted
    according to how many times the keyword-second node pairing occurs

    Inputs
        g - nx Graph of keyword-entity pairs (star graph)
        text: body of text from single article
        nlp: Spacy Language class to process text with
        keyword: central node of g
        keep_types: list of Spacy entity types to accept (see top)
    Outputs:
        g - updated with new nodes/edges from text
    """
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
    """
    Construct 'star' graph based on keyword. Keyword is the central node,
    with all named entities in each article in df added as new nodes connected
    to the central keyword. Edges will be weighted proportionally to the number of
    times the keyword-named entity coincides in articles.

    Inputs
        df - dataframe with URLs of keyword-containing articles, output of GDeltDoc.article_search()
        nlp_model - model to load with Spacy.load()
        html-tags - html tag to use to find actual article text in URLs
                    (CURRENTLY ONLY SUPPORTS ARTICLES WITH SAME FORMAT (e.g. from same domain/using same html format))
        keep_types: list of Spacy entity types to accept (see top)
        keyword: keyword (should be same as used for GDeltDoc.article_search(), and will be central node of resulting graph)
    Outputs:
        g - star graph of keyword-named entity pairs for all articles
    """
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
            master_graph = add_url_graph(
                g=master_graph,
                text=text,
                nlp=nlp,
                keep_types=keep_types,
                keyword=keyword,
            )
    return master_graph


def entity_graph(
    df: pd.DataFrame = None,
    nlp_model: str = "en_core_web_sm",
    html_tags: Dict = None,
    keep_types: List[str] = None,
):
    """
    Construct entity graph of named entities in articles in df. Edges weighted by proximity of entities to each
    other in article text

    Inputs
        df - dataframe with URLs of keyword-containing articles, output of GDeltDoc.article_search()
        nlp_model - model to load with Spacy.load()
        html-tags - html tag to use to find actual article text in URLs
                    (CURRENTLY ONLY SUPPORTS ARTICLES WITH SAME FORMAT (e.g. from same domain/using same html format))
        keep_types: list of Spacy entity types to accept (see top)
    Outputs:
        g -  entity graph
    """
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


def url_request(url: str = None):
    """
    Helper function to get content of URL with retries & appropriate headers
    """
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


def text_from_url(
    url: str = None, html_tags: Dict = None, parser: str = "html.parser"
) -> str:
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


def sentiment_analyser(
    df: pd.DataFrame = None, nlp_model: str = "en_core_web_sm", html_tags: Dict = None
) -> pd.DataFrame:
    """
    Analyse sentiments in each sentence in each article, capture scores and store in output dataframe

    Inputs:
        df - input dataframe, from GDeltDoc.article_search()
        nlp_model - spacy nlp model to use
        html_tags - dict specifying html class/ids to find article text in
                    (Currently only supports when df URLs are all from same domain/use same HTML structure)
    Outputs:
        result_df - dataframe, each row contains the sentiment scores for a specific sentence in an article
                    columns = ["url", "sentence", "pos", "neu", "neg", "compound"]
    """
    sia = SentimentIntensityAnalyzer()
    nlp = sp.load(nlp_model)
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row["url"]
        text = text_from_url(url, html_tags=html_tags)
        if text:
            doc = nlp(text)
            sent_list = [sent.text for sent in doc.sents]
            art_df = pd.DataFrame(
                index=sent_list, columns=["sentence", "pos", "neu", "neg", "compound"]
            )
            for sent in doc.sents:
                res = sia.polarity_scores(sent.text)
                art_df.at[sent.text, "pos"] = res["pos"]
                art_df.at[sent.text, "neu"] = res["neu"]
                art_df.at[sent.text, "neg"] = res["neg"]
                art_df.at[sent.text, "compound"] = res["compound"]
            art_df["url"] = url
        if idx == 0 and text:
            result_df = art_df
        elif text:
            result_df = pd.concat([result_df, art_df], axis=0)
    return result_df


"""
-------------------------------------------------------------------
Plot methods
-------------------------------------------------------------------
"""


def plot_sentiments(df: pd.DataFrame = None) -> None:
    """
    Plot histogram of positive, negative, neutral and compound scores,
    for result of sentiment_analyser()
    """
    _, axs = plt.subplots(2, 2)
    sns.histplot(df, x="compound", color="midnightblue", ax=axs[0, 0])
    sns.histplot(df, x="pos", color="forestgreen", ax=axs[0, 1])
    sns.histplot(df, x="neu", color="dimgrey", ax=axs[1, 0])
    sns.histplot(df, x="neg", color="firebrick", ax=axs[1, 1])
    plt.tight_layout()


def plot_all_communities(g) -> None:
    """
    Find communities in g using maximum modularity algorithm.
    Plot each community on a separate figure
    """
    # Find max modularity communities in g
    communities = nx.community.greedy_modularity_communities(g, weight="weight")
    #  tag nodes & edges with their community
    for idx, c in enumerate(communities):
        for v in c:
            g.nodes[v]["community"] = idx + 1
    for v, w in g.edges:
        if g.nodes[v]["community"] == g.nodes[w]["community"]:
            g.edges[v, w]["community"] = g.nodes[v]["community"]
        else:  # mark external edges as 0
            g.edges[v, w]["community"] = 0
    # Build graph for each community, plot it
    for idx in range(len(communities)):
        nodes = (node for node, data in g.nodes(data=True) if data.get("community") == idx+1)
        subgraph = g.subgraph(nodes)
        plot_single_graph(g=subgraph)


def plot_single_graph(g) -> None:
    """
    Plot g with nodes coloured by e_type attribute
    """
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
    legend_dict = {
        "KEYWORD": "keyword",
        "EVENT": "Event",
        "FAC": "Facility",
        "GPE": "Country/city/state",
        "LAW": "Law/Statute",
        "LOC": "Location",
        "NORP": "National/political/religious group",
        "ORG": "Company/Agency/Institution",
        "PERSON": "Person",
        "PRODUCT": "Product",
        "WORK_OF_ART": "Art",
    }

    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.kamada_kawai_layout(g, weight="weight")
    c_list = []
    node_types = []
    for node in dict(g.nodes(data="e_type")).items():
        node_type = node[1]
        node_colour = color_dict[node_type]
        c_list.append(plt_colours.to_rgba(node_colour))
        if node_type not in node_types:
            node_types.append(node_type)

    nx.draw(
        g,
        pos=pos,
        with_labels=True,
        ax=ax,
        node_color=c_list,
        edge_color="gainsboro",
    )
    fig.set_facecolor("lightblue")

    lgnd_colors = [val for key, val in color_dict.items() if key in node_types]
    lgnd_text = [val for key, val in legend_dict.items() if key in node_types]
    markers = [ plt.Line2D([0, 0], [0, 0], color=color, marker="o", linestyle="") for color in lgnd_colors]
    plt.legend(markers, lgnd_text, numpoints=1)

def map_color_list(g, color_dict, c_attr="e_type"):
    """
    Construct node color list to pass to nx.draw(),
    based on color_dict and e_type attribute of g.nodes
    """
    c_list = []
    for node in dict(g.nodes(data=c_attr)).items():
        node_colour = color_dict[node[1]]
        c_list.append(plt_colours.to_rgba(node_colour))
    return c_list
