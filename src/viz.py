from __future__ import annotations

import pandas as pd
import plotly.express as px


def plot_embeddings_2d(
    df: pd.DataFrame,
    output_html: str
):
    """
    Create an interactive 2D scatter plot of embeddings.

    Required columns in df:
    - x
    - y
    - paragraph_id
    - snippet
    """
    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_data={
            "paragraph_id": True,
            "snippet": True
        },
        title="Semantic Space of Moby-Dick Paragraphs (2D Projection)"
    )

    fig.write_html(output_html)
    return fig


def plot_embeddings_2d_Clusters(
    df: pd.DataFrame,
    output_html: str,
    color_col : str= "Cluster"
):
    """
    Create an interactive 2D scatter plot of embeddings with colors from learned clusters.

    Required columns in df:
    - x
    - y
    - paragraph_id
    - snippet
    """
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_data={
            "paragraph_id": True,
            "snippet": True
        },
        title="Semantic Space of Moby-Dick Paragraphs (With CLusters)"
    )

    fig.write_html(output_html)
    return fig