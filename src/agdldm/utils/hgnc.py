import os
import os.path as osp
import pandas as pd
from collections import defaultdict
from functools import lru_cache
from datetime import datetime

HGNC_URL = (
    "https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym"
    "&col=gd_app_name&col=gd_status&col=gd_prev_sym&col=gd_aliases&col=gd_pub_chrom_map"
    "&col=gd_pub_acc_ids&col=gd_pub_refseq_ids&col=gd_date2app_or_res&col=gd_locus_group"
    "&col=gd_locus_type&col=md_prot_id&status=Approved&status=Entry%20Withdrawn"
    "&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
)

UNKNOWN_GENE = "[UNK]"
DATA_PATH = osp.join(osp.dirname(__file__), "hgnc.csv")


def _download_hgnc_data():
    """Download and save the HGNC data if not already cached."""
    try:
        if not os.path.exists(DATA_PATH):
            response = pd.read_csv(HGNC_URL, sep="\t")
            response.to_csv(DATA_PATH, index=False)
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Failed to download HGNC data: {e}")


def load_hgnc_data(approved=True, current_date="2024-06-18"):
    """Load HGNC data from cache, optionally filtering for approved status."""
    try:
        hgnc = pd.read_csv(DATA_PATH)

    except FileNotFoundError:
        hgnc = _download_hgnc_data()

    if current_date is not None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        hgnc = hgnc[pd.to_datetime(hgnc["Date approved"]) <= pd.Timestamp(current_date)]

    if approved:
        hgnc = hgnc[hgnc["Status"] == "Approved"]
    return hgnc


@lru_cache(maxsize=2)
def create_symbol_mappings(approved=True):
    """
    Generate mappings from the HGNC DataFrame.

    Args:
        approved (bool, optional): Flag to filter only approved symbols. Defaults to True.

    Returns:
        tuple: A tuple containing two dictionaries:
            - symbol_to_hgncid: A dictionary mapping symbols to HGNC IDs.
            - prevsymbol_to_symbol: A dictionary mapping previous symbols to current symbols.
    """
    hgnc = load_hgnc_data(approved=approved)
    symbol_to_hgncid = dict()
    prevsymbol_to_symbolset = defaultdict(set)

    for i, row in hgnc.iterrows():
        if pd.notna(row["Approved symbol"]):
            symbol = row["Approved symbol"].strip()
            hgnc_id = row["HGNC ID"].strip().replace("HGNC:", "")
            symbol_to_hgncid[symbol] = hgnc_id

    for i, row in hgnc.iterrows():
        if pd.notna(row["Approved symbol"]) and pd.notna(row["Previous symbols"]):
            symbol = row["Approved symbol"].strip()
            for prev in row["Previous symbols"].split(","):
                prev = prev.strip()
                prevsymbol_to_symbolset[prev].add(symbol)

    for i, row in hgnc.iterrows():
        if pd.notna(row["Approved symbol"]) and pd.notna(row["Alias symbols"]):
            symbol = row["Approved symbol"].strip()
            for alias in row["Alias symbols"].split(","):
                alias = alias.strip()
                if alias not in prevsymbol_to_symbolset:
                    prevsymbol_to_symbolset[alias].add(symbol)

    prevsymbol_to_symbol = {
        k: list(v)[0] for k, v in prevsymbol_to_symbolset.items() if len(v) == 1
    }

    return symbol_to_hgncid, prevsymbol_to_symbol


def update_to_current_hgnc_symbol(gene_symbol, approved=True):
    """
    Updates a gene symbol to its current, HGNC-approved symbol.

    This function checks if a provided gene symbol is currently approved by the HGNC. If the symbol
    is outdated or has an alias, the function returns the current approved symbol. If the symbol is
    not recognized or has multiple possible matches, it returns a placeholder indicating an unknown gene.

    Parameters:
    gene_symbol (str): The gene symbol to be updated. Expected to be a string representing the gene symbol that may be outdated or an alias.

    Returns:
    str: The current HGNC-approved gene symbol if available; otherwise, returns a placeholder for unknown genes ('[UNK]').
    """
    SYMBOL2HGNCID, PREVSYMBOL2SYMBOL = create_symbol_mappings(approved)

    gene_symbol = gene_symbol.strip()
    if gene_symbol in SYMBOL2HGNCID:
        return gene_symbol
    if gene_symbol in PREVSYMBOL2SYMBOL:
        return PREVSYMBOL2SYMBOL[gene_symbol]
    return UNKNOWN_GENE


# Ensure data is available upon first module load
if not os.path.exists(DATA_PATH):
    _download_hgnc_data()
