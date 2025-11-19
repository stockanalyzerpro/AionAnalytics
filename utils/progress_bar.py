"""
progress_bar.py — Default tqdm helper (AION standard block-style)
---------------------------------------------------------------
This keeps the native tqdm look with full block bars (████),
just like:
  Normalizing: 100%|██████████████████████████████| 5537/5537 [00:01<00:00, 3400.00ticker/s]
"""

from tqdm import tqdm

def progress_bar(iterable, desc="Processing", unit="item", total=None):
    """
    Wrap any iterable with a tqdm progress bar using AION’s default visual style.
    Example:
        for sym in progress_bar(symbols, desc="Normalizing", unit="ticker"):
            ...
    """
    return tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        dynamic_ncols=True,   # adapts width automatically
        smoothing=0.1,        # keeps the rate stable
        ascii=False           # ensures Unicode block characters (████)
    )
