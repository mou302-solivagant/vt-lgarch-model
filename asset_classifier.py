# asset_classifier.py
ETF_MARKET_CAP = {
    "0050", "006208", "00922", "009816",
}

ETF_HIGH_DIVIDEND = {
    "0056", "00878", "00919", "00929", "00940", "00713",
}

ETF_SECTOR = {
    "00891", "00927", "00881", "00896",
}

ETF_BOND = {
    "00950B", "00984B", "00985B",
}

ETF_ACTIVE = {
    "00981A", "00982A", "00991A", "00992A", "00995A", "00994A",
}

ETF_LEVERAGED_INVERSE = {
    "00631L", "00632R", "00675L", "00676R",
}

CATEGORY_INFO = {
    "market_cap": ("大盤型", True),
    "high_dividend": ("高股息型", False),
    "sector": ("特定產業族群型", False),
    "bond": ("債券型", False),
    "active": ("主動式", False),
    "leveraged_inverse": ("槓桿反向型", False),
    "unknown_etf": ("ETF（子類未知）", False),
    "equity": ("個股", False),
    "unknown": ("未知", False),
}


def _normalize_code(ticker):
    return ticker.split(".")[0].upper()


def classify_asset(ticker, quote_type=None):
    code = _normalize_code(ticker)

    if code in ETF_MARKET_CAP:
        cat = "market_cap"
    elif code in ETF_HIGH_DIVIDEND:
        cat = "high_dividend"
    elif code in ETF_SECTOR:
        cat = "sector"
    elif code in ETF_BOND:
        cat = "bond"
    elif code in ETF_ACTIVE:
        cat = "active"
    elif code in ETF_LEVERAGED_INVERSE:
        cat = "leveraged_inverse"
    else:
        if quote_type == "EQUITY":
            cat = "equity"
        elif quote_type == "ETF":
            cat = "unknown_etf"
        else:
            cat = "unknown"

    label, allow_diversification = CATEGORY_INFO[cat]
    confidence = "confirmed" if code in (
        ETF_MARKET_CAP | ETF_HIGH_DIVIDEND | ETF_SECTOR
        | ETF_BOND | ETF_ACTIVE | ETF_LEVERAGED_INVERSE
    ) or quote_type == "EQUITY" else "unconfirmed"

    return {
        "category": cat,
        "label": label,
        "allow_diversification_wording": allow_diversification,
        "confidence": confidence,
    }


def get_neutral_term(classification):
    if classification["confidence"] == "unconfirmed" or classification["category"] == "unknown":
        return "此標的"
    return classification["label"]
